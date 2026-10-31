#include "soundbubble/wake.hpp"

#include "soundbubble/processing.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unordered_set>

#if defined(SOUNDBUBBLE_HAS_SHERPA)
#include <sherpa-onnx/c-api/c-api.h>
#endif
#if defined(SOUNDBUBBLE_HAS_SP)
#include <sentencepiece_processor.h>
#endif

namespace soundbubble {

struct WakeWord::Impl {
#if defined(SOUNDBUBBLE_HAS_SHERPA)
  const SherpaOnnxKeywordSpotter* kws = nullptr;
  const SherpaOnnxOnlineStream* stream = nullptr;
#endif
  std::unordered_map<std::string, std::string> people;
  int remaining = 0;
  int cooldown = 0;
};

WakeWord::WakeWord(const std::string& model_dir, const std::unordered_map<std::string, std::string>& phrases, float threshold,
                   double cooldown_seconds)
    : impl_(std::make_unique<Impl>()) {
  if (phrases.empty()) throw std::invalid_argument("Provide at least one phrase -> person ID mapping");
  if (!(threshold > 0.f && threshold <= 1.f) || cooldown_seconds < 0) {
    throw std::invalid_argument("threshold must be in (0, 1]; cooldown must be >= 0");
  }
#if !defined(SOUNDBUBBLE_HAS_SHERPA) || !defined(SOUNDBUBBLE_HAS_SP)
  (void)model_dir;
  throw std::runtime_error(
      "Wake-word support needs sherpa-onnx and sentencepiece. Configure -DSOUNDBUBBLE_WITH_SHERPA=ON "
      "-DSOUNDBUBBLE_WITH_SENTENCEPIECE=ON with both libraries installed.");
#else
  namespace fs = std::filesystem;
  auto part = [&](const char* name) {
    fs::path p = fs::path(model_dir) / (std::string(name) + "-epoch-12-avg-2-chunk-16-left-64.int8.onnx");
    if (!fs::is_regular_file(p)) throw std::runtime_error("Missing " + p.string());
    return p.string();
  };
  std::string tokens = (fs::path(model_dir) / "tokens.txt").string();
  std::string bpe = (fs::path(model_dir) / "bpe.model").string();
  if (!fs::is_regular_file(tokens) || !fs::is_regular_file(bpe)) throw std::runtime_error("Missing tokens.txt or bpe.model");
  sentencepiece::SentencePieceProcessor sp;
  if (!sp.Load(bpe).ok()) throw std::runtime_error("failed to load bpe.model");
  std::unordered_set<std::string> vocab;
  {
    std::ifstream in(tokens);
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      auto pos = line.find_last_of(" \t");
      vocab.insert(pos == std::string::npos ? line : line.substr(0, pos));
    }
  }
  auto lines = build_keyword_lines(phrases, [&](const std::string& phrase) {
    std::vector<std::string> pieces;
    sp.Encode(phrase, &pieces);
    return pieces;
  }, vocab);
  fs::path kw = fs::temp_directory_path() / "soundbubble-keywords.txt";
  {
    std::ofstream out(kw);
    for (const auto& line : lines) {
      out << line.line << "\n";
      impl_->people[line.label] = line.person;
    }
  }
  std::string enc = part("encoder");
  std::string dec = part("decoder");
  std::string joi = part("joiner");
  SherpaOnnxKeywordSpotterConfig config;
  std::memset(&config, 0, sizeof(config));
  config.feat_config.sample_rate = 16000;
  config.feat_config.feature_dim = 80;
  config.model_config.transducer.encoder = enc.c_str();
  config.model_config.transducer.decoder = dec.c_str();
  config.model_config.transducer.joiner = joi.c_str();
  config.model_config.tokens = tokens.c_str();
  config.model_config.num_threads = 1;
  config.model_config.provider = "cpu";
  config.max_active_paths = 4;
  config.num_trailing_blanks = 1;
  config.keywords_score = 1.0f;
  config.keywords_threshold = threshold;
  config.keywords_file = kw.string().c_str();
  impl_->kws = SherpaOnnxCreateKeywordSpotter(&config);
  if (!impl_->kws) throw std::runtime_error("SherpaOnnxCreateKeywordSpotter failed");
  impl_->stream = SherpaOnnxCreateKeywordStream(impl_->kws);
  impl_->cooldown = static_cast<int>(cooldown_seconds * 16000);
#endif
}

WakeWord::~WakeWord() {
#if defined(SOUNDBUBBLE_HAS_SHERPA)
  if (impl_ && impl_->stream) SherpaOnnxDestroyOnlineStream(impl_->stream);
  if (impl_ && impl_->kws) SherpaOnnxDestroyKeywordSpotter(impl_->kws);
#endif
}

void WakeWord::reset() {
#if defined(SOUNDBUBBLE_HAS_SHERPA)
  if (impl_->stream) SherpaOnnxDestroyOnlineStream(impl_->stream);
  impl_->stream = SherpaOnnxCreateKeywordStream(impl_->kws);
  impl_->remaining = 0;
#endif
}

std::optional<std::string> WakeWord::feed(const float* samples, int n) {
  if (n <= 0 || n > 3200) throw std::invalid_argument("Feed 1-D mono blocks of 1..3200 samples at 16 kHz");
#if !defined(SOUNDBUBBLE_HAS_SHERPA)
  (void)samples;
  return std::nullopt;
#else
  if (impl_->remaining) {
    int skipped = std::min(impl_->remaining, n);
    impl_->remaining -= skipped;
    samples += skipped;
    n -= skipped;
    if (n <= 0) return std::nullopt;
  }
  SherpaOnnxOnlineStreamAcceptWaveform(impl_->stream, 16000, samples, n);
  while (SherpaOnnxIsKeywordStreamReady(impl_->kws, impl_->stream)) {
    SherpaOnnxDecodeKeywordStream(impl_->kws, impl_->stream);
    const SherpaOnnxKeywordResult* result = SherpaOnnxGetKeywordResult(impl_->kws, impl_->stream);
    std::string label = result && result->keyword ? result->keyword : "";
    SherpaOnnxDestroyKeywordResult(result);
    if (!label.empty()) {
      SherpaOnnxResetKeywordStream(impl_->kws, impl_->stream);
      impl_->remaining = impl_->cooldown;
      auto it = impl_->people.find(label);
      if (it == impl_->people.end()) return std::nullopt;
      return it->second;
    }
  }
  return std::nullopt;
#endif
}

WakeWordWorker::WakeWordWorker(const std::string& model_dir, const std::unordered_map<std::string, std::string>& phrases,
                               float threshold, double cooldown_seconds, int queue_capacity_blocks)
    : detector_(std::make_unique<WakeWord>(model_dir, phrases, threshold, cooldown_seconds)),
      queue_(std::max(1, queue_capacity_blocks)) {}

WakeWordWorker::~WakeWordWorker() { stop(); }

void WakeWordWorker::start(int input_sr, std::function<void(const std::string&)> trigger_cb,
                           std::function<void(const std::string&)> error_cb) {
  if (thread_.joinable()) return;
  trigger_cb_ = std::move(trigger_cb);
  error_cb_ = std::move(error_cb);
  stop_ = false;
  thread_ = std::thread([this, input_sr] { run(input_sr); });
}

void WakeWordWorker::stop() {
  stop_ = true;
  queue_.close();
  if (thread_.joinable()) thread_.join();
}

void WakeWordWorker::push_input_block(const float* mono, int n) {
  if (n <= 0) return;
  queue_.push(std::vector<float>(mono, mono + n));
}

WakeWordWorker::Stats WakeWordWorker::stats() const {
  std::lock_guard<std::mutex> lock(mu_);
  Stats s;
  s.queue_drops = queue_.stats.dropped_oldest;
  s.feed_calls = feed_calls_;
  s.feed_mean_ms = feed_total_ms_ / std::max(1, feed_calls_);
  s.trigger_count = trigger_count_;
  s.last_person = last_person_;
  return s;
}

void WakeWordWorker::run(int input_sr) {
  try {
    StreamingResampler src(input_sr, 16000, 1);
    bool copy = input_sr == 16000;
    std::vector<float> acc;
    while (!stop_.load()) {
      auto item = queue_.pop(0.10);
      if (!item) continue;
      std::vector<float> x16 = copy ? *item : src.process(item->data(), static_cast<int>(item->size()));
      if (x16.empty()) continue;
      for (float& v : x16) {
        if (!std::isfinite(v)) v = 0;
        v = std::clamp(v, -1.f, 1.f);
      }
      acc.insert(acc.end(), x16.begin(), x16.end());
      while (!acc.empty()) {
        int n = std::min(3200, static_cast<int>(acc.size()));
        auto t0 = std::chrono::steady_clock::now();
        auto person = detector_->feed(acc.data(), n);
        acc.erase(acc.begin(), acc.begin() + n);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        {
          std::lock_guard<std::mutex> lock(mu_);
          feed_total_ms_ += ms;
          feed_calls_ += 1;
          if (person) {
            trigger_count_ += 1;
            last_person_ = *person;
          }
        }
        if (person && trigger_cb_) trigger_cb_(*person);
      }
    }
  } catch (const std::exception& exc) {
    if (error_cb_) error_cb_(std::string("wake-word worker disabled after error: ") + exc.what());
  }
}

}  // namespace soundbubble
