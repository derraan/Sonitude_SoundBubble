#include "soundbubble/wake.hpp"

#include "soundbubble/json_mini.hpp"
#include "soundbubble/processing.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace soundbubble {

DryHistory::DryHistory(int capacity_frames)
    : capacity_(std::max(1, capacity_frames)), buf_(static_cast<size_t>(capacity_), 0.f) {}

std::int64_t DryHistory::oldest_index() const { return std::max<std::int64_t>(0, write_index_ - capacity_); }

void DryHistory::clear() {
  std::fill(buf_.begin(), buf_.end(), 0.f);
  write_index_ = 0;
}

void DryHistory::write(const float* mono, int n_total) {
  if (n_total <= 0) return;
  int n = n_total;
  const float* x = mono;
  std::vector<float> tail;
  if (n >= capacity_) {
    x = mono + (n_total - capacity_);
    n = capacity_;
  }
  int start = static_cast<int>(write_index_ % capacity_);
  int end = start + n;
  if (end <= capacity_) {
    std::copy(x, x + n, buf_.begin() + start);
  } else {
    int n1 = capacity_ - start;
    std::copy(x, x + n1, buf_.begin() + start);
    std::copy(x + n1, x + n, buf_.begin());
  }
  write_index_ += n_total;
}

std::pair<std::vector<float>, double> DryHistory::read_fractional(double start_index, double rate, int n_frames) const {
  int n = std::max(0, n_frames);
  if (n == 0) return {{}, start_index};
  if (!std::isfinite(start_index)) start_index = static_cast<double>(oldest_index());
  if (!std::isfinite(rate) || rate <= 0) rate = 1;
  std::vector<float> out(static_cast<size_t>(n));
  std::int64_t oldest = oldest_index();
  std::int64_t newest = write_index_;
  for (int i = 0; i < n; ++i) {
    double pos = start_index + rate * static_cast<double>(i);
    auto idx0 = static_cast<std::int64_t>(std::floor(pos));
    float frac = static_cast<float>(pos - static_cast<double>(idx0));
    auto idx1 = idx0 + 1;
    float a = 0, b = 0;
    if (idx0 >= oldest && idx0 < newest) a = buf_[static_cast<size_t>(idx0 % capacity_)];
    if (idx1 >= oldest && idx1 < newest) b = buf_[static_cast<size_t>(idx1 % capacity_)];
    out[static_cast<size_t>(i)] = a * (1.f - frac) + b * frac;
  }
  return {out, start_index + rate * n};
}

BubbleOpenController::BubbleOpenController(int output_sr, float base_mix_dry, float replay_sec, float replay_rate, float hold_sec,
                                           float attack_ms, float release_ms)
    : output_sr_(output_sr),
      base_mix_dry_(std::clamp(base_mix_dry, 0.f, 1.f)),
      replay_frames_(std::max(0, static_cast<int>(std::llround(replay_sec * output_sr)))),
      replay_rate_(std::max(1.f, replay_rate)),
      hold_frames_total_(std::max(0, static_cast<int>(std::llround(hold_sec * output_sr)))),
      attack_frames_(std::max(1, static_cast<int>(std::llround((attack_ms / 1000.f) * output_sr)))),
      release_frames_(std::max(1, static_cast<int>(std::llround((release_ms / 1000.f) * output_sr)))),
      state_(STATE_IDLE),
      mix_current_(base_mix_dry_),
      mix_target_(base_mix_dry_) {}

void BubbleOpenController::trigger(const std::string& person) {
  std::lock_guard<std::mutex> lock(mu_);
  pending_trigger_ = true;
  if (!person.empty()) last_person_ = person;
}

void BubbleOpenController::set_base_mix(float mix_dry) {
  std::lock_guard<std::mutex> lock(mu_);
  base_mix_dry_ = std::clamp(mix_dry, 0.f, 1.f);
  if (state_ == STATE_IDLE && !pending_trigger_) {
    mix_current_ = base_mix_dry_;
    mix_target_ = base_mix_dry_;
    mix_frames_left_ = 0;
  }
}

void BubbleOpenController::set_mix_target_locked(float target, int duration_frames) {
  mix_target_ = std::clamp(target, 0.f, 1.f);
  mix_frames_left_ = std::max(0, duration_frames);
  if (mix_frames_left_ == 0) mix_current_ = mix_target_;
}

void BubbleOpenController::activate_trigger_locked(const DryHistory& history) {
  pending_trigger_ = false;
  trigger_count_ += 1;
  hold_frames_left_ = hold_frames_total_;
  if (state_ == STATE_IDLE) {
    state_ = STATE_OPEN_REPLAY;
    replay_pos_ = static_cast<double>(std::max(history.oldest_index(), history.write_index() - replay_frames_));
    set_mix_target_locked(1.f, attack_frames_);
  } else {
    state_ = STATE_OPEN_LIVE;
    if (mix_current_ < 1.f || mix_target_ < 1.f) set_mix_target_locked(1.f, attack_frames_);
  }
}

std::vector<float> BubbleOpenController::next_mix_locked(int n_frames) {
  int n = std::max(0, n_frames);
  if (n <= 0) return {};
  if (mix_frames_left_ <= 0) return std::vector<float>(static_cast<size_t>(n), mix_current_);
  int step_n = std::min(n, mix_frames_left_);
  std::vector<float> ramp(static_cast<size_t>(step_n));
  for (int i = 0; i < step_n; ++i) {
    if (step_n == 1) ramp[static_cast<size_t>(i)] = mix_current_;
    else ramp[static_cast<size_t>(i)] = mix_current_ + (mix_target_ - mix_current_) * (static_cast<float>(i) / static_cast<float>(step_n - 1));
  }
  mix_current_ = ramp.back();
  mix_frames_left_ -= step_n;
  if (mix_frames_left_ == 0) mix_current_ = mix_target_;
  if (step_n == n) return ramp;
  std::vector<float> tail(static_cast<size_t>(n - step_n), mix_current_);
  ramp.insert(ramp.end(), tail.begin(), tail.end());
  return ramp;
}

std::pair<std::vector<float>, std::vector<float>> BubbleOpenController::process_block(const float* live_dry, int n, const DryHistory& history) {
  if (n <= 0) return {{}, {}};
  std::lock_guard<std::mutex> lock(mu_);
  if (pending_trigger_) activate_trigger_locked(history);
  std::vector<float> dry_src;
  if (state_ == STATE_OPEN_REPLAY) {
    auto got = history.read_fractional(replay_pos_, replay_rate_, n);
    dry_src = std::move(got.first);
    replay_pos_ = got.second;
    if (replay_pos_ >= static_cast<double>(history.write_index())) state_ = STATE_OPEN_LIVE;
  } else {
    dry_src.assign(live_dry, live_dry + n);
  }
  if (state_ == STATE_OPEN_REPLAY || state_ == STATE_OPEN_LIVE) {
    hold_frames_left_ = std::max(0, hold_frames_left_ - n);
    if (hold_frames_left_ == 0) {
      state_ = STATE_RELEASING;
      set_mix_target_locked(base_mix_dry_, release_frames_);
    }
  }
  auto mix = next_mix_locked(n);
  if (state_ == STATE_RELEASING && mix_frames_left_ == 0) state_ = STATE_IDLE;
  return {dry_src, mix};
}

BubbleOpenController::Status BubbleOpenController::status() const {
  std::lock_guard<std::mutex> lock(mu_);
  Status s;
  s.state = state_;
  s.mix_current = mix_current_;
  s.mix_target = mix_target_;
  s.hold_frames_left = hold_frames_left_;
  s.trigger_count = trigger_count_;
  s.last_person = last_person_;
  return s;
}

std::unordered_map<std::string, std::string> parse_phrase_entries(const std::vector<std::string>& entries) {
  std::unordered_map<std::string, std::string> phrases;
  for (const auto& raw : entries) {
    auto eq = raw.find('=');
    if (eq == std::string::npos) throw std::invalid_argument("Invalid wake phrase entry; expected phrase=person_id");
    auto trim = [](std::string s) {
      size_t a = 0;
      while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
      size_t b = s.size();
      while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
      return s.substr(a, b - a);
    };
    std::string phrase = trim(raw.substr(0, eq));
    std::string person = trim(raw.substr(eq + 1));
    if (phrase.empty() || person.empty()) throw std::invalid_argument("phrase and person_id are required");
    phrases[phrase] = person;
  }
  return phrases;
}

std::unordered_map<std::string, std::string> load_phrase_book_json(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Wake phrase book not found: " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  Json j = parse_json(ss.str());
  if (j.type != Json::Type::Object) throw std::runtime_error("Wake phrase book must be a JSON object");
  const Json* phrases = j.find("phrases");
  if (!phrases || phrases->type != Json::Type::Object) throw std::runtime_error("Wake phrase book must contain object key \"phrases\"");
  std::unordered_map<std::string, std::string> out;
  for (const auto& kv : phrases->o) {
    if (kv.first.empty() || kv.second.type != Json::Type::String || kv.second.s.empty()) {
      throw std::runtime_error("Phrase keys and person IDs must be non-empty strings");
    }
    out[kv.first] = kv.second.s;
  }
  return out;
}

std::unordered_map<std::string, std::string> build_phrases(const std::string& phrase_book_path, const std::vector<std::string>& cli_entries) {
  std::unordered_map<std::string, std::string> phrases;
  if (!phrase_book_path.empty()) phrases = load_phrase_book_json(phrase_book_path);
  auto cli = parse_phrase_entries(cli_entries);
  for (auto& kv : cli) phrases[kv.first] = std::move(kv.second);
  return phrases;
}

std::vector<KeywordLine> build_keyword_lines(const std::unordered_map<std::string, std::string>& phrases,
                                             const std::function<std::vector<std::string>(const std::string&)>& encode,
                                             const std::unordered_set<std::string>& vocabulary) {
  std::vector<std::pair<std::string, std::string>> ordered(phrases.begin(), phrases.end());
  std::sort(ordered.begin(), ordered.end());
  std::vector<KeywordLine> lines;
  std::unordered_set<std::string> seen;
  int index = 0;
  for (const auto& kv : ordered) {
    std::string normalized;
    std::istringstream iss(kv.first);
    std::string word;
    while (iss >> word) {
      for (char& c : word) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
      if (!normalized.empty()) normalized.push_back(' ');
      normalized += word;
    }
    if (normalized.empty() || kv.second.empty() || seen.count(normalized)) {
      throw std::invalid_argument("Use distinct nonempty phrases and nonempty string person IDs");
    }
    seen.insert(normalized);
    auto tokens = encode(normalized);
    if (tokens.empty()) throw std::invalid_argument("Model cannot encode phrase: " + kv.first);
    for (const auto& tok : tokens) {
      if (tok == "<unk>" || !vocabulary.count(tok)) throw std::invalid_argument("Model cannot encode phrase: " + kv.first);
    }
    KeywordLine line;
    line.label = "PERSON_" + std::to_string(index);
    line.person = kv.second;
    std::ostringstream oss;
    for (size_t i = 0; i < tokens.size(); ++i) {
      if (i) oss << ' ';
      oss << tokens[i];
    }
    oss << " @" << line.label;
    line.line = oss.str();
    lines.push_back(std::move(line));
    ++index;
  }
  return lines;
}

WakeRuntime::WakeRuntime(int input_sr, int output_sr, int history_frames, float base_mix, float replay_sec, float replay_rate,
                         float hold_sec, float release_ms, const std::string& model_dir,
                         const std::unordered_map<std::string, std::string>& phrases, float threshold)
    : history_(std::make_unique<DryHistory>(history_frames)),
      ctrl_(std::make_unique<BubbleOpenController>(output_sr, base_mix, replay_sec, replay_rate, hold_sec, 20.f, release_ms)) {
  worker_ = std::make_unique<WakeWordWorker>(model_dir, phrases, threshold);
  worker_->start(input_sr, [this](const std::string& person) { ctrl_->trigger(person); },
                 [](const std::string& msg) { /* logged by caller via exceptions during feed */ (void)msg; });
  enabled_ = true;
}

WakeRuntime::~WakeRuntime() { stop(); }

void WakeRuntime::stop() {
  if (worker_) worker_->stop();
}

void WakeRuntime::push_input(const float* mono, int n) {
  if (worker_) worker_->push_input_block(mono, n);
}

void WakeRuntime::set_base_mix(float mix_dry) {
  if (ctrl_) ctrl_->set_base_mix(mix_dry);
}

std::pair<std::vector<float>, std::vector<float>> WakeRuntime::mix(const float* live_dry, int n) {
  history_->write(live_dry, n);
  return ctrl_->process_block(live_dry, n, *history_);
}

std::string WakeRuntime::profile_suffix() const {
  if (!ctrl_ || !worker_) return {};
  auto st = ctrl_->status();
  auto ks = worker_->stats();
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(2);
  oss << "  wake_state=" << st.state << "  wake_triggers=" << st.trigger_count << "  kws_queue_drops=" << ks.queue_drops
      << "  kws_feed_ms=" << ks.feed_mean_ms;
  return oss.str();
}

}  // namespace soundbubble
