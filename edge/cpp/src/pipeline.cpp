#include "soundbubble/pipeline.hpp"

#include "soundbubble/audio_io.hpp"
#include "soundbubble/mixctl.hpp"
#include "soundbubble/processing.hpp"
#include "soundbubble/wake.hpp"
#include "soundbubble/wav.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>

namespace soundbubble {

class DryPath {
 public:
  DryPath(int capacity, int input_sr, int output_sr)
      : ring_(1, capacity), src_(input_sr, output_sr, 1) {}
  void clear() {
    ring_.clear();
    src_.reset();
  }
  int drop_oldest_frames(int n) { return ring_.drop_oldest_frames(n); }
  std::vector<float> read_block(int n) {
    std::vector<float> mono(static_cast<size_t>(std::max(0, n)), 0.f);
    if (n <= 0) return mono;
    std::vector<float> tmp(static_cast<size_t>(n), 0.f);
    ring_.read_into(tmp.data(), n);
    return tmp;
  }
  int push_input_block(const float* mono, int n) {
    auto y = src_.process(mono, n);
    if (y.empty()) return 0;
    return ring_.write_drop_oldest(y.data(), static_cast<int>(y.size()));
  }
  const std::string& backend() const { return src_.backend(); }

 private:
  OutputRingBuffer ring_;
  StreamingResampler src_;
};

// In-memory WAV tap buffers. Append from the worker (or duplex callback); flush on stop/EOS.
class TapRecorder {
 public:
  struct Select {
    bool raw = false;
    bool mapped = true;
    bool dry = true;
    bool wet = true;
  };

  static Select parse_select(const std::string& text) {
    Select s{false, false, false, false};
    std::string token;
    std::istringstream iss(text);
    while (std::getline(iss, token, ',')) {
      // trim spaces
      size_t b = token.find_first_not_of(" \t");
      size_t e = token.find_last_not_of(" \t");
      if (b == std::string::npos) continue;
      token = token.substr(b, e - b + 1);
      if (token == "raw") s.raw = true;
      else if (token == "mapped") s.mapped = true;
      else if (token == "dry") s.dry = true;
      else if (token == "wet") s.wet = true;
      else if (token == "all") {
        s.raw = s.mapped = s.dry = s.wet = true;
      } else {
        throw std::runtime_error("unknown --tap-select token: " + token + " (want raw,mapped,dry,wet)");
      }
    }
    if (!s.raw && !s.mapped && !s.dry && !s.wet) {
      s.mapped = s.dry = s.wet = true;  // plan default: 6ch + dry + wet
    }
    return s;
  }

  TapRecorder(std::string dir, Select sel, double max_seconds, double max_mb, int input_sr, int output_sr, int raw_ch,
              int mapped_ch)
      : dir_(std::move(dir)),
        sel_(sel),
        max_seconds_(std::max(0.0, max_seconds)),
        max_bytes_(static_cast<std::uint64_t>(std::max(0.0, max_mb) * 1024.0 * 1024.0)),
        input_sr_(std::max(1, input_sr)),
        output_sr_(std::max(1, output_sr)),
        raw_ch_(std::max(1, raw_ch)),
        mapped_ch_(std::max(1, mapped_ch)),
        armed_(true) {
    namespace fs = std::filesystem;
    fs::create_directories(dir_);
    // Pre-reserve to keep appends allocation-light on the worker / callback path.
    const size_t in_frames = max_seconds_ > 0 ? static_cast<size_t>(max_seconds_ * input_sr_) + 1 : 0;
    const size_t out_frames = max_seconds_ > 0 ? static_cast<size_t>(max_seconds_ * output_sr_) + 1 : 0;
    if (sel_.raw && in_frames > 0) raw_.reserve(in_frames * static_cast<size_t>(raw_ch_));
    if (sel_.mapped && in_frames > 0) mapped_.reserve(in_frames * static_cast<size_t>(mapped_ch_));
    if (sel_.dry && out_frames > 0) dry_.reserve(out_frames);
    if (sel_.wet && out_frames > 0) wet_.reserve(out_frames);
    std::cout << "[tap] armed dir=" << dir_ << " select="
              << (sel_.raw ? "raw," : "") << (sel_.mapped ? "mapped," : "") << (sel_.dry ? "dry," : "")
              << (sel_.wet ? "wet" : "") << " max_seconds=" << max_seconds_ << " max_mb=" << (max_mb) << "\n";
  }

  bool armed() const { return armed_.load(std::memory_order_relaxed); }
  bool want_raw() const { return sel_.raw; }
  bool want_mapped() const { return sel_.mapped; }
  bool want_dry() const { return sel_.dry; }
  bool want_wet() const { return sel_.wet; }

  void arm() { set_armed(true); }
  void disarm() { set_armed(false); }
  void toggle() { set_armed(!armed()); }

  void set_armed(bool on) {
    armed_.store(on, std::memory_order_relaxed);
    if (on) capped_ = false;  // re-arm after a cap hit allows a fresh segment
    std::cout << "[tap] " << (on ? "armed" : "disarmed") << "\n";
  }

  // Clear buffers and point at a new directory without writing WAVs.
  void retarget(const std::string& dir) {
    std::lock_guard<std::mutex> lock(flush_mu_);
    {
      std::lock_guard<std::mutex> lk(buf_mu_);
      raw_.clear();
      mapped_.clear();
      dry_.clear();
      wet_.clear();
      bytes_ = 0;
      capped_ = false;
      armed_.store(false, std::memory_order_relaxed);
    }
    dir_ = dir;
    written_.clear();
    std::filesystem::create_directories(dir_);
    std::cout << "[tap] dir=" << dir_ << " (cleared, disarmed)\n";
  }

  const std::string& dir() const { return dir_; }

  void append_raw_interleaved(const float* data, int frames, int channels) {
    if (!armed() || !sel_.raw || capped_ || frames <= 0 || !data) return;
    std::lock_guard<std::mutex> lk(buf_mu_);
    if (!armed()) return;
    append_interleaved(raw_, data, frames, channels, input_sr_);
  }

  // planar: channel-major [c * frames + t]
  void append_mapped_planar(const float* planar, int channels, int frames) {
    if (!armed() || !sel_.mapped || capped_ || frames <= 0 || channels <= 0 || !planar) return;
    std::lock_guard<std::mutex> lk(buf_mu_);
    if (!armed()) return;
    if (check_cap(static_cast<size_t>(frames) * static_cast<size_t>(channels) * sizeof(float), frames, input_sr_)) return;
    const size_t n = static_cast<size_t>(frames) * static_cast<size_t>(channels);
    const size_t old = mapped_.size();
    mapped_.resize(old + n);
    for (int t = 0; t < frames; ++t) {
      for (int c = 0; c < channels; ++c) {
        mapped_[old + static_cast<size_t>(t * channels + c)] = planar[static_cast<size_t>(c * frames + t)];
      }
    }
  }

  void append_dry(const float* mono, int frames) {
    if (!armed() || !sel_.dry || capped_ || frames <= 0 || !mono) return;
    std::lock_guard<std::mutex> lk(buf_mu_);
    if (!armed()) return;
    append_interleaved(dry_, mono, frames, 1, output_sr_);
  }

  void append_wet(const float* mono, int frames) {
    if (!armed() || !sel_.wet || capped_ || frames <= 0 || !mono) return;
    std::lock_guard<std::mutex> lk(buf_mu_);
    if (!armed()) return;
    append_interleaved(wet_, mono, frames, 1, output_sr_);
  }

  // A flush closes the segment: disarm + swap buffers out atomically, then write
  // outside the buffer lock (the audio worker never waits on disk I/O). Without
  // the disarm, frames kept accumulating between "TAP flush" and "TAP disarm"
  // and a later flush (e.g. pipeline stop) overwrote the take's WAVs.
  void flush() {
    std::lock_guard<std::mutex> lock(flush_mu_);
    std::vector<float> raw, mapped, dry, wet;
    {
      std::lock_guard<std::mutex> lk(buf_mu_);
      const bool was_armed = armed_.exchange(false, std::memory_order_relaxed);
      if (was_armed) std::cout << "[tap] disarmed (flush closes segment)\n";
      raw.swap(raw_);
      mapped.swap(mapped_);
      dry.swap(dry_);
      wet.swap(wet_);
      bytes_ = 0;
      // Keep capped_ so a post-cap flush does not reopen logging until toggle/re-arm.
    }
    write_one("raw_8ch.wav", input_sr_, raw_ch_, raw, sel_.raw);
    write_one("mapped_6ch.wav", input_sr_, mapped_ch_, mapped, sel_.mapped);
    write_one("dry_ref.wav", output_sr_, 1, dry, sel_.dry);
    write_one("wet_out.wav", output_sr_, 1, wet, sel_.wet);
  }

 private:
  bool check_cap(size_t add_bytes, int add_frames, int sr) {
    if (capped_) return true;
    const std::uint64_t next_bytes = bytes_ + static_cast<std::uint64_t>(add_bytes);
    if (max_bytes_ > 0 && next_bytes > max_bytes_) {
      stop_cap("max-mb");
      return true;
    }
    // Duration cap uses the longest stream so far + this append at its sample rate.
    if (max_seconds_ > 0 && sr > 0) {
      const double raw_sec = raw_ch_ > 0 ? static_cast<double>(raw_.size() / static_cast<size_t>(raw_ch_)) / input_sr_ : 0.0;
      const double mapped_sec =
          mapped_ch_ > 0 ? static_cast<double>(mapped_.size() / static_cast<size_t>(mapped_ch_)) / input_sr_ : 0.0;
      const double dry_sec = static_cast<double>(dry_.size()) / output_sr_;
      const double wet_sec = static_cast<double>(wet_.size()) / output_sr_;
      const double this_sec = static_cast<double>(add_frames) / static_cast<double>(sr);
      const double longest = std::max({raw_sec, mapped_sec, dry_sec, wet_sec}) + this_sec;
      if (longest > max_seconds_) {
        stop_cap("max-seconds");
        return true;
      }
    }
    bytes_ = next_bytes;
    return false;
  }

  void stop_cap(const char* why) {
    if (capped_) return;
    capped_ = true;
    armed_.store(false, std::memory_order_relaxed);
    std::cout << "[tap] stopped: cap reached (" << why << ")\n";
  }

  void append_interleaved(std::vector<float>& buf, const float* data, int frames, int channels, int sr) {
    const size_t n = static_cast<size_t>(frames) * static_cast<size_t>(channels);
    if (check_cap(n * sizeof(float), frames, sr)) return;
    buf.insert(buf.end(), data, data + n);
  }

  void write_one(const char* name, int sr, int ch, const std::vector<float>& data, bool enabled) {
    if (!enabled || data.empty()) return;
    // Never clobber a file this recorder already wrote into the current dir:
    // later segments land in <stem>.partN.wav so a take's WAVs stay intact.
    std::string out_name = name;
    int& seg = written_[name];
    seg += 1;
    if (seg > 1) {
      std::string stem = out_name.substr(0, out_name.rfind('.'));
      out_name = stem + ".part" + std::to_string(seg) + ".wav";
      std::cout << "[tap] " << name << " already written in this dir; saving segment as " << out_name << "\n";
    }
    const std::filesystem::path final_path = std::filesystem::path(dir_) / out_name;
    write_wav_atomic(final_path.string(), sr, ch, data);
    std::cout << "[tap] wrote " << final_path.string()
              << " frames=" << (data.size() / static_cast<size_t>(std::max(1, ch))) << "\n";
  }

  std::string dir_;
  Select sel_;
  double max_seconds_ = 120.0;
  std::uint64_t max_bytes_ = 0;
  int input_sr_ = 24000;
  int output_sr_ = 48000;
  int raw_ch_ = 8;
  int mapped_ch_ = 6;
  std::atomic<bool> armed_{false};
  bool capped_ = false;
  std::uint64_t bytes_ = 0;
  std::vector<float> raw_;
  std::vector<float> mapped_;
  std::vector<float> dry_;
  std::vector<float> wet_;
  std::mutex flush_mu_;
  std::mutex buf_mu_;  // guards raw_/mapped_/dry_/wet_/bytes_ (worker appends vs flush)
  std::map<std::string, int> written_;  // files written into dir_ since last retarget
};

namespace {

double now_sec() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

Engine::Engine(PipelineConfig cfg)
    : cfg_(std::move(cfg)), capture_(std::max(1, cfg_.max_capture_blocks)) {}

Engine::~Engine() {
  request_stop("destroy");
  if (worker_.joinable()) worker_.join();
  if (wake_) wake_->stop();
  if (mix_) mix_->stop();
  try {
    flush_taps();
  } catch (...) {
  }
}

void Engine::set_inferencer_for_test(std::unique_ptr<Inferencer> inf) { infer_ = std::move(inf); }

void Engine::request_stop(const std::string& reason) {
  bool expected = false;
  if (stop_.compare_exchange_strong(expected, true)) {
    signal_stop_ = reason.rfind("signal:", 0) == 0;
    std::cout << "[shutdown] " << reason << "\n";
    capture_.close();
  }
}

void Engine::toggle_taps() {
  if (taps_) taps_->toggle();
}

void Engine::request_tap_flush() {
  if (!taps_) return;
  tap_flush_req_.store(true, std::memory_order_relaxed);
}

bool Engine::taps_armed() const { return taps_ && taps_->armed(); }

void Engine::arm_taps() {
  if (taps_) taps_->arm();
}

void Engine::disarm_taps() {
  if (taps_) taps_->disarm();
}

void Engine::retarget_tap_dir(const std::string& dir) {
  if (dir.empty()) {
    std::cerr << "[tap] retarget ignored: empty dir\n";
    return;
  }
  cfg_.tap_dir = dir;
  if (cfg_.tap_select.empty()) cfg_.tap_select = "raw,mapped,dry,wet";
  if (!prepared_) {
    std::cout << "[tap] dir queued until prepare: " << dir << "\n";
    return;
  }
  if (!taps_) {
    auto sel = TapRecorder::parse_select(cfg_.tap_select);
    taps_ = std::make_unique<TapRecorder>(dir, sel, cfg_.tap_max_seconds, cfg_.tap_max_mb, cfg_.input_sr,
                                          cfg_.output_sr, cfg_.input_channels,
                                          static_cast<int>(selected_.empty() ? 6 : selected_.size()));
    taps_->disarm();
    std::cout << "[tap] dir=" << dir << " (cleared, disarmed)\n";
  } else {
    taps_->retarget(dir);
  }
}

void Engine::handle_control_line(const std::string& line) {
  // Trim CR/LF and surrounding whitespace.
  std::string s = line;
  while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || s.back() == ' ' || s.back() == '\t')) s.pop_back();
  size_t b = s.find_first_not_of(" \t");
  if (b == std::string::npos) return;
  s = s.substr(b);

  auto lower = [](std::string t) {
    for (char& c : t) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return t;
  };
  std::string head = s;
  size_t sp = s.find_first_of(" \t");
  std::string cmd = lower(sp == std::string::npos ? s : s.substr(0, sp));
  std::string rest = sp == std::string::npos ? std::string() : s.substr(sp + 1);
  while (!rest.empty() && (rest.front() == ' ' || rest.front() == '\t')) rest.erase(rest.begin());

  if (cmd == "stop") {
    request_stop("signal:STOP");
    return;
  }
  if (cmd == "tap") {
    size_t sp2 = rest.find_first_of(" \t");
    std::string verb = lower(sp2 == std::string::npos ? rest : rest.substr(0, sp2));
    std::string arg = sp2 == std::string::npos ? std::string() : rest.substr(sp2 + 1);
    while (!arg.empty() && (arg.front() == ' ' || arg.front() == '\t')) arg.erase(arg.begin());
    if (verb == "arm") {
      arm_taps();
      return;
    }
    if (verb == "disarm" || verb == "off") {
      disarm_taps();
      return;
    }
    if (verb == "flush") {
      request_tap_flush();
      return;
    }
    if (verb == "toggle") {
      toggle_taps();
      return;
    }
    if (verb == "dir" || verb == "path") {
      if (arg.empty()) {
        std::cerr << "[tap] TAP dir requires a path\n";
        return;
      }
      // Strip optional quotes.
      if (arg.size() >= 2 && ((arg.front() == '"' && arg.back() == '"') || (arg.front() == '\'' && arg.back() == '\''))) {
        arg = arg.substr(1, arg.size() - 2);
      }
      retarget_tap_dir(arg);
      return;
    }
    std::cerr << "[tap] unknown TAP cmd (want arm|disarm|flush|toggle|dir <path>)\n";
    return;
  }
  std::cerr << "[ctrl] unknown command: " << head << "\n";
}

void Engine::flush_taps() {
  if (!taps_) return;
  taps_->flush();
}

void Engine::poll_mix_tap_cmds() {
  if (!mix_ || !taps_) return;
  switch (mix_->take_tap_cmd()) {
    case TapUartCmd::On:
      taps_->arm();
      break;
    case TapUartCmd::Off:
      taps_->disarm();
      break;
    case TapUartCmd::Toggle:
      taps_->toggle();
      break;
    case TapUartCmd::Flush:
      request_tap_flush();
      break;
    case TapUartCmd::None:
    default:
      break;
  }
}

void Engine::maybe_flush_taps() {
  poll_mix_tap_cmds();
  if (!tap_flush_req_.exchange(false, std::memory_order_relaxed)) return;
  try {
    flush_taps();
    std::cout << "[tap] flush complete\n";
  } catch (const std::exception& exc) {
    std::cerr << "[tap] flush failed: " << exc.what() << "\n";
  }
}

void Engine::tap_dry_wet_then_gain(std::vector<float>& mono, const float* dry_or_null) {
  if (taps_ && taps_->armed() && !mono.empty()) {
    const int n = static_cast<int>(mono.size());
    if (taps_->want_dry() && dry_or_null) taps_->append_dry(dry_or_null, n);
    if (taps_->want_wet()) taps_->append_wet(mono.data(), n);
  }
  apply_output_gain_and_limit(mono);
}

float Engine::current_mix_dry() const { return mix_ ? mix_->dry() : cfg_.mix_dry; }

int Engine::output_ahead_frames() const { return output_ring_ ? output_ring_->ahead_frames() : 0; }

void Engine::apply_contract_and_open_model() {
  if (prepared_) return;
  bool zoo = !cfg_.zoo_dir.empty();
  if (zoo && !cfg_.bubble_radius) throw std::runtime_error("--zoo-dir requires --bubble-radius");
  if (cfg_.model.empty() && !zoo && !infer_) throw std::runtime_error("Provide --model or --zoo-dir + --bubble-radius");
  if (!cfg_.model.empty() && zoo) {
    std::cout << "[zoo] --model given explicitly; ignoring --zoo-dir\n";
  } else if (zoo && cfg_.model.empty()) {
    auto pick = resolve_zoo_model(cfg_.zoo_dir, *cfg_.bubble_radius);
    cfg_.model = pick.onnx_path;
    cfg_.contract_path = pick.contract_path;
    std::cout << "[zoo] selected radius " << *cfg_.bubble_radius << "\n";
  }
  if (!cfg_.contract_path.empty()) {
    contract_ = load_contract(cfg_.contract_path);
    have_contract_ = true;
    cfg_.model_sr = contract_.model_sr;
    cfg_.model_num_ch = contract_.model_num_ch;
    cfg_.model_chunk = contract_.model_chunk;
    cfg_.model_pad = contract_.model_pad;
    if (cfg_.channel_map == "0,1,2,3,4,5" && !contract_.expected_channel_map.empty()) {
      std::ostringstream oss;
      for (size_t i = 0; i < contract_.expected_channel_map.size(); ++i) {
        if (i) oss << ',';
        oss << contract_.expected_channel_map[i];
      }
      cfg_.channel_map = oss.str();
    }
    if (cfg_.input_channels == 8) cfg_.input_channels = contract_.expected_input_channels;
  }
  RealtimeContract rt;
  rt.max_capture_blocks = cfg_.max_capture_blocks;
  rt.max_output_ahead_sec = cfg_.max_output_ahead_sec;
  rt.target_output_ahead_sec = cfg_.target_output_ahead_sec;
  rt.rt_fifo_priority = cfg_.rt_fifo_priority;
  rt.validate();
  if (rt.rt_fifo_priority > 0) {
    auto res = try_set_realtime_fifo(rt.rt_fifo_priority);
    std::cout << (res.applied ? "[rt] applied " : "[warn] rt scheduling not applied: ") << res.message << "\n";
  }
  try_lock_memory();

  model_frame_len_ = cfg_.model_chunk + cfg_.model_pad;
  int model_io = cfg_.model_chunk * std::max(1, cfg_.hops_per_io);
  input_block_ = static_cast<int>(std::llround(static_cast<double>(model_io) * cfg_.input_sr / cfg_.model_sr));
  output_block_ = static_cast<int>(std::llround(static_cast<double>(model_io) * cfg_.output_sr / cfg_.model_sr));
  if (input_block_ <= 0 || output_block_ <= 0) throw std::runtime_error("invalid block size");
  // DAC callback size independent of hop quantum. Auto ≈ 2 ms (sr/500), at most one hop.
  if (cfg_.output_device_block > 0) {
    out_dev_block_ = std::clamp(cfg_.output_device_block, 1, output_block_);
  } else {
    int min_frames = std::max(1, cfg_.output_sr / 500);  // ~2 ms
    out_dev_block_ = std::clamp(output_block_ / 4, min_frames, output_block_);
  }
  // Shared duplex framesPerBuffer when Rapid (worker/callback). Unused for Normal.
  if (cfg_.duplex_mode == "callback") {
    duplex_block_ = input_block_;
  } else if (cfg_.duplex_block > 0) {
    duplex_block_ = std::clamp(cfg_.duplex_block, 1, input_block_);
  } else {
    duplex_block_ = out_dev_block_;
  }
  selected_ = parse_channel_map(cfg_.channel_map, cfg_.input_channels, cfg_.model_num_ch);
  if (cfg_.monitor_input_channel < 0 || cfg_.monitor_input_channel >= cfg_.input_channels) {
    throw std::runtime_error("--monitor-input-channel out of range");
  }
  ring_params_ = compute_output_ring_params(cfg_.output_sr, output_block_, cfg_.max_output_ahead_sec, cfg_.target_output_ahead_sec);
  int ring_cap = std::max(1, static_cast<int>(std::llround((cfg_.model_ring_ms / 1000.0) * cfg_.model_sr)));
  model_ring_ = std::make_unique<AudioRingBuffer>(cfg_.model_num_ch, ring_cap);
  output_ring_ = std::make_unique<OutputRingBuffer>(cfg_.output_channels, ring_params_.ring_n_frames);
  dry_ = std::make_unique<DryPath>(ring_params_.ring_n_frames, cfg_.input_sr, cfg_.output_sr);
  dc_ = std::make_unique<DCOffsetFilter>(cfg_.model_num_ch, cfg_.alpha);
  levels_ = std::make_unique<ChannelLevelValidator>(cfg_.model_num_ch, cfg_.silence_dbfs, cfg_.fault_min_dbfs, cfg_.fault_max_dbfs,
                                                    cfg_.silence_window_frames);
  in_src_ = std::make_unique<StreamingResampler>(cfg_.input_sr, cfg_.model_sr, cfg_.model_num_ch);
  out_src_ = std::make_unique<StreamingResampler>(cfg_.model_sr, cfg_.output_sr, 1);
  if (cfg_.require_soxr && (in_src_->backend() != "soxr-stream" || out_src_->backend() != "soxr-stream")) {
    throw std::runtime_error("Streaming SoXR backend required but unavailable.");
  }
  std::cout << "[src] input=" << in_src_->backend() << " output=" << out_src_->backend() << "\n";
  current_frame_.assign(static_cast<size_t>(cfg_.model_num_ch * model_frame_len_), 0.f);
  last_output_chunk_.assign(static_cast<size_t>(cfg_.model_chunk), 0.f);
  auto phrases = build_phrases(cfg_.wake_phrases_json, cfg_.wake_phrase);
  if (!phrases.empty()) {
    std::string model_dir = cfg_.wake_model_dir.empty() ? std::string("models/kws") : cfg_.wake_model_dir;
    try {
      wake_ = std::make_unique<WakeRuntime>(cfg_.input_sr, cfg_.output_sr, std::max(ring_params_.ring_n_frames, cfg_.output_sr * 3),
                                            cfg_.mix_dry, cfg_.wake_replay_sec, cfg_.wake_replay_rate, cfg_.wake_hold_sec,
                                            cfg_.wake_release_ms, model_dir, phrases, cfg_.wake_threshold);
      std::cout << "[wake] enabled\n";
    } catch (const std::exception& exc) {
      wake_.reset();
      std::cout << "[wake] disabled after init failure: " << exc.what() << "\n";
    }
  }
  if (!cfg_.mix_uart_port.empty()) {
    mix_ = std::make_unique<MixControl>(cfg_.mix_uart_port, cfg_.mix_uart_baud, cfg_.mix_dry);
    mix_->start();
  }
  if (cfg_.output_limiter) {
    out_limiter_.configure({cfg_.limiter_ceiling, cfg_.limiter_release_ms, cfg_.limiter_hold_ms},
                           static_cast<std::uint32_t>(std::max(1, cfg_.output_sr)));
  }
  if (cfg_.wet_gate_dbfs > -150.f) {
    ResidualGateConfig gc;
    gc.threshold_dbfs = cfg_.wet_gate_dbfs;
    wet_gate_.configure(gc, static_cast<std::uint32_t>(std::max(1, cfg_.output_sr)));
  }
  if (!cfg_.tap_dir.empty() && !taps_) {
    auto sel = TapRecorder::parse_select(cfg_.tap_select);
    taps_ = std::make_unique<TapRecorder>(cfg_.tap_dir, sel, cfg_.tap_max_seconds, cfg_.tap_max_mb, cfg_.input_sr,
                                          cfg_.output_sr, cfg_.input_channels, static_cast<int>(selected_.size()));
  }
  if (!infer_) {
    infer_ = make_onnx_inferencer(cfg_.model, cfg_.intra_op_threads, cfg_.inter_op_threads, have_contract_ ? &contract_ : nullptr,
                                  cfg_.bubble_radius);
  }
  if (cfg_.ort_warmup_runs > 0) {
    // First ORT runs allocate arenas / pick kernels and can take 10–100× a
    // steady hop. Doing that while the DAC already pulls frames is the
    // cold-start underrun burst. Warm on silence, then zero recurrent state.
    std::vector<float> zeros(static_cast<size_t>(cfg_.model_num_ch * model_frame_len_), 0.f);
    double first_ms = 0, last_ms = 0;
    for (int i = 0; i < cfg_.ort_warmup_runs; ++i) {
      auto h = infer_->run(zeros.data(), cfg_.model_num_ch, model_frame_len_);
      if (i == 0) first_ms = h.infer_ms;
      last_ms = h.infer_ms;
    }
    infer_->reset();
    infer_ema_ms_ = last_ms;
    std::cout << "[ort] warmed " << cfg_.ort_warmup_runs << " runs (first=" << first_ms << "ms last=" << last_ms << "ms)\n";
  }
  prepared_ = true;
  std::cout << "=== Sound Bubble ONNX Live Pipeline ===\n";
  std::cout << "model=" << cfg_.model << "\n";
  std::cout << "input_block=" << input_block_ << ", output_block=" << output_block_
            << ", output_device_block=" << out_dev_block_ << ", model_chunk=" << cfg_.model_chunk << "\n";
  std::cout << "duplex_mode=" << (cfg_.duplex_mode.empty() ? "off" : cfg_.duplex_mode)
            << ", duplex_block=" << duplex_block_ << "\n";
  std::cout << "output_gain_db=" << cfg_.output_gain_db
            << " limiter=" << (cfg_.output_limiter ? "on" : "off")
            << " ceiling=" << cfg_.limiter_ceiling
            << " release_ms=" << cfg_.limiter_release_ms << "\n";
  if (!cfg_.mic_gains.empty()) {
    std::cout << "mic_gains(dev ch, linear)=";
    for (size_t i = 0; i < cfg_.mic_gains.size(); ++i) std::cout << (i ? "," : "") << cfg_.mic_gains[i];
    std::cout << "\n";
  }
}

void Engine::reset_stream_timeline(bool clear_io_rings, bool clear_model_ring) {
  discontinuities_ += 1;
  in_src_->reset();
  out_src_->reset();
  infer_->reset();
  std::fill(current_frame_.begin(), current_frame_.end(), 0.f);
  std::fill(last_output_chunk_.begin(), last_output_chunk_.end(), 0.f);
  // Soft gaps keep already-resampled hops so the next block can still infer.
  // Only a model-ring overflow (hard) discards them.
  if (clear_model_ring) model_ring_->clear();
  if (clear_io_rings) {
    output_ring_->clear();
    dry_->clear();
  }
  if (cfg_.output_limiter) out_limiter_.reset();
}

void Engine::apply_output_gain_and_limit(std::vector<float>& mono) {
  if (mono.empty()) return;
  // Residual expander first (pure-wet product path only; taps already captured
  // pre-gate), then output gain, then peak limiter (ceiling guards DAC).
  if (wet_gate_.configured() && current_mix_dry() <= 0.f && !wake_) {
    wet_gate_.process_mono(mono);
  }
  const float g = dbfs_to_linear(cfg_.output_gain_db);
  if (g != 1.f) {
    for (float& v : mono) v *= g;
  }
  if (cfg_.output_limiter) {
    out_limiter_.process_mono(mono);
  } else {
    for (float& v : mono) v = std::clamp(v, -1.f, 1.f);
  }
}

void Engine::declick_fill_edges(std::vector<float>& y_out, std::vector<std::uint8_t>& fill_mask, int frames) {
  // Silent fill spliced into a live wet stream is a hard step at both edges;
  // at block rate that is a buzz. Ramp audio down into each fill run and back
  // up out of it (fill frames are already zero).
  if (frames <= 0) return;
  fill_mask.resize(static_cast<size_t>(frames), 0);
  const int C = cfg_.output_channels;
  const int K = kRingDeclickFrames;
  auto scale = [&](int t, float g) {
    for (int c = 0; c < C; ++c) y_out[static_cast<size_t>(t * C + c)] *= g;
  };
  // Edge against what is already queued in the ring.
  if (fill_mask[0] && !last_write_was_fill_) output_ring_->fade_tail(K);
  for (int t = 0; t < frames; ++t) {
    const bool prev_fill = (t == 0) ? last_write_was_fill_ : fill_mask[static_cast<size_t>(t - 1)] != 0;
    const bool cur_fill = fill_mask[static_cast<size_t>(t)] != 0;
    if (prev_fill && !cur_fill) {  // fill -> audio: fade in
      for (int j = 0; j < K && t + j < frames && !fill_mask[static_cast<size_t>(t + j)]; ++j)
        scale(t + j, static_cast<float>(j + 1) / static_cast<float>(K + 1));
    } else if (!prev_fill && cur_fill && t > 0) {  // audio -> fill: fade out
      for (int j = 0; j < K && t - 1 - j >= 0 && !fill_mask[static_cast<size_t>(t - 1 - j)]; ++j)
        scale(t - 1 - j, static_cast<float>(j) / static_cast<float>(K));
    }
  }
  last_write_was_fill_ = fill_mask.back() != 0;
}

std::vector<float> Engine::mix_wet_with_dry(const std::vector<float>& wet, const float* live_dry_or_null, int n) {
  if (n <= 0) return {};
  std::vector<float> live;
  const float* live_ptr = live_dry_or_null;
  if (!live_ptr) {
    live = dry_->read_block(n);
    live_ptr = live.data();
  }
  if (wake_) {
    if (mix_) wake_->set_base_mix(mix_->dry());
    auto mixed = wake_->mix(live_ptr, n);
    std::vector<float> y(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) y[static_cast<size_t>(i)] = (1.f - mixed.second[static_cast<size_t>(i)]) * wet[static_cast<size_t>(i)] +
                                                           mixed.second[static_cast<size_t>(i)] * mixed.first[static_cast<size_t>(i)];
    return y;
  }
  float a = mix_ ? mix_->dry() : cfg_.mix_dry;
  if (a <= 0.f) return std::vector<float>(wet.begin(), wet.begin() + n);
  std::vector<float> y(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) y[static_cast<size_t>(i)] = (1.f - a) * wet[static_cast<size_t>(i)] + a * live_ptr[i];
  return y;
}

bool Engine::handle_item(const CaptureItem& item, int drained) {
  const int C = item.channels;
  const int T = C > 0 ? static_cast<int>(item.interleaved.size()) / C : 0;
  double t0 = now_sec();
  double capture_age_ms = (t0 - item.t_capture) * 1000.0;
  int capture_gap = 0;
  if (last_capture_idx_) capture_gap = static_cast<int>(std::max<std::int64_t>(0, static_cast<std::int64_t>(item.capture_idx - *last_capture_idx_ - 1)));
  // Latest-wins (adaptive/latest) routinely drains 1–2 older capture blocks when the
  // worker trails the PortAudio callback — especially Windows two-stream WDM-KS.
  // Treating every mild drain as a hard discontinuity forced permanent shed +
  // reset_stream_timeline every block, which zeroed wet / model_out.
  // Hard discontinuity = stale capture, or more than a mild skip (true loss).
  constexpr int kMildSkipMax = 2;
  const int skipped_hint = std::max(std::max(0, drained), std::max(0, capture_gap));
  bool disc = false;
  if (cfg_.queue_policy == "adaptive" && capture_age_ms > cfg_.max_capture_age_ms) disc = true;
  if (skipped_hint > kMildSkipMax) disc = true;
  last_capture_age_ms_ = capture_age_ms;
  last_capture_gap_ = capture_gap;
  last_capture_drained_ = drained;
  drained_total_ += static_cast<std::uint64_t>(std::max(0, drained));
  std::uint64_t underrun_now = output_ring_->underrun_frames();
  int recent_underruns = static_cast<int>(underrun_now - last_underrun_frames_);
  last_underrun_frames_ = underrun_now;
  BudgetInput bin;
  bin.queue_policy = cfg_.queue_policy;
  bin.mode = mode_;
  bin.hops_per_io = cfg_.hops_per_io;
  bin.max_steps_per_block = cfg_.max_steps_per_block;
  bin.recover_steps_per_block = cfg_.recover_steps_per_block;
  bin.recover_ahead_multiplier = cfg_.recover_ahead_multiplier;
  bin.max_capture_age_ms = cfg_.max_capture_age_ms;
  bin.block_budget_ms = 1000.0 * static_cast<double>(input_block_) / static_cast<double>(cfg_.input_sr);
  bin.infer_ema_ms = infer_ema_ms_;
  bin.dc_ema_ms = dc_ema_ms_;
  bin.level_ema_ms = level_ema_ms_;
  bin.frame_to_infer_ema_ms = frame_ema_ms_;
  bin.post_ema_ms = post_ema_ms_;
  bin.out_ahead_ms = 1000.0 * static_cast<double>(output_ring_->ahead_frames()) / static_cast<double>(cfg_.output_sr);
  bin.target_ahead_ms = 1000.0 * static_cast<double>(ring_params_.ring_target_ahead_frames) / static_cast<double>(cfg_.output_sr);
  bin.capture_age_ms = capture_age_ms;
  bin.capture_discontinuity = disc;
  bin.model_has_chunk = model_ring_->size() >= cfg_.model_chunk;
  bin.recent_underrun_frames = recent_underruns;
  bin.output_block_ms = 1000.0 * static_cast<double>(output_block_) / static_cast<double>(cfg_.output_sr);
  bin.max_ahead_ms = 1000.0 * static_cast<double>(ring_params_.ring_max_ahead_frames) / static_cast<double>(cfg_.output_sr);
  bin.adaptive_extra_ms = adaptive_extra_ms_;
  auto decision = decide_budget(bin);
  adaptive_extra_ms_ = decision.adaptive_extra_ms;
  last_sustainable_steps_ = decision.sustainable_steps;
  mode_ = decision.mode;
  last_capture_idx_ = item.capture_idx;

  std::vector<float> planar(static_cast<size_t>(C * T), 0.f);
  for (int c = 0; c < C; ++c)
    for (int t = 0; t < T; ++t) planar[static_cast<size_t>(c * T + t)] = item.interleaved[static_cast<size_t>(t * C + c)];
  // Per-mic calibration trims by device channel (raw tap above reads item.interleaved, untrimmed).
  for (int c = 0; c < C && c < static_cast<int>(cfg_.mic_gains.size()); ++c) {
    const float g = cfg_.mic_gains[static_cast<size_t>(c)];
    if (g == 1.f) continue;
    for (int t = 0; t < T; ++t) planar[static_cast<size_t>(c * T + t)] *= g;
  }
  std::vector<float> xref(static_cast<size_t>(T));
  for (int t = 0; t < T; ++t) xref[static_cast<size_t>(t)] = planar[static_cast<size_t>(cfg_.monitor_input_channel * T + t)];
  std::vector<float> selected(static_cast<size_t>(selected_.size() * T));
  for (size_t i = 0; i < selected_.size(); ++i) {
    int c = selected_[i];
    std::copy(planar.begin() + static_cast<size_t>(c * T), planar.begin() + static_cast<size_t>((c + 1) * T),
              selected.begin() + static_cast<size_t>(i * T));
  }
  if (taps_ && taps_->armed()) {
    if (taps_->want_raw()) taps_->append_raw_interleaved(item.interleaved.data(), T, C);
    if (taps_->want_mapped()) taps_->append_mapped_planar(selected.data(), static_cast<int>(selected_.size()), T);
  }
  auto fitted = fit_channels(selected.data(), static_cast<int>(selected_.size()), T, cfg_.model_num_ch);
  double t_dc = now_sec();
  auto x_dc = dc_->process(fitted.data(), T);
  double dc_ms = (now_sec() - t_dc) * 1000.0;
  double t_lv = now_sec();
  auto valid = levels_->validate(x_dc.data(), T);
  double level_ms = (now_sec() - t_lv) * 1000.0;
  // Silence/FAULT stay on pre-gain (USB/mic health). Monitor ch* levels match Python:
  // recompute after input gain so they show what the model hears.
  latest_statuses_ = valid.statuses;
  float gain = dbfs_to_linear(cfg_.input_gain_db);
  if (gain != 1.f) {
    for (float& v : xref) v *= gain;
  }
  dry_->push_input_block(xref.data(), T);
  if (wake_) wake_->push_input(xref.data(), T);
  if (gain != 1.f) {
    // Float headroom: do not hard-clip before the model; output limiter guards the DAC.
    for (float& v : valid.audio) v *= gain;
  }
  latest_levels_dbfs_.assign(static_cast<size_t>(cfg_.model_num_ch), -120.f);
  if (T > 0) {
    for (int c = 0; c < cfg_.model_num_ch; ++c) {
      double acc = 0;
      for (int t = 0; t < T; ++t) {
        float v = valid.audio[static_cast<size_t>(c * T + t)];
        acc += static_cast<double>(v) * static_cast<double>(v);
      }
      double rms = std::sqrt(acc / static_cast<double>(T) + 1e-12);
      latest_levels_dbfs_[static_cast<size_t>(c)] =
          static_cast<float>(20.0 * std::log10(std::max(rms, 1e-10)));
    }
  }
  auto x_model = in_src_->process(valid.audio.data(), T);
  int model_frames = x_model.empty() ? 0 : static_cast<int>(x_model.size()) / cfg_.model_num_ch;
  auto before = model_ring_->stats.dropped_samples;
  if (model_frames > 0) model_ring_->push(x_model.data(), cfg_.model_num_ch, model_frames);
  if (model_ring_->stats.dropped_samples > before) {
    // Push already drop-oldest. Do NOT reset infer/resamplers — that zeros
    // recurrent state mid-phrase and is the wet-path pop/buzz when keep opens.
    model_ring_drops_ += 1;
  }

  double infer_start = now_sec();
  std::vector<float> y_chunks;
  std::vector<double> infer_times;
  int steps = 0;
  while (steps < decision.step_budget) {
    std::vector<float> step;
    if (!model_ring_->pop(cfg_.model_chunk, step)) break;
    int hop = cfg_.model_chunk;
    int frame = model_frame_len_;
    int keep = frame - hop;
    for (int c = 0; c < cfg_.model_num_ch; ++c) {
      float* row = current_frame_.data() + static_cast<size_t>(c * frame);
      std::move(row + hop, row + frame, row);
      std::copy(step.begin() + static_cast<size_t>(c * hop), step.begin() + static_cast<size_t>((c + 1) * hop), row + keep);
    }
    auto hop_out = infer_->run(current_frame_.data(), cfg_.model_num_ch, frame);
    infer_times.push_back(hop_out.infer_ms);
    latest_distance_status_ = hop_out.distance_status;
    latest_distance_m_ = hop_out.distance_m;
    latest_count_status_ = hop_out.count_status;
    latest_speaker_count_ = hop_out.speaker_count;
    latest_speaker_distances_ = hop_out.speaker_distances;
    int take = std::min(hop, static_cast<int>(hop_out.samples.size()));
    y_chunks.insert(y_chunks.end(), hop_out.samples.begin(), hop_out.samples.begin() + take);
    if (take < hop) y_chunks.insert(y_chunks.end(), static_cast<size_t>(hop - take), 0.f);
    steps += 1;
  }
  double frame_to_infer_ms = (infer_start - t0) * 1000.0;
  double t_post = now_sec();
  int steps_run = static_cast<int>(infer_times.size());
  int missing = std::max(0, cfg_.hops_per_io - steps_run);
  int frames_per_hop_out = static_cast<int>(std::llround(static_cast<double>(cfg_.output_sr) * (static_cast<double>(cfg_.model_chunk) / cfg_.model_sr)));
  std::vector<float> y_out;
  // 1 = silent fill frame (missing hop / pad / catch-up lead), 0 = audio.
  std::vector<std::uint8_t> fill_mask;
  bool pending_reset = disc;
  if (!y_chunks.empty()) {
    model_out_db_ = 0.8 * model_out_db_ + 0.2 * static_cast<double>(rms_dbfs(y_chunks.data(), static_cast<int>(y_chunks.size())));
    auto y_io = out_src_->process(y_chunks.data(), static_cast<int>(y_chunks.size()));
    const int n_out = static_cast<int>(y_io.size());
    // Read dry once so mix and dry_ref tap share the same reference (pre-output_gain).
    auto dry_live = dry_->read_block(n_out);
    auto mixed = mix_wet_with_dry(y_io, dry_live.data(), n_out);
    tap_dry_wet_then_gain(mixed, dry_live.data());
    output_db_ = 0.8 * output_db_ + 0.2 * static_cast<double>(rms_dbfs(mixed.data(), static_cast<int>(mixed.size())));
    y_out = to_multichannel(mixed.data(), static_cast<int>(mixed.size()), cfg_.output_channels);
    fill_mask.assign(static_cast<size_t>(mixed.size()), 0);
  }
  auto policy = parse_overload_policy(cfg_.overload_policy);
  // Product path (mix_dry=0): never splice dry into the wet DAC stream on
  // missing hops — that alternates dry/wet at block rate and buzzes when keep opens.
  const bool splice_dry = (policy == OverloadPolicy::DryFill) && (cfg_.mix_dry > 0.f || wake_ != nullptr);
  if (missing > 0 && policy != OverloadPolicy::Strict) {
    int missing_frames = missing * frames_per_hop_out;
    std::vector<float> fill;
    if (splice_dry) {
      auto dry_fill = dry_->read_block(missing_frames);
      auto mixed = mix_wet_with_dry(std::vector<float>(static_cast<size_t>(missing_frames), 0.f), dry_fill.data(), missing_frames);
      if (!wake_) mixed = dry_fill;
      tap_dry_wet_then_gain(mixed, dry_fill.data());
      fill = to_multichannel(mixed.data(), missing_frames, cfg_.output_channels);
    } else {
      // Keep tap clocks moving with silence on wet; still consume dry for alignment.
      auto dry_fill = dry_->read_block(missing_frames);
      std::vector<float> silent(static_cast<size_t>(missing_frames), 0.f);
      tap_dry_wet_then_gain(silent, dry_fill.data());
      fill.assign(static_cast<size_t>(missing_frames * cfg_.output_channels), 0.f);
    }
    y_out.insert(y_out.end(), fill.begin(), fill.end());
    fill_mask.insert(fill_mask.end(), static_cast<size_t>(missing_frames), splice_dry ? 0 : 1);
  }
  if (policy != OverloadPolicy::Strict) {
    auto fit_exact = [&](std::vector<float>& y, int frames_needed) {
      int have = y.empty() ? 0 : static_cast<int>(y.size()) / std::max(1, cfg_.output_channels);
      if (have == frames_needed) return;
      short_block_writes_ += 1;
      std::vector<float> fixed(static_cast<size_t>(frames_needed * cfg_.output_channels), 0.f);
      int ncopy = std::min(have, frames_needed);
      if (ncopy > 0) {
        std::copy(y.begin(), y.begin() + static_cast<size_t>(ncopy * cfg_.output_channels), fixed.begin());
      }
      if (have < frames_needed) {
        int gap = frames_needed - have;
        auto dry_gap = dry_->read_block(gap);
        if (splice_dry) {
          tap_dry_wet_then_gain(dry_gap, dry_gap.data());
          auto fill = to_multichannel(dry_gap.data(), gap, cfg_.output_channels);
          std::copy(fill.begin(), fill.end(), fixed.begin() + static_cast<size_t>(have * cfg_.output_channels));
        } else {
          std::vector<float> silent(static_cast<size_t>(gap), 0.f);
          tap_dry_wet_then_gain(silent, dry_gap.data());
        }
      }
      y.swap(fixed);
    };
    fit_exact(y_out, output_block_);
    fill_mask.resize(static_cast<size_t>(output_block_), splice_dry ? 0 : 1);
    // Each skipped capture index already advanced the output clock: queue drops
    // show up as capture_gap, and pop_latest collapses show up as drained.
    // Write one output block per skipped index so those periods are not holes.
    int skipped = std::max(std::max(0, drained), std::max(0, capture_gap));
    int room = std::max(1, ring_params_.ring_max_ahead_frames / std::max(1, output_block_));
    int catchup = std::min(skipped, std::max(0, room - 1));
    if (catchup > 0) {
      int extra = catchup * output_block_;
      std::vector<float> lead;
      if (splice_dry) {
        auto dry_lead = dry_->read_block(extra);
        tap_dry_wet_then_gain(dry_lead, dry_lead.data());
        lead = to_multichannel(dry_lead.data(), extra, cfg_.output_channels);
      } else {
        auto dry_lead = dry_->read_block(extra);
        std::vector<float> silent(static_cast<size_t>(extra), 0.f);
        tap_dry_wet_then_gain(silent, dry_lead.data());
        lead.assign(static_cast<size_t>(extra * cfg_.output_channels), 0.f);
      }
      y_out.insert(y_out.begin(), lead.begin(), lead.end());
      fill_mask.insert(fill_mask.begin(), static_cast<size_t>(extra), splice_dry ? 0 : 1);
    }
  }
  if (!y_out.empty()) {
    int frames = static_cast<int>(y_out.size()) / cfg_.output_channels;
    declick_fill_edges(y_out, fill_mask, frames);
    int dropped = output_ring_->write_drop_oldest(y_out.data(), frames);
    out_dropped_ += static_cast<std::uint64_t>(std::max(0, dropped));
    int ahead = output_ring_->ahead_frames();
    // Trim toward the effective target (+ 1/4 block of slack). A full block of
    // slack (~8 ms) made a 9 ms target unreachable (ring stuck at 16 ms).
    int target = static_cast<int>(decision.effective_target_ahead_ms * cfg_.output_sr / 1000.0);
    int slack = std::max(1, output_block_ / 4);
    int trim_threshold = std::min(ring_params_.ring_max_ahead_frames, target + slack);
    // Never trim while recovering from underruns — that fights the cushion.
    if (recent_underruns == 0 && ahead > trim_threshold) {
      int trim = ahead - target;
      int d2 = output_ring_->drop_oldest_frames(trim);
      if (d2 > 0) {
        out_dropped_ += static_cast<std::uint64_t>(d2);
        dry_->drop_oldest_frames(d2);
      }
    }
  }
  if (pending_reset) reset_stream_timeline(false, false);
  double post_ms = (now_sec() - t_post) * 1000.0;
  double e2e_ms = (now_sec() - item.t_capture) * 1000.0;
  double infer_total = 0;
  for (double v : infer_times) infer_total += v;
  double infer_mean = steps_run > 0 ? infer_total / steps_run : 0;
  constexpr double a = 0.2;
  if (steps_run > 0) {
    zero_step_blocks_ = 0;
    infer_ema_ms_ = (1 - a) * infer_ema_ms_ + a * infer_mean;
  } else {
    zero_step_blocks_ += 1;
    // A frozen-high EMA never updates while step_budget stays 0. Pull it
    // toward one hop's share of the block so the next fresh block can run.
    if (zero_step_blocks_ >= 8) {
      double hop_budget = bin.block_budget_ms / static_cast<double>(std::max(1, cfg_.hops_per_io));
      infer_ema_ms_ = (1 - a) * infer_ema_ms_ + a * hop_budget;
    }
  }
  dc_ema_ms_ = (1 - a) * dc_ema_ms_ + a * dc_ms;
  level_ema_ms_ = (1 - a) * level_ema_ms_ + a * level_ms;
  frame_ema_ms_ = (1 - a) * frame_ema_ms_ + a * frame_to_infer_ms;
  post_ema_ms_ = (1 - a) * post_ema_ms_ + a * post_ms;

  double mono = now_sec();
  if (cfg_.profile && (mono - last_profile_at_) >= std::max(0.1, cfg_.profile_interval)) {
    ProfileFields f;
    double out_ahead_ms = 1000.0 * output_ring_->ahead_frames() / cfg_.output_sr;
    f.mode = (steps_run == 0 && out_ahead_ms <= 0.1) ? "starved" : mode_;
    f.capture_age_ms = last_capture_age_ms_;
    f.capture_gap = last_capture_gap_;
    f.capture_drained = last_capture_drained_;
    f.capture_to_frame_ms = (t0 - item.t_capture) * 1000.0;
    f.dc_ms = dc_ms;
    f.level_ms = level_ms;
    f.frame_to_infer_ms = frame_to_infer_ms;
    f.infer_mean_ms = infer_mean;
    f.infer_total_ms = infer_total;
    f.steps = steps_run;
    f.post_ms = post_ms;
    f.e2e_ms = e2e_ms;
    f.q = capture_.size();
    f.qcap = cfg_.max_capture_blocks;
    f.drops_oldest = capture_.stats.dropped_oldest;
    f.out_ahead_ms = out_ahead_ms;
    f.dropped = out_dropped_;
    f.underrun_frames = output_ring_->underrun_frames();
    f.model_ring_ms = 1000.0 * model_ring_->size() / cfg_.model_sr;
    f.model_ring_drops = model_ring_drops_;
    f.model_ring_resets = model_ring_resets_;
    f.model_out_db = model_out_db_;
    f.output_db = output_db_;
    f.sustainable_steps = last_sustainable_steps_;
    f.drained_total = drained_total_;
    f.primed = primed_ ? 1 : 0;
    f.short_writes = short_block_writes_;
    last_profile_ = format_profile_line(f);
    if (wake_) last_profile_ += wake_->profile_suffix();
    std::cout << last_profile_ << "\n";
    last_profile_at_ = mono;
  }
  if (cfg_.monitor && (mono - last_monitor_at_) >= std::max(0.1, cfg_.monitor_interval)) {
    MonitorBlockFields mon;
    mon.levels_dbfs = latest_levels_dbfs_;
    mon.statuses = latest_statuses_;
    mon.distance_status = latest_distance_status_;
    mon.distance_m = latest_distance_m_;
    mon.count_status = latest_count_status_;
    mon.speaker_count = latest_speaker_count_;
    mon.model_out_db = model_out_db_;
    mon.output_db = output_db_;
    for (const auto& row : latest_speaker_distances_) {
      std::ostringstream part;
      part << "S" << row.speaker_index << ": ";
      if (row.distance_m) {
        part.setf(std::ios::fixed);
        part.precision(2);
        part << *row.distance_m << "m (" << row.status << ")";
      } else {
        part << row.status;
      }
      mon.speaker_parts.push_back(part.str());
    }
    std::cout << format_monitor_block(mon) << "\n";
    last_monitor_at_ = mono;
  }
  return true;
}

void Engine::push_capture_for_test(CaptureItem item) { capture_.push(std::move(item)); }

bool Engine::process_one_for_test(double pop_timeout_sec) {
  apply_contract_and_open_model();
  std::optional<CaptureItem> item;
  int drained = 0;
  if (cfg_.queue_policy == "fifo") {
    item = capture_.pop(pop_timeout_sec);
  } else {
    auto got = capture_.pop_latest(pop_timeout_sec);
    item = std::move(got.first);
    drained = got.second;
  }
  if (!item) return false;
  return handle_item(*item, drained);
}

void Engine::worker_loop() {
  try {
    apply_contract_and_open_model();
    while (!stop_.load()) {
      std::optional<CaptureItem> item;
      int drained = 0;
      if (cfg_.queue_policy == "fifo") item = capture_.pop(0.05);
      else {
        auto got = capture_.pop_latest(0.05);
        item = std::move(got.first);
        drained = got.second;
      }
      if (!item) continue;
      handle_item(*item, drained);
    }
  } catch (const std::exception& exc) {
    std::lock_guard<std::mutex> lock(err_mu_);
    worker_error_ = exc.what();
    worker_failed_ = true;
    request_stop("worker exception");
  }
}

int Engine::preflight() const {
  try {
    const bool duplex = cfg_.duplex_mode == "worker" || cfg_.duplex_mode == "callback";
    if (duplex) {
      preflight_duplex(AudioFormat{cfg_.input_device, cfg_.input_channels, cfg_.input_sr},
                       AudioFormat{cfg_.output_device, cfg_.output_channels, cfg_.output_sr});
      std::cout << "[deploy] full-duplex audio format ok\n";
    } else {
      preflight_devices(AudioFormat{cfg_.input_device, cfg_.input_channels, cfg_.input_sr},
                        AudioFormat{cfg_.output_device, cfg_.output_channels, cfg_.output_sr});
      std::cout << "[deploy] audio devices ok\n";
    }
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "[audio-config-error] " << exc.what() << "\n";
    return 1;
  }
}

int Engine::run_offline_wav() {
  apply_contract_and_open_model();
  auto wav = read_wav(cfg_.input_wav);
  if (wav.channels < cfg_.input_channels) {
    throw std::runtime_error("input wav has fewer channels than --input-channels");
  }
  std::uint64_t idx = 0;
  int frames = static_cast<int>(wav.interleaved.size()) / wav.channels;
  std::vector<float> rendered;
  for (int pos = 0; pos + input_block_ <= frames; pos += input_block_) {
    CaptureItem item;
    item.channels = cfg_.input_channels;
    item.interleaved.resize(static_cast<size_t>(input_block_ * cfg_.input_channels));
    for (int t = 0; t < input_block_; ++t) {
      for (int c = 0; c < cfg_.input_channels; ++c) {
        item.interleaved[static_cast<size_t>(t * cfg_.input_channels + c)] =
            wav.interleaved[static_cast<size_t>((pos + t) * wav.channels + c)];
      }
    }
    item.t_capture = now_sec();
    item.capture_idx = idx++;
    handle_item(item, 0);
    maybe_flush_taps();
    int ahead = output_ring_->ahead_frames();
    if (ahead > 0) {
      std::vector<float> block(static_cast<size_t>(ahead * cfg_.output_channels));
      int got = output_ring_->read_into(block.data(), ahead);
      rendered.insert(rendered.end(), block.begin(), block.begin() + static_cast<size_t>(got * cfg_.output_channels));
    }
  }
  if (!cfg_.output_wav.empty()) write_wav(cfg_.output_wav, cfg_.output_sr, cfg_.output_channels, rendered);
  flush_taps();
  return 0;
}

int Engine::run_live() {
  if (cfg_.list_devices) {
    print_devices();
    return 0;
  }
  if (cfg_.preflight_only) return preflight();
  if (preflight() != 0) return 1;

  const bool duplex = cfg_.duplex_mode == "worker" || cfg_.duplex_mode == "callback";
  if (!cfg_.duplex_mode.empty() && cfg_.duplex_mode != "off" && !duplex) {
    std::cerr << "[audio-config-error] --duplex-mode must be off, worker, or callback\n";
    return 1;
  }
  if (duplex && cfg_.input_sr != cfg_.output_sr) {
    std::cerr << "[audio-config-error] full-duplex requires input_sr == output_sr\n";
    return 1;
  }

  ScopedTimerResolution timer_resolution(1);
  apply_contract_and_open_model();

  if (!duplex) {
    // Normal: two independent PortAudio streams.
    int prime_frames = ring_params_.ring_target_ahead_frames;
    if (output_ring_ && prime_frames > 0) {
      output_ring_->write_silence(prime_frames);
      primed_ = true;
      std::cout << "[audio] primed output ring " << prime_frames << " frames\n";
    }
    worker_ = std::thread([this] { worker_loop(); });
    try {
      DuplexSession session(
          AudioFormat{cfg_.input_device, cfg_.input_channels, cfg_.input_sr},
          AudioFormat{cfg_.output_device, cfg_.output_channels, cfg_.output_sr}, input_block_, out_dev_block_,
          cfg_.latency,
          [this](const float* in, int frames, int channels) {
            CaptureItem item;
            item.channels = channels;
            item.interleaved.assign(in, in + static_cast<size_t>(frames * channels));
            item.t_capture = now_sec();
            static std::atomic<std::uint64_t> seq{0};
            item.capture_idx = seq.fetch_add(1);
            capture_.push(std::move(item));
          },
          [this](float* out, int frames, int) {
            if (!output_ring_) {
              std::fill(out, out + static_cast<size_t>(frames * cfg_.output_channels), 0.f);
              return;
            }
            output_ring_->read_into(out, frames);
          },
          [](bool input, const char*) {
            std::cout << (input ? "[audio] input callback reported status\n"
                                : "[audio] output callback reported status\n");
          });
      session.start();
      std::cout << "[audio] streams started (normal/two-stream), Ctrl+C to stop\n";
      while (!stop_.load()) {
        if (worker_failed_.load()) break;
        if (!session.active()) {
          request_stop("stream inactive");
          break;
        }
        maybe_flush_taps();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      session.stop();
    } catch (const std::exception& exc) {
      std::cerr << "[audio-runtime-error] " << exc.what() << "\n";
      request_stop("audio error");
      exit_code_ = 1;
    }
  } else {
    // Rapid: single full-duplex PortAudio stream.
    const bool callback_mode = cfg_.duplex_mode == "callback";
    int prime_frames = callback_mode ? 0 : ring_params_.ring_target_ahead_frames;
    if (output_ring_ && prime_frames > 0) {
      output_ring_->write_silence(prime_frames);
      primed_ = true;
      std::cout << "[audio] primed output ring " << prime_frames << " frames\n";
    } else {
      primed_ = true;
      std::cout << "[audio] callback-mode: no ring prime (latency = one duplex callback)\n";
    }
    if (!callback_mode) {
      worker_ = std::thread([this] { worker_loop(); });
    }
    std::vector<float> in_accum;
    in_accum.reserve(static_cast<size_t>(input_block_ * cfg_.input_channels * 2));
    std::atomic<std::uint64_t> capture_seq{0};
    try {
      DuplexSyncSession session(
          AudioFormat{cfg_.input_device, cfg_.input_channels, cfg_.input_sr},
          AudioFormat{cfg_.output_device, cfg_.output_channels, cfg_.output_sr}, duplex_block_, cfg_.latency,
          [this, callback_mode, &in_accum, &capture_seq](const float* in, float* out, int frames, int in_ch,
                                                         int out_ch) {
            if (!out) return;
            const int out_samples = frames * out_ch;
            if (!in || frames <= 0) {
              std::fill(out, out + out_samples, 0.f);
              return;
            }
            if (callback_mode) {
              CaptureItem item;
              item.channels = in_ch;
              item.interleaved.assign(in, in + static_cast<size_t>(frames * in_ch));
              item.t_capture = now_sec();
              item.capture_idx = capture_seq.fetch_add(1);
              try {
                handle_item(item, 0);
              } catch (const std::exception& exc) {
                std::lock_guard<std::mutex> lock(err_mu_);
                worker_error_ = exc.what();
                worker_failed_ = true;
                request_stop("callback exception");
                std::fill(out, out + out_samples, 0.f);
                return;
              }
              if (!output_ring_) {
                std::fill(out, out + out_samples, 0.f);
                return;
              }
              output_ring_->read_into(out, frames);
              return;
            }
            in_accum.insert(in_accum.end(), in, in + static_cast<size_t>(frames * in_ch));
            const int need = input_block_ * in_ch;
            while (static_cast<int>(in_accum.size()) >= need) {
              CaptureItem item;
              item.channels = in_ch;
              item.interleaved.assign(in_accum.begin(), in_accum.begin() + need);
              item.t_capture = now_sec();
              item.capture_idx = capture_seq.fetch_add(1);
              capture_.push(std::move(item));
              in_accum.erase(in_accum.begin(), in_accum.begin() + need);
            }
            if (!output_ring_) {
              std::fill(out, out + out_samples, 0.f);
              return;
            }
            output_ring_->read_into(out, frames);
          },
          [](const char*) { std::cout << "[audio] duplex callback reported status\n"; });
      session.start();
      std::cout << "[audio] full-duplex stream started (mode=" << cfg_.duplex_mode
                << ", framesPerBuffer=" << duplex_block_ << "), Ctrl+C to stop\n";
      while (!stop_.load()) {
        if (worker_failed_.load()) break;
        if (!session.active()) {
          request_stop("stream inactive");
          break;
        }
        maybe_flush_taps();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      session.stop();
    } catch (const std::exception& exc) {
      std::cerr << "[audio-runtime-error] " << exc.what() << "\n";
      request_stop("audio error");
      exit_code_ = 1;
    }
  }

  if (worker_.joinable()) worker_.join();
  if (wake_) wake_->stop();
  try {
    flush_taps();
  } catch (const std::exception& exc) {
    std::cerr << "[tap] flush failed: " << exc.what() << "\n";
    exit_code_ = 1;
  }
  if (worker_failed_.load()) {
    std::lock_guard<std::mutex> lock(err_mu_);
    std::cerr << "[worker-error] " << worker_error_ << "\n";
    return 1;
  }
  return signal_stop_ ? 0 : exit_code_;
}

void print_usage() {
  std::cout << "soundbubble-rt --model PATH [--contract-path PATH] [--input-wav PATH --output-wav PATH]\n"
               "  Live flags match edge/inference_pipeline.py. Add --preflight to check devices and exit.\n"
               "  --duplex-mode off|worker|callback   (default off; Rapid uses worker)\n"
               "  --duplex-block FRAMES               (0=auto; callback forces input_block)\n"
               "  --mix-uart-port PORT applies MIX wet/dry lines from the USB knob.\n"
               "  --tap-dir PATH                     write raw_8ch/mapped_6ch/dry_ref/wet_out WAVs (off by default)\n"
               "  --tap-select raw,mapped,dry,wet    (default raw,mapped,dry,wet; wet_out is pre-output_gain)\n"
               "  --tap-max-seconds N                (default 120; 0=unlimited)\n"
               "  --tap-max-mb M                     (default 1024; 0=unlimited)\n"
               "  Linux: SIGUSR1 toggles taps; SIGUSR2 flushes *.wav.partial -> *.wav\n";
}

PipelineConfig parse_args(int argc, char** argv) {
  PipelineConfig cfg;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + flag);
      return argv[++i];
    };
    if (a == "--help" || a == "-h") {
      print_usage();
      std::exit(0);
    } else if (a == "--list-devices") cfg.list_devices = true;
    else if (a == "--preflight") cfg.preflight_only = true;
    else if (a == "--profile") cfg.profile = true;
    else if (a == "--monitor") cfg.monitor = true;
    else if (a == "--require-soxr") cfg.require_soxr = true;
    else if (a == "--model") cfg.model = need(a.c_str());
    else if (a == "--contract-path") cfg.contract_path = need(a.c_str());
    else if (a == "--zoo-dir") cfg.zoo_dir = need(a.c_str());
    else if (a == "--bubble-radius") cfg.bubble_radius = std::stod(need(a.c_str()));
    else if (a == "--input-device") cfg.input_device = std::stoi(need(a.c_str()));
    else if (a == "--output-device") cfg.output_device = std::stoi(need(a.c_str()));
    else if (a == "--input-channels") cfg.input_channels = std::stoi(need(a.c_str()));
    else if (a == "--output-channels") cfg.output_channels = std::stoi(need(a.c_str()));
    else if (a == "--channel-map") cfg.channel_map = need(a.c_str());
    else if (a == "--monitor-input-channel") cfg.monitor_input_channel = std::stoi(need(a.c_str()));
    else if (a == "--input-sr") cfg.input_sr = std::stoi(need(a.c_str()));
    else if (a == "--model-sr") cfg.model_sr = std::stoi(need(a.c_str()));
    else if (a == "--output-sr") cfg.output_sr = std::stoi(need(a.c_str()));
    else if (a == "--model-chunk") cfg.model_chunk = std::stoi(need(a.c_str()));
    else if (a == "--model-pad") cfg.model_pad = std::stoi(need(a.c_str()));
    else if (a == "--model-num-ch") cfg.model_num_ch = std::stoi(need(a.c_str()));
    else if (a == "--hops-per-io") cfg.hops_per_io = std::stoi(need(a.c_str()));
    else if (a == "--output-device-block") cfg.output_device_block = std::stoi(need(a.c_str()));
    else if (a == "--duplex-mode") cfg.duplex_mode = need(a.c_str());
    else if (a == "--duplex-block") cfg.duplex_block = std::stoi(need(a.c_str()));
    else if (a == "--latency") cfg.latency = need(a.c_str());
    else if (a == "--max-capture-blocks") cfg.max_capture_blocks = std::stoi(need(a.c_str()));
    else if (a == "--max-output-ahead-sec") cfg.max_output_ahead_sec = std::stod(need(a.c_str()));
    else if (a == "--target-output-ahead-sec") cfg.target_output_ahead_sec = std::stod(need(a.c_str()));
    else if (a == "--rt-fifo-priority") cfg.rt_fifo_priority = std::stoi(need(a.c_str()));
    else if (a == "--output-gain-db") cfg.output_gain_db = std::stof(need(a.c_str()));
    else if (a == "--no-output-limiter") cfg.output_limiter = false;
    else if (a == "--limiter-ceiling") {
      float v = std::stof(need(a.c_str()));
      // Negative => dBFS (e.g. -0.5); positive stays linear full-scale.
      if (v < 0.f) v = dbfs_to_linear(v);
      cfg.limiter_ceiling = v;
    }
    else if (a == "--limiter-release-ms") cfg.limiter_release_ms = std::stof(need(a.c_str()));
    else if (a == "--limiter-hold-ms") cfg.limiter_hold_ms = std::stof(need(a.c_str()));
    else if (a == "--mic-gains" || a == "--mic-gains-db") {
      const bool db = (a == "--mic-gains-db");
      cfg.mic_gains.clear();
      std::stringstream ss(need(a.c_str()));
      std::string tok;
      while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        float v = std::stof(tok);
        if (db) v = dbfs_to_linear(v);
        if (!(v > 0.f) || !std::isfinite(v)) throw std::runtime_error("--mic-gains values must be finite and > 0");
        cfg.mic_gains.push_back(v);
      }
    }
    else if (a == "--wet-gate-dbfs") {
      std::string v = need(a.c_str());
      cfg.wet_gate_dbfs = (v == "off") ? -200.f : std::stof(v);
    }
    else if (a == "--ort-warmup-runs") cfg.ort_warmup_runs = std::stoi(need(a.c_str()));
    else if (a == "--output-gain") {
      throw std::runtime_error("--output-gain (linear) was removed; use --output-gain-db (0 = unity)");
    }
    else if (a == "--mix-dry") cfg.mix_dry = std::stof(need(a.c_str()));
    else if (a == "--input-gain-db") cfg.input_gain_db = std::stof(need(a.c_str()));
    else if (a == "--alpha") cfg.alpha = std::stof(need(a.c_str()));
    else if (a == "--silence-dbfs") cfg.silence_dbfs = std::stof(need(a.c_str()));
    else if (a == "--silence-window-frames") cfg.silence_window_frames = std::stoi(need(a.c_str()));
    else if (a == "--fault-min-dbfs") cfg.fault_min_dbfs = std::stof(need(a.c_str()));
    else if (a == "--fault-max-dbfs") cfg.fault_max_dbfs = std::stof(need(a.c_str()));
    else if (a == "--max-overrun-ms") cfg.max_overrun_ms = std::stof(need(a.c_str()));
    else if (a == "--intra-op-threads") cfg.intra_op_threads = std::stoi(need(a.c_str()));
    else if (a == "--inter-op-threads") cfg.inter_op_threads = std::stoi(need(a.c_str()));
    else if (a == "--profile-interval") cfg.profile_interval = std::stod(need(a.c_str()));
    else if (a == "--monitor-interval") cfg.monitor_interval = std::stod(need(a.c_str()));
    else if (a == "--model-ring-ms") cfg.model_ring_ms = std::stod(need(a.c_str()));
    else if (a == "--max-steps-per-block") cfg.max_steps_per_block = std::stoi(need(a.c_str()));
    else if (a == "--queue-policy") cfg.queue_policy = need(a.c_str());
    else if (a == "--max-capture-age-ms") cfg.max_capture_age_ms = std::stod(need(a.c_str()));
    else if (a == "--recover-steps-per-block") cfg.recover_steps_per_block = std::stoi(need(a.c_str()));
    else if (a == "--recover-ahead-multiplier") cfg.recover_ahead_multiplier = std::stod(need(a.c_str()));
    else if (a == "--overload-policy") cfg.overload_policy = need(a.c_str());
    else if (a == "--wake-model-dir") cfg.wake_model_dir = need(a.c_str());
    else if (a == "--wake-phrases-json") cfg.wake_phrases_json = need(a.c_str());
    else if (a == "--wake-phrase") cfg.wake_phrase.push_back(need(a.c_str()));
    else if (a == "--wake-threshold") cfg.wake_threshold = std::stof(need(a.c_str()));
    else if (a == "--wake-hold-sec") cfg.wake_hold_sec = std::stof(need(a.c_str()));
    else if (a == "--wake-replay-sec") cfg.wake_replay_sec = std::stof(need(a.c_str()));
    else if (a == "--wake-replay-rate") cfg.wake_replay_rate = std::stof(need(a.c_str()));
    else if (a == "--wake-release-ms") cfg.wake_release_ms = std::stof(need(a.c_str()));
    else if (a == "--mix-uart-port") cfg.mix_uart_port = need(a.c_str());
    else if (a == "--mix-uart-baud") cfg.mix_uart_baud = std::stoi(need(a.c_str()));
    else if (a == "--input-wav") cfg.input_wav = need(a.c_str());
    else if (a == "--output-wav") cfg.output_wav = need(a.c_str());
    else if (a == "--tap-dir") cfg.tap_dir = need(a.c_str());
    else if (a == "--tap-select") cfg.tap_select = need(a.c_str());
    else if (a == "--tap-max-seconds") cfg.tap_max_seconds = std::stod(need(a.c_str()));
    else if (a == "--tap-max-mb") cfg.tap_max_mb = std::stod(need(a.c_str()));
    else throw std::runtime_error("unknown argument " + a);
  }
  if (!cfg.tap_dir.empty()) {
    // Validate select early so bad tokens fail before audio/model open.
    (void)TapRecorder::parse_select(cfg.tap_select);
  }
  return cfg;
}

}  // namespace soundbubble
