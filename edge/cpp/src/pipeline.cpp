#include "soundbubble/pipeline.hpp"

#include "soundbubble/audio_io.hpp"
#include "soundbubble/mixctl.hpp"
#include "soundbubble/processing.hpp"
#include "soundbubble/wake.hpp"
#include "soundbubble/wav.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
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
    out_limiter_.configure({cfg_.limiter_ceiling, cfg_.limiter_release_ms},
                           static_cast<std::uint32_t>(std::max(1, cfg_.output_sr)));
  }
  if (!infer_) {
    infer_ = make_onnx_inferencer(cfg_.model, cfg_.intra_op_threads, cfg_.inter_op_threads, have_contract_ ? &contract_ : nullptr,
                                  cfg_.bubble_radius);
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
  // Order is fixed: output gain first, then peak limiter (ceiling guards DAC).
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
  bool disc = false;
  if (drained > 0) disc = true;
  if (capture_gap > 0) disc = true;
  if (cfg_.queue_policy == "adaptive" && capture_age_ms > cfg_.max_capture_age_ms) disc = true;
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
  std::vector<float> xref(static_cast<size_t>(T));
  for (int t = 0; t < T; ++t) xref[static_cast<size_t>(t)] = planar[static_cast<size_t>(cfg_.monitor_input_channel * T + t)];
  std::vector<float> selected(static_cast<size_t>(selected_.size() * T));
  for (size_t i = 0; i < selected_.size(); ++i) {
    int c = selected_[i];
    std::copy(planar.begin() + static_cast<size_t>(c * T), planar.begin() + static_cast<size_t>((c + 1) * T),
              selected.begin() + static_cast<size_t>(i * T));
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
    model_ring_drops_ += 1;
    reset_stream_timeline(true, true);
    model_ring_resets_ += 1;
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
  bool pending_reset = disc;
  if (!y_chunks.empty()) {
    model_out_db_ = 0.8 * model_out_db_ + 0.2 * static_cast<double>(rms_dbfs(y_chunks.data(), static_cast<int>(y_chunks.size())));
    auto y_io = out_src_->process(y_chunks.data(), static_cast<int>(y_chunks.size()));
    auto mixed = mix_wet_with_dry(y_io, nullptr, static_cast<int>(y_io.size()));
    apply_output_gain_and_limit(mixed);
    output_db_ = 0.8 * output_db_ + 0.2 * static_cast<double>(rms_dbfs(mixed.data(), static_cast<int>(mixed.size())));
    y_out = to_multichannel(mixed.data(), static_cast<int>(mixed.size()), cfg_.output_channels);
  }
  auto policy = parse_overload_policy(cfg_.overload_policy);
  if (missing > 0 && policy != OverloadPolicy::Strict) {
    int missing_frames = missing * frames_per_hop_out;
    std::vector<float> fill;
    if (policy == OverloadPolicy::DryFill) {
      auto dry_fill = dry_->read_block(missing_frames);
      auto mixed = mix_wet_with_dry(std::vector<float>(static_cast<size_t>(missing_frames), 0.f), dry_fill.data(), missing_frames);
      if (!wake_) mixed = dry_fill;
      apply_output_gain_and_limit(mixed);
      fill = to_multichannel(mixed.data(), missing_frames, cfg_.output_channels);
    } else {
      fill.assign(static_cast<size_t>(missing_frames * cfg_.output_channels), 0.f);
    }
    y_out.insert(y_out.end(), fill.begin(), fill.end());
    pending_reset = true;
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
      if (have < frames_needed && policy == OverloadPolicy::DryFill) {
        int gap = frames_needed - have;
        auto dry_gap = dry_->read_block(gap);
        apply_output_gain_and_limit(dry_gap);
        auto fill = to_multichannel(dry_gap.data(), gap, cfg_.output_channels);
        std::copy(fill.begin(), fill.end(), fixed.begin() + static_cast<size_t>(have * cfg_.output_channels));
      }
      y.swap(fixed);
    };
    fit_exact(y_out, output_block_);
    // Each skipped capture index already advanced the output clock: queue drops
    // show up as capture_gap, and pop_latest collapses show up as drained.
    // Write one output block per skipped index so those periods are not holes.
    int skipped = std::max(std::max(0, drained), std::max(0, capture_gap));
    int room = std::max(1, ring_params_.ring_max_ahead_frames / std::max(1, output_block_));
    int catchup = std::min(skipped, std::max(0, room - 1));
    if (catchup > 0) {
      int extra = catchup * output_block_;
      std::vector<float> lead;
      if (policy == OverloadPolicy::DryFill) {
        auto dry_lead = dry_->read_block(extra);
        apply_output_gain_and_limit(dry_lead);
        lead = to_multichannel(dry_lead.data(), extra, cfg_.output_channels);
      } else {
        lead.assign(static_cast<size_t>(extra * cfg_.output_channels), 0.f);
      }
      y_out.insert(y_out.begin(), lead.begin(), lead.end());
    }
  }
  if (!y_out.empty()) {
    int frames = static_cast<int>(y_out.size()) / cfg_.output_channels;
    int dropped = output_ring_->write_drop_oldest(y_out.data(), frames);
    out_dropped_ += static_cast<std::uint64_t>(std::max(0, dropped));
    int ahead = output_ring_->ahead_frames();
    // Trim toward the effective target (+ 1/4 block of slack). A full block of
    // slack (~8 ms) made a 9 ms target unreachable (ring stuck at 16 ms).
    int target = static_cast<int>(decision.effective_target_ahead_ms * cfg_.output_sr / 1000.0);
    int slack = std::max(1, output_block_ / 4);
    int trim_threshold = std::min(ring_params_.ring_max_ahead_frames, target + slack);
    if (ahead > trim_threshold) {
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
    int ahead = output_ring_->ahead_frames();
    if (ahead > 0) {
      std::vector<float> block(static_cast<size_t>(ahead * cfg_.output_channels));
      int got = output_ring_->read_into(block.data(), ahead);
      rendered.insert(rendered.end(), block.begin(), block.begin() + static_cast<size_t>(got * cfg_.output_channels));
    }
  }
  if (!cfg_.output_wav.empty()) write_wav(cfg_.output_wav, cfg_.output_sr, cfg_.output_channels, rendered);
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
               "  --mix-uart-port PORT applies MIX wet/dry lines from the USB knob.\n";
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
    else throw std::runtime_error("unknown argument " + a);
  }
  return cfg;
}

}  // namespace soundbubble
