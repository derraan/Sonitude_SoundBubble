#pragma once

#include "soundbubble/contract.hpp"
#include "soundbubble/limiter.hpp"
#include "soundbubble/policy.hpp"
#include "soundbubble/realtime.hpp"

#include <atomic>
#include <functional>
#include <optional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace soundbubble {

struct PipelineConfig {
  std::string model;
  std::string contract_path;
  std::string zoo_dir;
  std::optional<double> bubble_radius;
  std::optional<int> input_device;
  std::optional<int> output_device;
  int input_channels = 8;
  int output_channels = 2;
  std::string channel_map = "5,4,3,2,1,0";  // MIC_INDEX_CANON.md §4
  int monitor_input_channel = 0;
  int input_sr = 24000;
  int model_sr = 24000;
  int output_sr = 48000;
  int model_chunk = 192;
  int model_pad = 96;
  int model_num_ch = 6;
  int hops_per_io = 2;
  // 0 = auto (~2 ms DAC callback). Decouples PortAudio output framesPerBuffer
  // from the model hop quantum so a 9 ms ring cushion has real jitter headroom.
  int output_device_block = 0;
  // "off" = two PortAudio streams (Normal). "worker"|"callback" = single full-duplex
  // stream (Rapid). Empty treated as off.
  std::string duplex_mode = "off";
  // Shared framesPerBuffer for DuplexSyncSession. 0 = auto (~2 ms in worker mode;
  // callback mode always uses input_block).
  int duplex_block = 0;
  std::string latency = "low";
  int max_capture_blocks = 2;
  double max_output_ahead_sec = 0.10;
  double target_output_ahead_sec = 0.04;
  int rt_fifo_priority = 0;
  float output_gain_db = 0.f;
  bool output_limiter = true;
  // Default −0.5 dBFS ≈ 0.9440609 linear full-scale.
  float limiter_ceiling = 0.9440609f;
  float limiter_release_ms = 25.f;
  float limiter_hold_ms = 15.f;
  // Per-mic linear input gains indexed by DEVICE channel, applied right after
  // capture (before channel map / DC / input gain). Raw tap stays untrimmed.
  // --mic-gains (linear, > 0) or --mic-gains-db.
  std::vector<float> mic_gains;
  // Wet residual expander threshold (pre-output-gain dBFS). Applied to the DAC
  // path only when the output is pure wet (mix_dry == 0, no wake). Taps stay
  // pre-gate. <= -150 disables.
  float wet_gate_dbfs = -58.f;
  // ONNX dry runs before streams open (cold first inferences starve the ring).
  int ort_warmup_runs = 12;
  float mix_dry = 0.f;
  float input_gain_db = 0.f;
  float alpha = 0.9999f;
  float silence_dbfs = -50.f;
  int silence_window_frames = 8;
  float fault_min_dbfs = -1.f;
  float fault_max_dbfs = 0.f;
  float max_overrun_ms = 25.f;
  int intra_op_threads = 2;
  int inter_op_threads = 1;
  bool profile = false;
  double profile_interval = 1.0;
  bool monitor = false;
  double monitor_interval = 0.5;
  double model_ring_ms = 64.0;
  int max_steps_per_block = 4;
  bool require_soxr = false;
  std::string queue_policy = "adaptive";
  double max_capture_age_ms = 24.0;
  int recover_steps_per_block = 2;
  double recover_ahead_multiplier = 1.5;
  std::string overload_policy = "dry_fill";
  std::string wake_model_dir;
  std::string wake_phrases_json;
  std::vector<std::string> wake_phrase;
  float wake_threshold = 0.25f;
  float wake_hold_sec = 8.f;
  float wake_replay_sec = 1.5f;
  float wake_replay_rate = 1.25f;
  float wake_release_ms = 300.f;
  std::string mix_uart_port;
  int mix_uart_baud = 115200;
  std::string input_wav;
  std::string output_wav;
  // WAV taps: off unless --tap-dir is set. Default select is mapped,dry,wet.
  std::string tap_dir;
  std::string tap_select = "raw,mapped,dry,wet";
  double tap_max_seconds = 120.0;
  double tap_max_mb = 1024.0;
  bool list_devices = false;
  bool preflight_only = false;
};

class Inferencer {
 public:
  virtual ~Inferencer() = default;
  virtual void reset() = 0;
  virtual bool accepts_dis_embed() const { return false; }
  virtual void set_radius(double) {}
  virtual void set_distance_runtime_config(bool /*trained*/, float /*valid_threshold*/, bool /*gate_on_count*/) {}
  virtual void set_count_runtime_config(bool /*trained*/, float /*min_confidence*/) {}
  struct SpeakerDistance {
    int speaker_index = 0;
    std::optional<float> distance_m;
    std::string status = "unavailable";
  };
  struct Hop {
    std::vector<float> samples;
    double infer_ms = 0;
    std::string distance_status = "unavailable";
    std::optional<float> distance_m;
    std::string count_status = "unavailable";
    std::optional<std::string> speaker_count;
    std::vector<SpeakerDistance> speaker_distances;
  };
  virtual Hop run(const float* frame_nct, int channels, int frame_len) = 0;
};

std::unique_ptr<Inferencer> make_onnx_inferencer(const std::string& model_path, int intra, int inter,
                                                 const RuntimeContract* contract, std::optional<double> radius);

struct CaptureItem {
  std::vector<float> interleaved;
  int channels = 0;
  double t_capture = 0;
  std::uint64_t capture_idx = 0;
};

class Engine {
 public:
  explicit Engine(PipelineConfig cfg);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  void set_inferencer_for_test(std::unique_ptr<Inferencer> inf);
  int run_live();
  int run_offline_wav();
  bool wants_offline() const { return !cfg_.input_wav.empty(); }
  int preflight() const;
  void request_stop(const std::string& reason);
  bool stopped() const { return stop_.load(); }

  // Tap logging (SIGUSR1 toggle / SIGUSR2 flush on Linux; stdin TAP cmds everywhere).
  void toggle_taps();
  void request_tap_flush();
  bool taps_armed() const;
  void arm_taps();
  void disarm_taps();
  // Retarget in-memory tap sink to a new directory (creates dirs; clears buffers; disarms).
  void retarget_tap_dir(const std::string& dir);
  // Stdin control lines: STOP | TAP arm|disarm|flush|toggle|dir <path>
  void handle_control_line(const std::string& line);

  void push_capture_for_test(CaptureItem item);
  bool process_one_for_test(double pop_timeout_sec);
  int output_ahead_frames() const;
  std::uint64_t model_ring_drops() const { return model_ring_drops_; }
  const std::string& last_profile() const { return last_profile_; }
  float current_mix_dry() const;

 private:
  void apply_contract_and_open_model();
  void prepare_graph();
  void worker_loop();
  bool handle_item(const CaptureItem& item, int drained);
  void reset_stream_timeline(bool clear_io_rings, bool clear_model_ring);
  std::vector<float> mix_wet_with_dry(const std::vector<float>& wet, const float* live_dry_or_null, int n);
  void apply_output_gain_and_limit(std::vector<float>& mono);
  void declick_fill_edges(std::vector<float>& y_out, std::vector<std::uint8_t>& fill_mask, int frames);
  // Capture wet (and optional aligned dry) pre-output_gain, then apply gain/limit.
  void tap_dry_wet_then_gain(std::vector<float>& mono, const float* dry_or_null);
  void poll_mix_tap_cmds();
  void maybe_flush_taps();
  void flush_taps();

  PipelineConfig cfg_;
  PeakLimiter out_limiter_;
  ResidualGate wet_gate_;
  bool last_write_was_fill_ = false;  // output ring tail state for gap declick
  RuntimeContract contract_{};
  bool have_contract_ = false;
  std::vector<int> selected_;
  int input_block_ = 0;
  int output_block_ = 0;
  int out_dev_block_ = 0;
  int duplex_block_ = 0;
  int model_frame_len_ = 0;
  OutputRingParams ring_params_{};
  std::unique_ptr<class AudioRingBuffer> model_ring_;
  std::unique_ptr<OutputRingBuffer> output_ring_;
  std::unique_ptr<class DryPath> dry_;
  std::unique_ptr<Inferencer> infer_;
  std::unique_ptr<class DCOffsetFilter> dc_;
  std::unique_ptr<class ChannelLevelValidator> levels_;
  std::unique_ptr<class StreamingResampler> in_src_;
  std::unique_ptr<class StreamingResampler> out_src_;
  std::unique_ptr<class WakeRuntime> wake_;
  std::unique_ptr<class MixControl> mix_;
  std::unique_ptr<class TapRecorder> taps_;
  DropOldestBuffer<CaptureItem> capture_;
  std::vector<float> current_frame_;
  std::vector<float> last_output_chunk_;
  std::string mode_ = "normal";
  double infer_ema_ms_ = 8.5;
  double dc_ema_ms_ = 0.1;
  double level_ema_ms_ = 0.2;
  double frame_ema_ms_ = 0.6;
  double post_ema_ms_ = 0.2;
  double adaptive_extra_ms_ = 0;
  int zero_step_blocks_ = 0;
  int last_sustainable_steps_ = 0;
  std::uint64_t drained_total_ = 0;
  std::uint64_t last_underrun_frames_ = 0;
  std::uint64_t short_block_writes_ = 0;
  bool primed_ = false;
  std::optional<std::uint64_t> last_capture_idx_;
  double last_capture_age_ms_ = 0;
  int last_capture_gap_ = 0;
  int last_capture_drained_ = 0;
  std::uint64_t drops_note_ = 0;
  std::uint64_t model_ring_drops_ = 0;
  std::uint64_t model_ring_resets_ = 0;
  std::uint64_t out_dropped_ = 0;
  std::uint64_t discontinuities_ = 0;
  std::atomic<bool> stop_{false};
  std::atomic<bool> worker_failed_{false};
  std::atomic<bool> callback_failed_{false};
  std::atomic<bool> tap_flush_req_{false};
  std::string worker_error_;
  std::mutex err_mu_;
  std::thread worker_;
  std::string last_profile_;
  double last_profile_at_ = 0;
  double last_monitor_at_ = 0;
  std::vector<float> latest_levels_dbfs_;
  std::vector<std::string> latest_statuses_;
  std::string latest_distance_status_ = "unavailable";
  std::optional<float> latest_distance_m_;
  std::string latest_count_status_ = "unavailable";
  std::optional<std::string> latest_speaker_count_;
  std::vector<Inferencer::SpeakerDistance> latest_speaker_distances_;
  double model_out_db_ = -120.0;
  double output_db_ = -120.0;
  bool prepared_ = false;
  int exit_code_ = 0;
  bool signal_stop_ = false;
};

PipelineConfig parse_args(int argc, char** argv);
void print_usage();

}  // namespace soundbubble
