#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace soundbubble {

struct ControllerConfig {
  std::optional<std::string> model = std::string("edge/model.onnx");
  std::optional<std::string> contract_path = std::string("edge/model.runtime.json");
  std::optional<std::string> zoo_dir;
  std::optional<double> bubble_radius;
  std::optional<int> input_device;
  std::optional<int> output_device;
  int input_channels = 8;
  int output_channels = 2;
  std::string channel_map = "0,1,2,3,4,5";
  int monitor_input_channel = 0;
  int input_sr = 24000;
  int model_sr = 24000;
  // Class B combined Pico duplex defaults to 24 kHz (see HARDWARE_CLASSES.md).
  int output_sr = 24000;
  int model_chunk = 192;
  int model_pad = 96;
  int model_num_ch = 6;
  int hops_per_io = 2;
  // 0 = auto (~2 ms DAC callback). See PipelineConfig::output_device_block.
  int output_device_block = 0;
  // "off" = Normal two-stream; "worker"|"callback" = Rapid full-duplex.
  std::string duplex_mode = "off";
  int duplex_block = 0;
  std::string latency = "low";
  int max_capture_blocks = 2;
  double max_output_ahead_sec = 0.10;
  double target_output_ahead_sec = 0.04;
  int rt_fifo_priority = 0;
  double output_gain_db = 0.0;
  bool output_limiter = true;
  // Default −0.5 dBFS ≈ 0.9440609 linear full-scale.
  double limiter_ceiling = 0.9440608762859234;
  double limiter_release_ms = 25.0;
  double mix_dry = 0.0;
  double input_gain_db = 0.0;
  double alpha = 0.9999;
  double silence_dbfs = -50.0;
  int silence_window_frames = 8;
  double fault_min_dbfs = -1.0;
  double fault_max_dbfs = 0.0;
  double max_overrun_ms = 25.0;
  int intra_op_threads = 2;
  int inter_op_threads = 1;
  double model_ring_ms = 64.0;
  int max_steps_per_block = 4;
  bool require_soxr = false;
  std::optional<std::string> wake_model_dir;
  std::optional<std::string> wake_phrases_json;
  std::vector<std::string> wake_phrase;
  double wake_threshold = 0.25;
  double wake_hold_sec = 8.0;
  double wake_replay_sec = 1.5;
  double wake_replay_rate = 1.25;
  double wake_release_ms = 300.0;
  bool profile = true;
  double profile_interval = 1.0;
  bool monitor = false;
  double monitor_interval = 0.5;
  std::string mix_uart_port;
  int mix_uart_baud = 115200;
  std::string runtime = "cpp";
};

ControllerConfig controller_config_from_json(const std::string& json_text);
std::string controller_config_to_json(const ControllerConfig& cfg);
ControllerConfig validate_controller_config(ControllerConfig cfg, const std::string& repo_root);
std::vector<std::string> build_pipeline_argv(const ControllerConfig& cfg, const std::string& repo_root,
                                             const std::string& rt_bin);

struct ProfileParse {
  bool ok = false;
  double infer_ms = 0;
  double capture_to_frame_ms = 0;
  double dc_ms = 0;
  double level_valid_ms = 0;
  double frame_to_infer_ms = 0;
  double post_out_ms = 0;
  double e2e_ms = 0;
  double drift_ppm = 0;
  int capture_queue_fill = 0;
  int capture_queue_capacity = 0;
  std::uint64_t drops_oldest = 0;
  double out_ahead_ms = 0;
  std::uint64_t out_dropped_frames = 0;
  std::uint64_t underrun_frames = 0;
  double model_ring_ms = 0;
  std::uint64_t model_ring_drops = 0;
  std::uint64_t model_ring_resets = 0;
};

ProfileParse parse_profile_line(const std::string& line);

// One accumulated --monitor snapshot. Python prints channel rows, then
// speaker_count= and distance= on following lines. Missing pieces stay empty.
struct MonitorParse {
  bool any = false;
  std::vector<double> levels_dbfs;
  std::vector<std::string> statuses;
  std::optional<double> distance_m;
  std::string distance_status;
  std::optional<std::string> speaker_count;
  std::optional<double> bubble_radius_m;
  // From the same --monitor block as Python: model_out=… dBFS  out=… dBFS
  std::optional<double> model_out_db;
  std::optional<double> output_db;
  // Raw `speakers=[S1: …]` tail when present.
  std::string speakers;
};

bool parse_monitor_line(const std::string& line, MonitorParse& acc);

struct TapResult {
  int status = 200;
  std::string body;
};

class ProcessManager {
 public:
  explicit ProcessManager(std::string repo_root, std::string rt_bin, std::string python_exe);
  ~ProcessManager();
  bool is_running();
  std::string start(const std::string& config_json);
  std::string stop();
  std::string restart(const std::string& config_json);
  std::string status_json();
  std::string logs_json(int tail) const;
  std::string telemetry_json();
  std::string trials_json(const std::string& run_id, const std::string& since) const;
  // SIGUSR1 is a single toggle in the capture runtime. arm and disarm both send
  // it and the body is {"ok":true,"action":"toggle"}. Not owned -> 409.
  TapResult signal_taps(const std::string& action);
  void ingest_line(const std::string& line);
  // Records a ControllerConfig as the owned child's config without spawning.
  void note_owned_config_for_test(const std::string& config_json);

 private:
  void append(const std::string& line, const std::string& stream);
  void note_runtime(const std::string& line);
  void reader_fd(int fd, const std::string& stream);
  void observe_main();
  void follow_journal();
  bool unit_is_active() const;
  // Stops systemd unit sonitude-soundbubble so an owned child can open the Pico.
  // Requires passwordless sudo for that unit (edge/rpi5/sudoers-sonitude-soundbubble-ctl).
  void ensure_production_stopped();
  // waitpid(WNOHANG): clear process_ when the owned child has already exited.
  void reap_exited_child();
  std::string measure_root() const;
  void load_trial_records(const std::string& run_id, const std::string& since, std::string& resolved_run,
                          std::vector<std::string>& raw_lines) const;
  std::string repo_;
  std::string rt_bin_;
  std::string python_;
  mutable std::recursive_mutex mu_;
  std::optional<std::intptr_t> pid_;
  void* process_ = nullptr;
  std::string started_at_;
  std::string config_json_;
  std::string owned_config_json_;
  bool test_owned_ = false;
  std::vector<std::string> argv_;
  std::optional<int> returncode_;
  std::deque<std::string> logs_;
  std::deque<std::string> errors_;
  std::string last_profile_json_;
  MonitorParse monitor_;
  bool has_monitor_ = false;
  std::optional<double> bubble_radius_;
  std::string runtime_ = "cpp";
  bool unit_active_ = false;
  bool taps_armed_ = false;
  std::string taps_select_;
  bool taps_truncated_ = false;
  std::uint64_t seq_ = 0;
  std::atomic<bool> observe_stop_{false};
  int journal_pid_ = 0;
  std::thread observe_thread_;
  std::thread out_thread_;
  std::thread err_thread_;
};

class ControllerServer {
 public:
  ControllerServer(std::string host, int port, std::string token, ProcessManager& manager);
  ~ControllerServer();
  void serve();
  void stop();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace soundbubble
