#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace soundbubble {

struct BudgetInput {
  std::string queue_policy = "adaptive";
  std::string mode = "normal";
  int hops_per_io = 2;
  int max_steps_per_block = 4;
  int recover_steps_per_block = 2;
  double recover_ahead_multiplier = 1.5;
  double max_capture_age_ms = 24.0;
  double block_budget_ms = 16.0;
  double infer_ema_ms = 8.5;
  double dc_ema_ms = 0.1;
  double level_ema_ms = 0.2;
  double frame_to_infer_ema_ms = 0.6;
  double post_ema_ms = 0.2;
  double out_ahead_ms = 40.0;
  double target_ahead_ms = 40.0;
  double capture_age_ms = 0.0;
  bool capture_discontinuity = false;
  // True when the model ring already holds at least one hop. Lets a frozen
  // infer EMA still attempt one hop so the estimate can be remeasured.
  bool model_has_chunk = false;
  // Output frames the playback ring failed to supply since the previous decision.
  int recent_underrun_frames = 0;
  double output_block_ms = 16.0;
  double max_ahead_ms = 80.0;
  // Extra cushion above target_ahead_ms, persisted by the caller.
  double adaptive_extra_ms = 0.0;
};

struct BudgetDecision {
  std::string mode;
  int step_budget = 0;
  double effective_target_ahead_ms = 0.0;
  bool mode_changed = false;
  int sustainable_steps = 0;
  double adaptive_extra_ms = 0.0;
};

BudgetDecision decide_budget(const BudgetInput& in);

enum class OverloadPolicy { DryFill, SilenceFill, Strict };
OverloadPolicy parse_overload_policy(const std::string& name);

struct ProfileFields {
  std::string mode;
  double capture_age_ms = 0;
  int capture_gap = 0;
  int capture_drained = 0;
  double capture_to_frame_ms = 0;
  double dc_ms = 0;
  double level_ms = 0;
  double frame_to_infer_ms = 0;
  double infer_mean_ms = 0;
  double infer_total_ms = 0;
  int steps = 0;
  double post_ms = 0;
  double e2e_ms = 0;
  double drift_ppm = 0;
  int q = 0;
  int qcap = 0;
  std::uint64_t drops_oldest = 0;
  double out_ahead_ms = 0;
  std::uint64_t dropped = 0;
  std::uint64_t underrun_frames = 0;
  double model_ring_ms = 0;
  std::uint64_t model_ring_drops = 0;
  std::uint64_t model_ring_resets = 0;
  // Appended after the controller contract fields (shared with Python).
  double model_out_db = -120.0;
  double output_db = -120.0;
  int sustainable_steps = 0;
  std::uint64_t drained_total = 0;
  int primed = 0;
  std::uint64_t short_writes = 0;
};

std::string format_profile_line(const ProfileFields& f);

struct MonitorBlockFields {
  std::vector<float> levels_dbfs;
  std::vector<std::string> statuses;
  std::string distance_status = "unavailable";
  std::optional<float> distance_m;
  std::string count_status = "unavailable";
  std::optional<std::string> speaker_count;
  double model_out_db = -120.0;
  double output_db = -120.0;
  // Optional "speakers=[S1: …]" rows; empty skips the line.
  std::vector<std::string> speaker_parts;
};

std::string format_monitor_block(const MonitorBlockFields& m);

}  // namespace soundbubble
