#include "soundbubble/policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace soundbubble {

BudgetDecision decide_budget(const BudgetInput& in) {
  BudgetDecision d;
  d.mode = in.mode;
  double infer_ema = std::max(0.001, in.infer_ema_ms);
  double noninfer = in.dc_ema_ms + in.level_ema_ms + in.frame_to_infer_ema_ms + in.post_ema_ms;
  constexpr double reserve_ms = 1.0;
  double safe = std::max(0.0, in.block_budget_ms - noninfer - reserve_ms);
  int sustainable = std::max(0, static_cast<int>(std::floor(safe / infer_ema)));
  bool fresh = !in.capture_discontinuity && in.capture_age_ms <= in.max_capture_age_ms;
  if (fresh && in.model_has_chunk) sustainable = std::max(1, sustainable);
  d.sustainable_steps = sustainable;
  int steady = std::min(in.hops_per_io, std::min(sustainable, in.max_steps_per_block));
  int recover_target = std::min(in.recover_steps_per_block, std::min(in.hops_per_io, in.max_steps_per_block));
  bool need_recover = in.queue_policy == "adaptive" && in.mode != "shed" &&
                      in.capture_age_ms < 0.5 * in.max_capture_age_ms && in.out_ahead_ms < in.target_ahead_ms;
  double extra = std::max(0.0, in.adaptive_extra_ms);
  double block_ms = std::max(1.0, in.output_block_ms);
  if (in.recent_underrun_frames > 0) {
    extra += block_ms;
  } else {
    extra = std::max(0.0, extra - block_ms * 0.02);
  }
  // Low-latency targets (<20 ms) must not be doubled by the underrun cushion —
  // that alone can push effective ahead from 15 ms toward 30 ms.
  double ahead_mult = in.target_ahead_ms < 20.0 ? 1.25 : 2.0;
  double cap = std::min(in.target_ahead_ms * ahead_mult, std::max(in.target_ahead_ms, in.max_ahead_ms));
  extra = std::min(extra, std::max(0.0, cap - in.target_ahead_ms));
  d.adaptive_extra_ms = extra;
  if (in.capture_discontinuity) {
    if (d.mode != "shed") d.mode_changed = true;
    d.mode = "shed";
  } else if (d.mode == "shed") {
    d.mode_changed = true;
    d.mode = "normal";
  }
  auto clamp_ahead = [&](double base) {
    return std::min(cap, base + extra);
  };
  if (d.mode == "shed") {
    d.step_budget = steady;
    d.effective_target_ahead_ms = clamp_ahead(in.target_ahead_ms);
  } else if (need_recover && sustainable >= recover_target && recover_target > 0) {
    if (d.mode != "recover") d.mode_changed = true;
    d.mode = "recover";
    d.step_budget = recover_target;
    // Keep recover multiplier mild under 20 ms so a 9 ms target is not lifted to ~14 ms.
    double recover_mult = in.target_ahead_ms < 20.0 ? 1.0 : in.recover_ahead_multiplier;
    d.effective_target_ahead_ms = clamp_ahead(in.target_ahead_ms * recover_mult);
  } else {
    if (d.mode == "recover") d.mode_changed = true;
    if (d.mode != "shed") d.mode = "normal";
    d.step_budget = steady;
    d.effective_target_ahead_ms = clamp_ahead(in.target_ahead_ms);
  }
  return d;
}

OverloadPolicy parse_overload_policy(const std::string& name) {
  if (name == "dry_fill") return OverloadPolicy::DryFill;
  if (name == "silence_fill") return OverloadPolicy::SilenceFill;
  if (name == "strict") return OverloadPolicy::Strict;
  throw std::invalid_argument("overload policy must be dry_fill, silence_fill, or strict");
}

std::string format_profile_line(const ProfileFields& f) {
  std::ostringstream o;
  o.setf(std::ios::fixed);
  o.precision(2);
  o << "[profile] mode=" << f.mode << "  capture_age=" << f.capture_age_ms << "ms  capture_gap=" << f.capture_gap
    << "  capture_drained=" << f.capture_drained << "  capture->frame=" << f.capture_to_frame_ms
    << "ms  dc=" << f.dc_ms << "ms  level-valid=" << f.level_ms << "ms  frame->infer=" << f.frame_to_infer_ms
    << "ms  infer_mean=" << f.infer_mean_ms << "ms  infer_total=" << f.infer_total_ms << "ms  steps=" << f.steps
    << "  post+out=" << f.post_ms << "ms  e2e=" << f.e2e_ms << "ms  drift=";
  o << std::showpos << f.drift_ppm << std::noshowpos << "ppm  q=" << f.q << "/" << f.qcap
    << "  drops_oldest=" << f.drops_oldest << "  out_ahead=";
  o.precision(1);
  o << f.out_ahead_ms << "ms dropped=" << f.dropped << " underrun_frames=" << f.underrun_frames << "  model_ring=";
  o.precision(1);
  o << f.model_ring_ms << "ms drops=" << f.model_ring_drops << " resets=" << f.model_ring_resets;
  o.precision(1);
  o << "  model_out=" << f.model_out_db << "dBFS out=" << f.output_db << "dBFS";
  o.precision(0);
  o << "  sustainable=" << f.sustainable_steps << " drained_total=" << f.drained_total << " primed=" << f.primed
    << " short_writes=" << f.short_writes;
  return o.str();
}

std::string format_monitor_block(const MonitorBlockFields& m) {
  const int n = static_cast<int>(std::min(m.levels_dbfs.size(), m.statuses.size()));
  const int split = std::max(1, n / 2);
  auto token = [&](int ch) {
    char buf[64];
    float lvl = ch < static_cast<int>(m.levels_dbfs.size()) ? m.levels_dbfs[static_cast<size_t>(ch)] : -120.f;
    std::string st = ch < static_cast<int>(m.statuses.size()) ? m.statuses[static_cast<size_t>(ch)] : "ok";
    if (st.size() > 7) st.resize(7);
    // Match Python render_monitor: f"ch{ch}: {lvl:6.1f} dBFS {status:7s}"
    std::snprintf(buf, sizeof(buf), "ch%d: %6.1f dBFS %-7s", ch, static_cast<double>(lvl), st.c_str());
    return std::string(buf);
  };
  std::ostringstream o;
  o.setf(std::ios::fixed);
  for (int ch = 0; ch < split && ch < n; ++ch) {
    if (ch) o << "    ";
    o << token(ch);
  }
  if (n > split) {
    o << "\n";
    for (int ch = split; ch < n; ++ch) {
      if (ch > split) o << "    ";
      o << token(ch);
    }
  }
  // Same block order as inference_pipeline.py --monitor print.
  o << "\n";
  if (m.speaker_count) {
    o << "speaker_count=" << *m.speaker_count << " (" << m.count_status << ")\n";
  } else {
    o << "speaker_count=" << m.count_status << "\n";
  }
  if (m.distance_m) {
    o << std::setprecision(2) << "distance=" << *m.distance_m << "m (" << m.distance_status << ")\n";
  } else {
    o << "distance=" << m.distance_status << "\n";
  }
  o << std::setprecision(1) << "model_out=" << m.model_out_db << " dBFS  out=" << m.output_db << " dBFS";
  if (!m.speaker_parts.empty()) {
    o << "\nspeakers=[";
    for (size_t i = 0; i < m.speaker_parts.size(); ++i) {
      if (i) o << ", ";
      o << m.speaker_parts[i];
    }
    o << "]";
  }
  return o.str();
}

}  // namespace soundbubble
