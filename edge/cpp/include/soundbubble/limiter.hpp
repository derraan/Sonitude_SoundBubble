#pragma once

#include <cstdint>
#include <span>

namespace soundbubble {

// Sample-peak limiter: instant attack (ceiling is guaranteed), then a hold and
// an exponential release toward unity. No look-ahead — zero added latency.
// The hold must exceed a voice pitch period (~2.5–12 ms): without it the gain
// re-opens between glottal peaks and is re-clamped every cycle, which
// modulates the voice at pitch rate (buzzy distortion under high output gain).
// Linked stereo uses max(|L|,|R|).
struct LimiterConfig {
  float ceiling = 0.9440609f;  // −0.5 dBFS
  float release_ms = 25.f;     // exponential time constant after the hold
  float hold_ms = 15.f;
};

class PeakLimiter {
 public:
  void configure(const LimiterConfig& cfg, std::uint32_t sample_rate_hz);
  void reset();
  void process_mono(std::span<float> samples);
  void process_interleaved_stereo(std::span<float> interleaved_lr);
  float current_gain() const { return gain_; }

 private:
  float next_gain(float amplitude);

  LimiterConfig cfg_{};
  bool configured_ = false;
  float gain_ = 1.f;
  float release_coeff_ = 0.f;  // per-sample pull toward 1
  int hold_samples_ = 0;
  int hold_left_ = 0;
};

// Downward expander for the wet path (before output gain). When the model is
// suppressing, its residual (−65…−80 dBFS pre-gain) is dominated by a hop-rate
// comb (125/250 Hz harmonics) that output gain turns into an audible buzz.
// Below threshold the gain falls at (ratio−1) dB per dB, capped at range_db.
struct ResidualGateConfig {
  float threshold_dbfs = -58.f;
  float ratio = 3.f;
  float range_db = 40.f;
  float attack_ms = 2.f;    // gain opening
  float release_ms = 60.f;  // gain closing
};

class ResidualGate {
 public:
  void configure(const ResidualGateConfig& cfg, std::uint32_t sample_rate_hz);
  void reset();
  void process_mono(std::span<float> samples);
  bool configured() const { return configured_; }
  float current_gain() const { return gain_; }

 private:
  ResidualGateConfig cfg_{};
  bool configured_ = false;
  float env_ = 0.f;
  float gain_ = 1.f;
  float env_att_ = 0.f, env_rel_ = 0.f, g_att_ = 0.f, g_rel_ = 0.f;
};

}  // namespace soundbubble
