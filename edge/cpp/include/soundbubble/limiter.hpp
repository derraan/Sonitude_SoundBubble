#pragma once

#include <cstdint>
#include <span>

namespace soundbubble {

// Sample-peak limiter: instant attack, linear release toward unity.
// No look-ahead — zero added latency. Linked stereo uses max(|L|,|R|).
struct LimiterConfig {
  float ceiling = 0.9440609f;  // −0.5 dBFS
  float release_ms = 25.f;
};

class PeakLimiter {
 public:
  void configure(const LimiterConfig& cfg, std::uint32_t sample_rate_hz);
  void reset();
  void process_mono(std::span<float> samples);
  void process_interleaved_stereo(std::span<float> interleaved_lr);
  float current_gain() const { return gain_; }

 private:
  LimiterConfig cfg_{};
  bool configured_ = false;
  float gain_ = 1.f;
  float release_step_ = 0.01f;
};

}  // namespace soundbubble
