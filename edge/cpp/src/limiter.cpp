#include "soundbubble/limiter.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace soundbubble {
namespace {

float update_gain(float amplitude, float ceiling, float release_step, float current) {
  float target = 1.f;
  if (amplitude > ceiling && amplitude > 0.f) {
    target = ceiling / amplitude;
  }
  if (target < current) return target;
  return std::min(1.f, current + release_step);
}

}  // namespace

void PeakLimiter::configure(const LimiterConfig& cfg, std::uint32_t sample_rate_hz) {
  if (sample_rate_hz == 0) throw std::runtime_error("limiter sample_rate_hz must be non-zero");
  cfg_ = cfg;
  cfg_.ceiling = std::clamp(cfg_.ceiling, 0.1f, 1.f);
  cfg_.release_ms = std::max(1.f, cfg_.release_ms);
  const float release_samples =
      std::max(1.f, (cfg_.release_ms * 0.001f) * static_cast<float>(sample_rate_hz));
  release_step_ = 1.f / release_samples;
  gain_ = 1.f;
  configured_ = true;
}

void PeakLimiter::reset() { gain_ = 1.f; }

void PeakLimiter::process_mono(std::span<float> samples) {
  if (!configured_) throw std::runtime_error("limiter used before configure");
  for (float& s : samples) {
    if (!std::isfinite(s)) s = 0.f;
    const float amp = std::fabs(s);
    gain_ = update_gain(amp, cfg_.ceiling, release_step_, gain_);
    s *= gain_;
  }
}

void PeakLimiter::process_interleaved_stereo(std::span<float> interleaved_lr) {
  if (!configured_) throw std::runtime_error("limiter used before configure");
  if (interleaved_lr.size() % 2u != 0u) {
    throw std::runtime_error("interleaved stereo span length must be even");
  }
  for (std::size_t i = 0; i + 1 < interleaved_lr.size(); i += 2) {
    float l = interleaved_lr[i];
    float r = interleaved_lr[i + 1];
    if (!std::isfinite(l)) l = 0.f;
    if (!std::isfinite(r)) r = 0.f;
    const float amp = std::max(std::fabs(l), std::fabs(r));
    gain_ = update_gain(amp, cfg_.ceiling, release_step_, gain_);
    interleaved_lr[i] = l * gain_;
    interleaved_lr[i + 1] = r * gain_;
  }
}

}  // namespace soundbubble
