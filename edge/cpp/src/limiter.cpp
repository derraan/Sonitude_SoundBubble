#include "soundbubble/limiter.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace soundbubble {
namespace {

// One-pole coefficient for a time constant (fraction of the gap closed per sample).
float tc_coeff(float ms, std::uint32_t sr) {
  const float n = std::max(1.f, ms * 0.001f * static_cast<float>(sr));
  return 1.f - std::exp(-1.f / n);
}

}  // namespace

void PeakLimiter::configure(const LimiterConfig& cfg, std::uint32_t sample_rate_hz) {
  if (sample_rate_hz == 0) throw std::runtime_error("limiter sample_rate_hz must be non-zero");
  cfg_ = cfg;
  cfg_.ceiling = std::clamp(cfg_.ceiling, 0.1f, 1.f);
  cfg_.release_ms = std::max(1.f, cfg_.release_ms);
  cfg_.hold_ms = std::max(0.f, cfg_.hold_ms);
  release_coeff_ = tc_coeff(cfg_.release_ms, sample_rate_hz);
  hold_samples_ = static_cast<int>(std::lround(cfg_.hold_ms * 0.001f * static_cast<float>(sample_rate_hz)));
  gain_ = 1.f;
  hold_left_ = 0;
  configured_ = true;
}

void PeakLimiter::reset() {
  gain_ = 1.f;
  hold_left_ = 0;
}

float PeakLimiter::next_gain(float amplitude) {
  float target = 1.f;
  if (amplitude > cfg_.ceiling && amplitude > 0.f) target = cfg_.ceiling / amplitude;
  if (target <= gain_) {
    // Instant attack keeps the ceiling hard; re-arm the hold on every clamp.
    gain_ = target;
    hold_left_ = hold_samples_;
  } else if (hold_left_ > 0) {
    --hold_left_;
  } else {
    gain_ += (1.f - gain_) * release_coeff_;
    if (gain_ > 0.99999f) gain_ = 1.f;
  }
  return gain_;
}

void PeakLimiter::process_mono(std::span<float> samples) {
  if (!configured_) throw std::runtime_error("limiter used before configure");
  for (float& s : samples) {
    if (!std::isfinite(s)) s = 0.f;
    s *= next_gain(std::fabs(s));
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
    const float g = next_gain(std::max(std::fabs(l), std::fabs(r)));
    interleaved_lr[i] = l * g;
    interleaved_lr[i + 1] = r * g;
  }
}

void ResidualGate::configure(const ResidualGateConfig& cfg, std::uint32_t sample_rate_hz) {
  if (sample_rate_hz == 0) throw std::runtime_error("gate sample_rate_hz must be non-zero");
  cfg_ = cfg;
  cfg_.ratio = std::max(1.f, cfg_.ratio);
  cfg_.range_db = std::max(0.f, cfg_.range_db);
  env_att_ = tc_coeff(1.f, sample_rate_hz);   // envelope: fast rise
  env_rel_ = tc_coeff(30.f, sample_rate_hz);  // envelope: smooth fall
  g_att_ = tc_coeff(std::max(0.1f, cfg_.attack_ms), sample_rate_hz);
  g_rel_ = tc_coeff(std::max(1.f, cfg_.release_ms), sample_rate_hz);
  reset();
  configured_ = true;
}

void ResidualGate::reset() {
  env_ = 0.f;
  gain_ = 1.f;
}

void ResidualGate::process_mono(std::span<float> samples) {
  if (!configured_) return;
  const float floor_lin = std::pow(10.f, -cfg_.range_db / 20.f);
  for (float& s : samples) {
    if (!std::isfinite(s)) s = 0.f;
    const float a = std::fabs(s);
    env_ += (a - env_) * (a > env_ ? env_att_ : env_rel_);
    const float lvl_db = 20.f * std::log10(std::max(env_, 1e-9f));
    float target = 1.f;
    if (lvl_db < cfg_.threshold_dbfs) {
      const float att_db = (cfg_.threshold_dbfs - lvl_db) * (cfg_.ratio - 1.f);
      target = std::max(floor_lin, std::pow(10.f, -att_db / 20.f));
    }
    gain_ += (target - gain_) * (target > gain_ ? g_att_ : g_rel_);
    s *= gain_;
  }
}

}  // namespace soundbubble
