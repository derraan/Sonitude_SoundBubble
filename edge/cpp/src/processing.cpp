#include "soundbubble/processing.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <stdexcept>

#if defined(SOUNDBUBBLE_HAS_SOXR)
#include <soxr.h>
#endif

namespace soundbubble {

float dbfs_to_linear(float db) { return std::pow(10.f, db / 20.f); }

float rms_dbfs(const float* samples, int n) {
  if (!samples || n <= 0) return -120.f;
  double acc = 0.0;
  for (int i = 0; i < n; ++i) {
    double v = static_cast<double>(samples[i]);
    acc += v * v;
  }
  double rms = std::sqrt(acc / static_cast<double>(n) + 1e-12);
  return static_cast<float>(20.0 * std::log10(std::max(rms, 1e-10)));
}

StreamingResampler::StreamingResampler(int src_sr, int dst_sr, int num_channels, const std::string& quality)
    : src_sr_(src_sr), dst_sr_(dst_sr), channels_(num_channels), quality_(quality), last_(static_cast<size_t>(num_channels), 0.f) {
  if (src_sr_ <= 0 || dst_sr_ <= 0 || channels_ <= 0) throw std::invalid_argument("resampler rates");
  recreate();
}

StreamingResampler::~StreamingResampler() {
#if defined(SOUNDBUBBLE_HAS_SOXR)
  if (stream_) soxr_delete(static_cast<soxr_t>(stream_));
#endif
}

void StreamingResampler::recreate() {
#if defined(SOUNDBUBBLE_HAS_SOXR)
  if (stream_) {
    soxr_delete(static_cast<soxr_t>(stream_));
    stream_ = nullptr;
  }
#endif
  phase_ = 0.0;
  std::fill(last_.begin(), last_.end(), 0.f);
  if (src_sr_ == dst_sr_) {
    backend_ = "copy";
    return;
  }
#if defined(SOUNDBUBBLE_HAS_SOXR)
  unsigned long recipe = SOXR_VHQ;
  if (quality_ == "HQ") recipe = SOXR_HQ;
  else if (quality_ == "MQ") recipe = SOXR_MQ;
  else if (quality_ == "LQ") recipe = SOXR_LQ;
  soxr_io_spec_t io = soxr_io_spec(SOXR_FLOAT32_I, SOXR_FLOAT32_I);
  soxr_quality_spec_t q = soxr_quality_spec(recipe, 0);
  soxr_error_t err = nullptr;
  soxr_t s = soxr_create(static_cast<double>(src_sr_), static_cast<double>(dst_sr_), static_cast<unsigned>(channels_), &err, &io, &q, nullptr);
  if (!s) throw std::runtime_error(err ? err : "soxr_create failed");
  stream_ = s;
  backend_ = "soxr-stream";
#else
  backend_ = "linear-fallback";
#endif
}

void StreamingResampler::reset() { recreate(); }

std::vector<float> StreamingResampler::process(const float* planar, int frames) {
  if (frames < 0) throw std::invalid_argument("frames");
  if (frames == 0 || src_sr_ == dst_sr_) {
    std::vector<float> y(static_cast<size_t>(channels_ * std::max(0, frames)));
    if (frames > 0) std::copy(planar, planar + y.size(), y.begin());
    return y;
  }
#if defined(SOUNDBUBBLE_HAS_SOXR)
  if (stream_) {
    std::vector<float> inter(static_cast<size_t>(frames * channels_));
    for (int t = 0; t < frames; ++t) {
      for (int c = 0; c < channels_; ++c) inter[static_cast<size_t>(t * channels_ + c)] = planar[static_cast<size_t>(c * frames + t)];
    }
    size_t out_cap = static_cast<size_t>(std::max(8, static_cast<int>(std::ceil(frames * (static_cast<double>(dst_sr_) / src_sr_))) + 64));
    std::vector<float> out_i(out_cap * static_cast<size_t>(channels_));
    size_t idone = 0;
    size_t odone = 0;
    soxr_error_t err = soxr_process(static_cast<soxr_t>(stream_), inter.data(), static_cast<size_t>(frames), &idone, out_i.data(), out_cap, &odone);
    if (err) throw std::runtime_error(err);
    std::vector<float> planar_out(static_cast<size_t>(channels_ * static_cast<int>(odone)));
    for (size_t t = 0; t < odone; ++t) {
      for (int c = 0; c < channels_; ++c) {
        planar_out[static_cast<size_t>(c * static_cast<int>(odone) + static_cast<int>(t))] = out_i[t * static_cast<size_t>(channels_) + static_cast<size_t>(c)];
      }
    }
    return planar_out;
  }
#endif
  double ratio = static_cast<double>(dst_sr_) / static_cast<double>(src_sr_);
  int out_n = static_cast<int>(std::llround(frames * ratio));
  if (out_n < 0) out_n = 0;
  std::vector<float> y(static_cast<size_t>(channels_ * out_n), 0.f);
  for (int i = 0; i < out_n; ++i) {
    double src_pos = (static_cast<double>(i) + phase_) / ratio;
    int i0 = static_cast<int>(std::floor(src_pos));
    float frac = static_cast<float>(src_pos - i0);
    for (int c = 0; c < channels_; ++c) {
      float a = (i0 < 0 || i0 >= frames) ? last_[static_cast<size_t>(c)] : planar[static_cast<size_t>(c * frames + std::min(i0, frames - 1))];
      int i1 = i0 + 1;
      float b = (i1 < 0 || i1 >= frames) ? a : planar[static_cast<size_t>(c * frames + i1)];
      if (i0 < 0) a = last_[static_cast<size_t>(c)];
      y[static_cast<size_t>(c * out_n + i)] = a * (1.f - frac) + b * frac;
    }
  }
  if (frames > 0) {
    for (int c = 0; c < channels_; ++c) last_[static_cast<size_t>(c)] = planar[static_cast<size_t>(c * frames + frames - 1)];
  }
  phase_ = std::fmod(phase_ + frames * ratio - out_n, 1.0);
  return y;
}

DCOffsetFilter::DCOffsetFilter(int num_channels, float alpha)
    : channels_(num_channels), alpha_(alpha), prev_x_(static_cast<size_t>(num_channels), 0.f), prev_y_(static_cast<size_t>(num_channels), 0.f) {}

void DCOffsetFilter::reset() {
  std::fill(prev_x_.begin(), prev_x_.end(), 0.f);
  std::fill(prev_y_.begin(), prev_y_.end(), 0.f);
}

std::vector<float> DCOffsetFilter::process(const float* planar, int frames) {
  std::vector<float> y(static_cast<size_t>(channels_ * frames));
  if (frames <= 0) return y;
  for (int c = 0; c < channels_; ++c) {
    float px = prev_x_[static_cast<size_t>(c)];
    float py = prev_y_[static_cast<size_t>(c)];
    for (int t = 0; t < frames; ++t) {
      float x = planar[static_cast<size_t>(c * frames + t)];
      float o = x - px + alpha_ * py;
      y[static_cast<size_t>(c * frames + t)] = o;
      px = x;
      py = o;
    }
    prev_x_[static_cast<size_t>(c)] = px;
    prev_y_[static_cast<size_t>(c)] = py;
  }
  return y;
}

ChannelLevelValidator::ChannelLevelValidator(int num_channels, float silence_dbfs, float fault_min_dbfs, float fault_max_dbfs,
                                             int silence_window_frames)
    : channels_(num_channels),
      silence_dbfs_(silence_dbfs),
      fault_min_(fault_min_dbfs),
      fault_max_(fault_max_dbfs),
      window_(std::max(1, silence_window_frames)),
      power_sum_(static_cast<size_t>(num_channels), 0.0) {}

ChannelLevelValidator::Result ChannelLevelValidator::validate(const float* planar, int frames) {
  Result r;
  r.audio.assign(planar, planar + static_cast<size_t>(channels_ * std::max(0, frames)));
  r.levels_dbfs.assign(static_cast<size_t>(channels_), 0.f);
  r.statuses.assign(static_cast<size_t>(channels_), "ok");
  if (frames <= 0) return r;
  std::vector<double> frame_power(static_cast<size_t>(channels_), 0.0);
  for (int c = 0; c < channels_; ++c) {
    double acc = 0;
    for (int t = 0; t < frames; ++t) {
      float v = planar[static_cast<size_t>(c * frames + t)];
      acc += static_cast<double>(v) * static_cast<double>(v);
    }
    frame_power[static_cast<size_t>(c)] = acc / static_cast<double>(frames);
  }
  if (static_cast<int>(history_.size()) == window_) {
    const auto& oldest = history_.front();
    for (int c = 0; c < channels_; ++c) power_sum_[static_cast<size_t>(c)] -= oldest[static_cast<size_t>(c)];
    history_.pop_front();
  }
  history_.push_back(frame_power);
  for (int c = 0; c < channels_; ++c) power_sum_[static_cast<size_t>(c)] += frame_power[static_cast<size_t>(c)];
  double denom = static_cast<double>(std::max<size_t>(1, history_.size()));
  for (int c = 0; c < channels_; ++c) {
    double rms = std::sqrt(frame_power[static_cast<size_t>(c)] + 1e-12);
    float dbfs = static_cast<float>(20.0 * std::log10(std::max(rms, 1e-10)));
    r.levels_dbfs[static_cast<size_t>(c)] = dbfs;
    double rolling = power_sum_[static_cast<size_t>(c)] / denom;
    float rolling_db = static_cast<float>(20.0 * std::log10(std::max(std::sqrt(rolling + 1e-12), 1e-10)));
    if (rolling_db < silence_dbfs_) {
      std::fill(r.audio.begin() + static_cast<size_t>(c * frames), r.audio.begin() + static_cast<size_t>((c + 1) * frames), 0.f);
      r.statuses[static_cast<size_t>(c)] = "silence";
    } else {
      // Fault silencer bypassed: near-FS frames stay in the wet path (status ok).
      r.statuses[static_cast<size_t>(c)] = "ok";
    }
  }
  return r;
}

std::vector<int> parse_channel_map(const std::string& spec, int input_channels, int model_num_ch) {
  if (spec.empty()) {
    std::vector<int> v;
    int n = std::min(input_channels, model_num_ch);
    for (int i = 0; i < n; ++i) v.push_back(i);
    return v;
  }
  std::vector<int> map;
  std::string token;
  for (size_t i = 0; i <= spec.size(); ++i) {
    if (i == spec.size() || spec[i] == ',') {
      if (!token.empty()) {
        int idx = std::stoi(token);
        if (idx < 0 || idx >= input_channels) throw std::invalid_argument("channel-map index out of range");
        map.push_back(idx);
      }
      token.clear();
    } else if (!std::isspace(static_cast<unsigned char>(spec[i]))) {
      token.push_back(spec[i]);
    }
  }
  if (map.empty()) throw std::invalid_argument("channel-map empty");
  if (static_cast<int>(map.size()) > model_num_ch) throw std::invalid_argument("channel-map longer than model channels");
  return map;
}

std::vector<float> fit_channels(const float* planar, int channels, int frames, int target_channels) {
  if (channels == target_channels) return std::vector<float>(planar, planar + static_cast<size_t>(channels * frames));
  std::vector<float> out(static_cast<size_t>(target_channels * frames), 0.f);
  int keep = std::min(channels, target_channels);
  for (int c = 0; c < keep; ++c) {
    std::copy(planar + static_cast<size_t>(c * frames), planar + static_cast<size_t>((c + 1) * frames),
              out.begin() + static_cast<size_t>(c * frames));
  }
  if (channels < target_channels && channels > 0) {
    int filled = channels;
    while (filled < target_channels) {
      int src = filled % channels;
      int n = std::min(channels, target_channels - filled);
      for (int k = 0; k < n; ++k) {
        int dst = filled + k;
        int s = (src + k) % channels;
        std::copy(planar + static_cast<size_t>(s * frames), planar + static_cast<size_t>((s + 1) * frames),
                  out.begin() + static_cast<size_t>(dst * frames));
      }
      filled += n;
    }
  }
  return out;
}

std::vector<float> to_multichannel(const float* mono, int frames, int out_channels) {
  std::vector<float> y(static_cast<size_t>(frames * out_channels));
  for (int t = 0; t < frames; ++t) {
    for (int c = 0; c < out_channels; ++c) y[static_cast<size_t>(t * out_channels + c)] = mono[t];
  }
  return y;
}

}  // namespace soundbubble
