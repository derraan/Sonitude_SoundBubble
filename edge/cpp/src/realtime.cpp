#include "soundbubble/realtime.hpp"

#include <algorithm>
#include <stdexcept>

namespace soundbubble {

AudioRingBuffer::AudioRingBuffer(int num_channels, int capacity_samples)
    : channels_(num_channels), capacity_(std::max(1, capacity_samples)), buf_(static_cast<size_t>(channels_ * capacity_), 0.f) {}

void AudioRingBuffer::clear() {
  write_ = 0;
  size_ = 0;
}

void AudioRingBuffer::push(const float* planar, int channels, int frames) {
  if (channels != channels_) throw std::invalid_argument("AudioRingBuffer channel mismatch");
  if (frames <= 0) return;
  if (frames >= capacity_) {
    int dropped = size_ + (frames - capacity_);
    for (int c = 0; c < channels_; ++c) {
      std::copy(planar + static_cast<size_t>(c * frames + (frames - capacity_)),
                planar + static_cast<size_t>(c * frames + frames),
                buf_.begin() + static_cast<size_t>(c * capacity_));
    }
    write_ = 0;
    size_ = capacity_;
    stats.dropped_samples += static_cast<std::uint64_t>(std::max(0, dropped));
    return;
  }
  int overflow = std::max(0, (size_ + frames) - capacity_);
  if (overflow) {
    size_ -= overflow;
    stats.dropped_samples += static_cast<std::uint64_t>(overflow);
  }
  int w = write_;
  int end = w + frames;
  if (end <= capacity_) {
    for (int c = 0; c < channels_; ++c) {
      std::copy(planar + static_cast<size_t>(c * frames), planar + static_cast<size_t>(c * frames + frames),
                buf_.begin() + static_cast<size_t>(c * capacity_ + w));
    }
  } else {
    int n1 = capacity_ - w;
    for (int c = 0; c < channels_; ++c) {
      std::copy(planar + static_cast<size_t>(c * frames), planar + static_cast<size_t>(c * frames + n1),
                buf_.begin() + static_cast<size_t>(c * capacity_ + w));
      std::copy(planar + static_cast<size_t>(c * frames + n1), planar + static_cast<size_t>(c * frames + frames),
                buf_.begin() + static_cast<size_t>(c * capacity_));
    }
  }
  write_ = end % capacity_;
  size_ = std::min(capacity_, size_ + frames);
}

bool AudioRingBuffer::pop(int n, std::vector<float>& out_planar) {
  if (n <= 0) throw std::invalid_argument("n must be > 0");
  if (size_ < n) return false;
  out_planar.assign(static_cast<size_t>(channels_ * n), 0.f);
  int start = (write_ - size_) % capacity_;
  if (start < 0) start += capacity_;
  int end = start + n;
  if (end <= capacity_) {
    for (int c = 0; c < channels_; ++c) {
      std::copy(buf_.begin() + static_cast<size_t>(c * capacity_ + start),
                buf_.begin() + static_cast<size_t>(c * capacity_ + end),
                out_planar.begin() + static_cast<size_t>(c * n));
    }
  } else {
    int n1 = capacity_ - start;
    for (int c = 0; c < channels_; ++c) {
      std::copy(buf_.begin() + static_cast<size_t>(c * capacity_ + start),
                buf_.begin() + static_cast<size_t>(c * capacity_ + capacity_),
                out_planar.begin() + static_cast<size_t>(c * n));
      std::copy(buf_.begin() + static_cast<size_t>(c * capacity_),
                buf_.begin() + static_cast<size_t>(c * capacity_ + (end % capacity_)),
                out_planar.begin() + static_cast<size_t>(c * n + n1));
    }
  }
  size_ -= n;
  return true;
}

OutputRingBuffer::OutputRingBuffer(int num_channels, int capacity_frames, int declick_frames)
    : channels_(num_channels),
      capacity_(std::max(1, capacity_frames)),
      declick_(std::max(1, declick_frames)),
      buf_(static_cast<size_t>(capacity_ * channels_), 0.f),
      last_frame_(static_cast<size_t>(std::max(1, num_channels)), 0.f),
      fade_in_pos_(std::max(1, declick_frames)) {}

void OutputRingBuffer::crossfade_splice_locked(int old_read, int dropped) {
  // read_ already advanced past `dropped` frames. Blend the dropped head into
  // the new head so the stream continues without a waveform jump.
  int k = std::min({declick_, size_, dropped});
  if (k < kRingMinCrossfadeFrames) return;
  for (int i = 0; i < k; ++i) {
    const float w = static_cast<float>(i + 1) / static_cast<float>(k + 1);
    float* dst = &buf_[static_cast<size_t>(((read_ + i) % capacity_) * channels_)];
    const float* src = &buf_[static_cast<size_t>(((old_read + i) % capacity_) * channels_)];
    for (int c = 0; c < channels_; ++c) dst[c] = w * dst[c] + (1.f - w) * src[c];
  }
}

void OutputRingBuffer::fade_tail(int frames) {
  std::lock_guard<std::mutex> lock(mu_);
  int k = std::min(std::max(0, frames), size_);
  for (int i = 0; i < k; ++i) {
    // i = 0 is the newest frame (gain → 0), i = k-1 nearly unity.
    const float g = static_cast<float>(i) / static_cast<float>(k);
    int idx = (write_ - 1 - i) % capacity_;
    if (idx < 0) idx += capacity_;
    float* f = &buf_[static_cast<size_t>(idx * channels_)];
    for (int c = 0; c < channels_; ++c) f[c] *= g;
  }
}

int OutputRingBuffer::ahead_frames() const {
  std::lock_guard<std::mutex> lock(mu_);
  return size_;
}

int OutputRingBuffer::free_frames() const {
  std::lock_guard<std::mutex> lock(mu_);
  return capacity_ - size_;
}

void OutputRingBuffer::clear() {
  std::lock_guard<std::mutex> lock(mu_);
  read_ = 0;
  write_ = 0;
  size_ = 0;
  // Keep last_frame_/hold_gain_: the reader decays the residual, and whatever
  // is written next fades in (a clear is a stream discontinuity).
  fade_in_pos_ = 0;
}

int OutputRingBuffer::drop_oldest_frames(int n) {
  n = std::max(0, n);
  if (n <= 0) return 0;
  std::lock_guard<std::mutex> lock(mu_);
  int dropped = std::min(n, size_);
  if (dropped) {
    const int old_read = read_;
    read_ = (read_ + dropped) % capacity_;
    size_ -= dropped;
    stats.dropped_oldest_frames += static_cast<std::uint64_t>(dropped);
    crossfade_splice_locked(old_read, dropped);
  }
  return dropped;
}

int OutputRingBuffer::write_silence(int frames) {
  if (frames <= 0) return 0;
  std::vector<float> z(static_cast<size_t>(frames * channels_), 0.f);
  return write_drop_oldest(z.data(), frames);
}

std::uint64_t OutputRingBuffer::underrun_frames() const {
  std::lock_guard<std::mutex> lock(mu_);
  return stats.underrun_frames;
}

int OutputRingBuffer::write_drop_oldest(const float* interleaved, int frames) {
  if (frames <= 0) return 0;
  int n = frames;
  const float* x = interleaved;
  std::vector<float> tail;
  if (n >= capacity_) {
    x = interleaved + static_cast<size_t>((n - capacity_) * channels_);
    n = capacity_;
  }
  std::lock_guard<std::mutex> lock(mu_);
  int free = capacity_ - size_;
  int dropped = std::max(0, n - free);
  if (dropped) {
    const int old_read = read_;
    read_ = (read_ + dropped) % capacity_;
    size_ -= dropped;
    stats.dropped_oldest_frames += static_cast<std::uint64_t>(dropped);
    // Blend before the new block overwrites the dropped region.
    crossfade_splice_locked(old_read, dropped);
  }
  int end = write_ + n;
  if (end <= capacity_) {
    std::copy(x, x + static_cast<size_t>(n * channels_), buf_.begin() + static_cast<size_t>(write_ * channels_));
  } else {
    int n1 = capacity_ - write_;
    std::copy(x, x + static_cast<size_t>(n1 * channels_), buf_.begin() + static_cast<size_t>(write_ * channels_));
    std::copy(x + static_cast<size_t>(n1 * channels_), x + static_cast<size_t>(n * channels_), buf_.begin());
  }
  write_ = end % capacity_;
  size_ += n;
  stats.write_calls += 1;
  return dropped;
}

int OutputRingBuffer::read_into(float* interleaved, int frames) {
  if (frames <= 0) return 0;
  std::lock_guard<std::mutex> lock(mu_);
  int take = std::min(frames, size_);
  if (take > 0) {
    int end = read_ + take;
    if (end <= capacity_) {
      std::copy(buf_.begin() + static_cast<size_t>(read_ * channels_),
                buf_.begin() + static_cast<size_t>(end * channels_), interleaved);
    } else {
      int n1 = capacity_ - read_;
      std::copy(buf_.begin() + static_cast<size_t>(read_ * channels_),
                buf_.begin() + static_cast<size_t>(capacity_ * channels_), interleaved);
      std::copy(buf_.begin(), buf_.begin() + static_cast<size_t>((end % capacity_) * channels_),
                interleaved + static_cast<size_t>(n1 * channels_));
    }
    read_ = end % capacity_;
    size_ -= take;
  }
  const float step = 1.f / static_cast<float>(declick_);
  // Resume after a gap: crossfade from the decaying pre-gap residual into audio.
  int i = 0;
  for (; i < take && fade_in_pos_ < declick_; ++i, ++fade_in_pos_) {
    const float r = static_cast<float>(fade_in_pos_ + 1) / static_cast<float>(declick_);
    float* f = interleaved + static_cast<size_t>(i * channels_);
    for (int c = 0; c < channels_; ++c) f[c] = r * f[c] + (1.f - r) * hold_gain_ * last_frame_[static_cast<size_t>(c)];
    hold_gain_ = std::max(0.f, hold_gain_ - step);
  }
  if (take > 0 && fade_in_pos_ >= declick_) {
    std::copy(interleaved + static_cast<size_t>((take - 1) * channels_),
              interleaved + static_cast<size_t>(take * channels_), last_frame_.begin());
    hold_gain_ = 1.f;
  }
  if (take < frames) {
    // Underrun: decay the last frame to zero over declick_ frames. A DC hold
    // (old behaviour) stepped to a flat level and jumped back on resume —
    // at block rate that is an audible buzz once output gain is applied.
    for (int j = take; j < frames; ++j) {
      hold_gain_ = std::max(0.f, hold_gain_ - step);
      float* f = interleaved + static_cast<size_t>(j * channels_);
      for (int c = 0; c < channels_; ++c) f[c] = hold_gain_ * last_frame_[static_cast<size_t>(c)];
    }
    fade_in_pos_ = 0;
    stats.underrun_frames += static_cast<std::uint64_t>(frames - take);
  }
  stats.read_calls += 1;
  return take;
}

void RealtimeContract::validate() const {
  if (max_capture_blocks < 1) throw std::invalid_argument("max_capture_blocks must be >= 1");
  if (max_output_ahead_sec <= 0) throw std::invalid_argument("max_output_ahead_sec must be > 0");
  if (target_output_ahead_sec <= 0) throw std::invalid_argument("target_output_ahead_sec must be > 0");
  if (target_output_ahead_sec >= max_output_ahead_sec) {
    throw std::invalid_argument("target_output_ahead_sec must be < max_output_ahead_sec");
  }
  if (rt_fifo_priority < 0) throw std::invalid_argument("rt_fifo_priority must be >= 0");
}

OutputRingParams compute_output_ring_params(int output_sr, int output_block, double max_ahead_sec, double target_ahead_sec) {
  int sr = output_sr;
  int block = output_block;
  int ring_n = std::max(block * 64, static_cast<int>(4.0 * sr));
  int ring_max = std::max(block * 3, static_cast<int>(max_ahead_sec * sr));
  int ring_target = std::max(block, static_cast<int>(target_ahead_sec * sr));
  if (ring_target >= ring_max) ring_target = std::max(block, ring_max / 2);
  if (ring_n < ring_max * 2) ring_n = ring_max * 2;
  return OutputRingParams{ring_n, ring_max, ring_target};
}

}  // namespace soundbubble
