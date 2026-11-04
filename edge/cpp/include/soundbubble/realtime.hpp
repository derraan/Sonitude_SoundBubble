#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace soundbubble {

struct DropStats {
  std::uint64_t dropped_oldest = 0;
};

template <typename T>
class DropOldestBuffer {
 public:
  explicit DropOldestBuffer(int capacity) : cap_(capacity < 1 ? 1 : capacity) {}

  int capacity() const { return cap_; }

  int size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<int>(q_.size());
  }

  void close() {
    std::lock_guard<std::mutex> lock(mu_);
    closed_ = true;
    cv_.notify_all();
  }

  void push(T item) {
    std::lock_guard<std::mutex> lock(mu_);
    if (closed_) return;
    if (static_cast<int>(q_.size()) >= cap_) {
      q_.pop_front();
      stats.dropped_oldest += 1;
    }
    q_.push_back(std::move(item));
    cv_.notify_one();
  }

  std::optional<T> pop(double timeout_sec) {
    std::unique_lock<std::mutex> lock(mu_);
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                        std::chrono::duration<double>(timeout_sec < 0 ? 0 : timeout_sec));
    while (q_.empty()) {
      if (closed_) return std::nullopt;
      if (timeout_sec < 0) {
        cv_.wait(lock);
      } else {
        if (cv_.wait_until(lock, deadline) == std::cv_status::timeout && q_.empty()) {
          return std::nullopt;
        }
      }
    }
    T item = std::move(q_.front());
    q_.pop_front();
    return item;
  }

  std::pair<std::optional<T>, int> pop_latest(double timeout_sec) {
    auto item = pop(timeout_sec);
    if (!item) return {std::nullopt, 0};
    int drained = 0;
    while (true) {
      auto nxt = pop(0.0);
      if (!nxt) break;
      item = std::move(nxt);
      drained += 1;
    }
    return {std::move(item), drained};
  }

  DropStats stats;

 private:
  int cap_;
  std::deque<T> q_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  bool closed_ = false;
};

struct RingStats {
  std::uint64_t dropped_samples = 0;
};

class AudioRingBuffer {
 public:
  AudioRingBuffer(int num_channels, int capacity_samples);
  int size() const { return size_; }
  int capacity() const { return capacity_; }
  void clear();
  void push(const float* planar, int channels, int frames);
  bool pop(int n, std::vector<float>& out_planar);

  RingStats stats;

 private:
  int channels_;
  int capacity_;
  int write_ = 0;
  int size_ = 0;
  std::vector<float> buf_;
};

struct OutputRingStats {
  std::uint64_t dropped_oldest_frames = 0;
  std::uint64_t underrun_frames = 0;
  std::uint64_t write_calls = 0;
  std::uint64_t read_calls = 0;
};

// Declick length for ring discontinuities (underrun fade, resume fade-in,
// trim/overflow splice crossfade). 48 frames = 2 ms @ 24 kHz.
inline constexpr int kRingDeclickFrames = 48;
// Splices shorter than this are applied hard (tiny test rings / degenerate cases).
inline constexpr int kRingMinCrossfadeFrames = 8;

class OutputRingBuffer {
 public:
  OutputRingBuffer(int num_channels, int capacity_frames, int declick_frames = kRingDeclickFrames);
  int capacity_frames() const { return capacity_; }
  int ahead_frames() const;
  int free_frames() const;
  void clear();
  // Drop the oldest n frames; the splice is crossfaded (no hard jump).
  int drop_oldest_frames(int n);
  int write_drop_oldest(const float* interleaved, int frames);
  int write_silence(int frames);
  // Fade the newest unread frames to zero (call before appending a silent gap
  // after audio so the gap does not start with a step).
  void fade_tail(int frames);
  // Underrun: decays the last delivered frame to zero over declick_frames
  // (never a DC hold), then fades the next real audio back in.
  int read_into(float* interleaved, int frames);
  std::uint64_t underrun_frames() const;

  OutputRingStats stats;

 private:
  void crossfade_splice_locked(int old_read, int dropped);

  int channels_;
  int capacity_;
  int declick_;
  int read_ = 0;
  int write_ = 0;
  int size_ = 0;
  std::vector<float> buf_;
  std::vector<float> last_frame_;  // last pre-gap frame (decays during underrun)
  float hold_gain_ = 0.f;          // residual weight of last_frame_ (1 → 0)
  int fade_in_pos_ = 0;            // < declick_ while fading audio back in
  mutable std::mutex mu_;
};

struct RealtimeContract {
  int max_capture_blocks = 4;
  double max_output_ahead_sec = 0.8;
  double target_output_ahead_sec = 0.4;
  int rt_fifo_priority = 0;
  void validate() const;
};

struct SchedulingResult {
  std::string requested_policy = "SCHED_FIFO";
  int requested_priority = 0;
  bool applied = false;
  std::string message;
};

SchedulingResult try_set_realtime_fifo(int priority);
void try_lock_memory();
void try_set_mmcss();

// Windows: timeBeginPeriod for the live session. No-op on other platforms.
// Pi SCHED_FIFO + mlockall stay in try_set_realtime_fifo / try_lock_memory,
// launched under `chrt -f` with LimitMEMLOCK=infinity.
class ScopedTimerResolution {
 public:
  explicit ScopedTimerResolution(unsigned period_ms);
  ~ScopedTimerResolution();
  ScopedTimerResolution(const ScopedTimerResolution&) = delete;
  ScopedTimerResolution& operator=(const ScopedTimerResolution&) = delete;

 private:
  unsigned period_ms_ = 1;
  bool active_ = false;
};

struct OutputRingParams {
  int ring_n_frames = 0;
  int ring_max_ahead_frames = 0;
  int ring_target_ahead_frames = 0;
};

OutputRingParams compute_output_ring_params(int output_sr, int output_block, double max_ahead_sec,
                                            double target_ahead_sec);

}  // namespace soundbubble
