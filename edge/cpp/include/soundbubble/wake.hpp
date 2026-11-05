#pragma once

#include "soundbubble/realtime.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace soundbubble {

class DryHistory {
 public:
  explicit DryHistory(int capacity_frames);
  std::int64_t write_index() const { return write_index_; }
  std::int64_t oldest_index() const;
  void clear();
  void write(const float* mono, int n);
  std::pair<std::vector<float>, double> read_fractional(double start_index, double rate, int n_frames) const;

 private:
  int capacity_;
  std::vector<float> buf_;
  std::int64_t write_index_ = 0;
};

class BubbleOpenController {
 public:
  static constexpr const char* STATE_IDLE = "idle";
  static constexpr const char* STATE_OPEN_REPLAY = "open_replay";
  static constexpr const char* STATE_OPEN_LIVE = "open_live";
  static constexpr const char* STATE_RELEASING = "releasing";

  BubbleOpenController(int output_sr, float base_mix_dry, float replay_sec, float replay_rate, float hold_sec,
                       float attack_ms = 20.f, float release_ms = 300.f);
  void trigger(const std::string& person);
  void set_base_mix(float mix_dry);
  std::pair<std::vector<float>, std::vector<float>> process_block(const float* live_dry, int n, const DryHistory& history);
  struct Status {
    std::string state;
    float mix_current = 0;
    float mix_target = 0;
    int hold_frames_left = 0;
    int trigger_count = 0;
    std::string last_person;
  };
  Status status() const;

 private:
  void set_mix_target_locked(float target, int duration_frames);
  void activate_trigger_locked(const DryHistory& history);
  std::vector<float> next_mix_locked(int n_frames);

  int output_sr_;
  float base_mix_dry_;
  int replay_frames_;
  float replay_rate_;
  int hold_frames_total_;
  int attack_frames_;
  int release_frames_;
  mutable std::mutex mu_;
  std::string state_;
  float mix_current_;
  float mix_target_;
  int mix_frames_left_ = 0;
  int hold_frames_left_ = 0;
  double replay_pos_ = 0;
  bool pending_trigger_ = false;
  std::string last_person_;
  int trigger_count_ = 0;
};

std::unordered_map<std::string, std::string> parse_phrase_entries(const std::vector<std::string>& entries);
std::unordered_map<std::string, std::string> load_phrase_book_json(const std::string& path);
std::unordered_map<std::string, std::string> build_phrases(const std::string& phrase_book_path,
                                                           const std::vector<std::string>& cli_entries);

struct KeywordLine {
  std::string line;
  std::string label;
  std::string person;
};

std::vector<KeywordLine> build_keyword_lines(
    const std::unordered_map<std::string, std::string>& phrases,
    const std::function<std::vector<std::string>(const std::string&)>& encode,
    const std::unordered_set<std::string>& vocabulary);

class WakeWord {
 public:
  WakeWord(const std::string& model_dir, const std::unordered_map<std::string, std::string>& phrases, float threshold,
           double cooldown_seconds);
  ~WakeWord();
  WakeWord(const WakeWord&) = delete;
  WakeWord& operator=(const WakeWord&) = delete;
  void reset();
  std::optional<std::string> feed(const float* samples, int n);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class WakeWordWorker {
 public:
  WakeWordWorker(const std::string& model_dir, const std::unordered_map<std::string, std::string>& phrases,
                 float threshold, double cooldown_seconds = 2.0, int queue_capacity_blocks = 16);
  ~WakeWordWorker();
  void start(int input_sr, std::function<void(const std::string&)> trigger_cb,
             std::function<void(const std::string&)> error_cb);
  void stop();
  void push_input_block(const float* mono, int n);
  struct Stats {
    std::uint64_t queue_drops = 0;
    int feed_calls = 0;
    double feed_mean_ms = 0;
    int trigger_count = 0;
    std::string last_person;
  };
  Stats stats() const;

 private:
  void run(int input_sr);
  std::unique_ptr<WakeWord> detector_;
  DropOldestBuffer<std::vector<float>> queue_;
  std::atomic<bool> stop_{false};
  std::thread thread_;
  std::function<void(const std::string&)> trigger_cb_;
  std::function<void(const std::string&)> error_cb_;
  mutable std::mutex mu_;
  double feed_total_ms_ = 0;
  int feed_calls_ = 0;
  int trigger_count_ = 0;
  std::string last_person_;
};

class WakeRuntime {
 public:
  WakeRuntime(int input_sr, int output_sr, int history_frames, float base_mix, float replay_sec, float replay_rate,
              float hold_sec, float release_ms, const std::string& model_dir,
              const std::unordered_map<std::string, std::string>& phrases, float threshold);
  ~WakeRuntime();
  void push_input(const float* mono, int n);
  std::pair<std::vector<float>, std::vector<float>> mix(const float* live_dry, int n);
  void set_base_mix(float mix_dry);
  void stop();
  std::string profile_suffix() const;

 private:
  std::unique_ptr<DryHistory> history_;
  std::unique_ptr<BubbleOpenController> ctrl_;
  std::unique_ptr<WakeWordWorker> worker_;
  bool enabled_ = false;
};

}  // namespace soundbubble
