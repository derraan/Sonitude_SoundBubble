#pragma once

#include <atomic>
#include <string>
#include <string_view>
#include <thread>

namespace soundbubble {

struct MixLine {
  bool ok = false;
  float wet = 0.f;
  float dry = 0.f;
};

MixLine parse_mix_line(std::string_view line);

class MixControl {
 public:
  MixControl(std::string port, int baud, float initial_dry);
  ~MixControl();
  void start();
  void stop();
  float dry() const { return dry_.load(); }
  void ingest_for_test(std::string_view bytes);

 private:
  void run();
  void consume(std::string_view chunk);
  std::string port_;
  int baud_;
  std::atomic<float> dry_;
  std::atomic<bool> stop_{false};
  std::thread thread_;
  std::string pending_;
  bool started_ = false;
};

}  // namespace soundbubble
