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

// UART console lines: "TAP on|off|toggle|flush" (case-insensitive).
// On/Arm and Off/Disarm are aliases so callers can use either spelling.
enum class TapUartCmd : int {
  None = 0,
  On = 1,
  Arm = 1,
  Off = 2,
  Disarm = 2,
  Toggle = 3,
  Flush = 4
};
TapUartCmd parse_tap_uart_line(std::string_view line);
inline TapUartCmd parse_tap_line(std::string_view line) { return parse_tap_uart_line(line); }

class MixControl {
 public:
  MixControl(std::string port, int baud, float initial_dry);
  ~MixControl();
  void start();
  void stop();
  float dry() const { return dry_.load(); }
  // Returns and clears the latest pending TAP UART command (None if idle).
  TapUartCmd take_tap_cmd();
  void ingest_for_test(std::string_view bytes);

 private:
  void run();
  void consume(std::string_view chunk);
  std::string port_;
  int baud_;
  std::atomic<float> dry_;
  std::atomic<int> tap_cmd_{0};
  std::atomic<bool> stop_{false};
  std::thread thread_;
  std::string pending_;
  bool started_ = false;
};

}  // namespace soundbubble
