#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace soundbubble {

struct DeviceInfo {
  int index = 0;
  std::string name;
  std::string hostapi;
  int max_input_channels = 0;
  int max_output_channels = 0;
  double default_samplerate = 0;
};

std::vector<DeviceInfo> list_devices();
void print_devices();

struct AudioFormat {
  std::optional<int> device;
  int channels = 0;
  int sample_rate = 0;
};

void preflight_devices(const AudioFormat& input, const AudioFormat& output);
// True full-duplex: both directions must be supported on one stream at one rate.
void preflight_duplex(const AudioFormat& input, const AudioFormat& output);

// Two-stream open (input-only + output-only). Used when duplex_mode is off.
class DuplexSession {
 public:
  using InputCb = std::function<void(const float* interleaved, int frames, int channels)>;
  using OutputCb = std::function<void(float* interleaved, int frames, int channels)>;
  using StatusCb = std::function<void(bool input, const char* status)>;

  DuplexSession(AudioFormat input, AudioFormat output, int input_block, int output_block, const std::string& latency,
                InputCb in_cb, OutputCb out_cb, StatusCb status_cb);
  ~DuplexSession();
  DuplexSession(const DuplexSession&) = delete;
  DuplexSession& operator=(const DuplexSession&) = delete;
  void start();
  void stop();
  bool active() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// True single-stream full-duplex: one Pa_OpenStream with in+out params, one callback.
class DuplexSyncSession {
 public:
  using DuplexCb = std::function<void(const float* in, float* out, int frames, int in_ch, int out_ch)>;
  using StatusCb = std::function<void(const char* status)>;

  DuplexSyncSession(AudioFormat input, AudioFormat output, int frames_per_buffer, const std::string& latency,
                    DuplexCb cb, StatusCb status_cb = {});
  ~DuplexSyncSession();
  DuplexSyncSession(const DuplexSyncSession&) = delete;
  DuplexSyncSession& operator=(const DuplexSyncSession&) = delete;
  void start();
  void stop();
  bool active() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace soundbubble
