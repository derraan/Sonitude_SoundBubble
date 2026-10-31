#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace soundbubble {

float dbfs_to_linear(float db);
float rms_dbfs(const float* samples, int n);

class StreamingResampler {
 public:
  StreamingResampler(int src_sr, int dst_sr, int num_channels, const std::string& quality = "VHQ");
  ~StreamingResampler();
  StreamingResampler(const StreamingResampler&) = delete;
  StreamingResampler& operator=(const StreamingResampler&) = delete;
  void reset();
  std::vector<float> process(const float* planar, int frames);
  const std::string& backend() const { return backend_; }

 private:
  void recreate();
  int src_sr_;
  int dst_sr_;
  int channels_;
  std::string quality_;
  std::string backend_;
  void* stream_ = nullptr;
  double phase_ = 0.0;
  std::vector<float> last_;
};

class DCOffsetFilter {
 public:
  DCOffsetFilter(int num_channels, float alpha);
  std::vector<float> process(const float* planar, int frames);
  void reset();

 private:
  int channels_;
  float alpha_;
  std::vector<float> prev_x_;
  std::vector<float> prev_y_;
};

class ChannelLevelValidator {
 public:
  ChannelLevelValidator(int num_channels, float silence_dbfs, float fault_min_dbfs, float fault_max_dbfs,
                        int silence_window_frames);
  struct Result {
    std::vector<float> audio;
    std::vector<float> levels_dbfs;
    std::vector<std::string> statuses;
  };
  Result validate(const float* planar, int frames);

 private:
  int channels_;
  float silence_dbfs_;
  float fault_min_;
  float fault_max_;
  int window_;
  std::deque<std::vector<double>> history_;
  std::vector<double> power_sum_;
};

std::vector<int> parse_channel_map(const std::string& spec, int input_channels, int model_num_ch);
std::vector<float> fit_channels(const float* planar, int channels, int frames, int target_channels);
std::vector<float> to_multichannel(const float* mono, int frames, int out_channels);

}  // namespace soundbubble
