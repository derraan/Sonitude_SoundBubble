#pragma once

#include <string>
#include <vector>

namespace soundbubble {

struct WavData {
  int sample_rate = 0;
  int channels = 0;
  std::vector<float> interleaved;
};

WavData read_wav(const std::string& path);
void write_wav(const std::string& path, int sample_rate, int channels, const std::vector<float>& interleaved);

}  // namespace soundbubble
