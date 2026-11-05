#pragma once

#include <optional>
#include <string>
#include <vector>

namespace soundbubble {

struct RuntimeContract {
  int model_sr = 24000;
  int model_num_ch = 6;
  int model_chunk = 192;
  int model_pad = 96;
  int frame_len = 288;
  int expected_input_channels = 8;
  std::vector<int> expected_channel_map = {0, 1, 2, 3, 4, 5};
  std::string notes;
  std::optional<std::string> model_class;
  std::optional<double> dis_threshold;
  std::vector<double> supported_radii;
  std::optional<std::string> source_config;
  std::optional<std::string> source_run_dir;
  std::optional<int> trained_epochs;
  std::optional<bool> distance_head_trained;
  std::optional<double> distance_valid_threshold;
  std::optional<bool> count_head_trained;
  std::optional<double> count_min_confidence;
};

RuntimeContract load_contract(const std::string& path);

struct ZooPick {
  std::string onnx_path;
  std::string contract_path;
  std::vector<double> available_radii;
};

ZooPick resolve_zoo_model(const std::string& zoo_dir, double bubble_radius);

}  // namespace soundbubble
