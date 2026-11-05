#include "soundbubble/contract.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

TEST(Contract, LoadsRepoContractIfPresent) {
  namespace fs = std::filesystem;
  fs::path path = fs::path(__FILE__).parent_path().parent_path().parent_path() / "model.runtime.json";
  if (!fs::exists(path)) GTEST_SKIP() << "model.runtime.json not present";
  auto c = soundbubble::load_contract(path.string());
  EXPECT_EQ(c.model_sr, 24000);
  EXPECT_EQ(c.model_num_ch, 6);
  EXPECT_EQ(c.model_chunk, 192);
  ASSERT_FALSE(c.expected_channel_map.empty());
}

TEST(Zoo, PrefersSingleDistance) {
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "sb-zoo-test";
  fs::remove_all(dir);
  fs::create_directories(dir);
  auto write = [&](const std::string& stem, const std::string& json) {
    std::ofstream(dir / (stem + ".runtime.json")) << json;
    std::ofstream(dir / (stem + ".onnx")) << "not-a-real-model";
  };
  write("single", R"({"model_sr":24000,"model_num_ch":6,"model_chunk":192,"model_pad":96,"frame_len":288,"expected_input_channels":8,"expected_channel_map":[0,1,2,3,4,5],"dis_threshold":1.5})");
  write("multi", R"({"model_sr":24000,"model_num_ch":6,"model_chunk":192,"model_pad":96,"frame_len":288,"expected_input_channels":8,"expected_channel_map":[0,1,2,3,4,5],"supported_radii":[1.0,1.5,2.0]})");
  auto pick = soundbubble::resolve_zoo_model(dir.string(), 1.5);
  EXPECT_NE(pick.onnx_path.find("single.onnx"), std::string::npos);
  EXPECT_EQ(pick.available_radii.size(), 3u);
  fs::remove_all(dir);
}
