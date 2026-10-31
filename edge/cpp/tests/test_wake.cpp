#include "soundbubble/wake.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <unordered_set>
#include <fstream>

TEST(DryHistory, WrapAndFractional) {
  soundbubble::DryHistory hist(8);
  std::vector<float> seq(16);
  for (int i = 0; i < 16; ++i) seq[static_cast<size_t>(i)] = static_cast<float>(i);
  hist.write(seq.data(), 16);
  EXPECT_EQ(hist.write_index(), 16);
  EXPECT_EQ(hist.oldest_index(), 8);
  auto [block, next] = hist.read_fractional(8.0, 1.0, 4);
  EXPECT_DOUBLE_EQ(next, 12.0);
  EXPECT_NEAR(block[0], 8.f, 1e-5);
  EXPECT_NEAR(block[3], 11.f, 1e-5);
  soundbubble::DryHistory hist2(16);
  float ramp[] = {0, 10, 20, 30};
  hist2.write(ramp, 4);
  auto [interp, _] = hist2.read_fractional(0.5, 1.0, 3);
  EXPECT_NEAR(interp[0], 5.f, 1e-4);
  EXPECT_NEAR(interp[1], 15.f, 1e-4);
  EXPECT_NEAR(interp[2], 25.f, 1e-4);
}

TEST(BubbleOpen, ReplayThenIdle) {
  const int sr = 100;
  soundbubble::BubbleOpenController ctrl(sr, 0.2f, 1.5f, 1.25f, 8.f, 100.f, 100.f);
  soundbubble::DryHistory hist(4 * sr);
  std::vector<float> zeros(3 * sr, 0.f);
  hist.write(zeros.data(), static_cast<int>(zeros.size()));
  ctrl.trigger("darren");
  bool entered = false;
  for (int elapsed = 0; elapsed < 800; elapsed += 20) {
    std::vector<float> live(20, 0.5f);
    hist.write(live.data(), 20);
    auto [dry, mix] = ctrl.process_block(live.data(), 20, hist);
    if (elapsed == 0) EXPECT_GT(mix.back(), mix.front());
    if (ctrl.status().state == soundbubble::BubbleOpenController::STATE_OPEN_LIVE) {
      entered = true;
      break;
    }
  }
  EXPECT_TRUE(entered);
  for (int total = 0; total < 900; total += 20) {
    std::vector<float> live(20, 0.f);
    hist.write(live.data(), 20);
    ctrl.process_block(live.data(), 20, hist);
  }
  for (int i = 0; i < 8; ++i) {
    std::vector<float> live(20, 0.f);
    hist.write(live.data(), 20);
    ctrl.process_block(live.data(), 20, hist);
  }
  auto st = ctrl.status();
  EXPECT_EQ(st.state, soundbubble::BubbleOpenController::STATE_IDLE);
  EXPECT_NEAR(st.mix_current, 0.2f, 1e-3);
}

TEST(BubbleOpen, RetriggerExtendsHold) {
  const int sr = 100;
  soundbubble::BubbleOpenController ctrl(sr, 0.f, 0.1f, 2.f, 0.5f, 20.f, 50.f);
  soundbubble::DryHistory hist(2 * sr);
  std::vector<float> zeros(sr, 0.f);
  hist.write(zeros.data(), sr);
  ctrl.trigger("darren");
  for (int i = 0; i < 2; ++i) {
    std::vector<float> live(20, 0.f);
    hist.write(live.data(), 20);
    ctrl.process_block(live.data(), 20, hist);
  }
  int before = ctrl.status().hold_frames_left;
  ctrl.trigger("darren");
  std::vector<float> live(20, 0.f);
  hist.write(live.data(), 20);
  ctrl.process_block(live.data(), 20, hist);
  EXPECT_GT(ctrl.status().hold_frames_left, before);
}

TEST(PhraseBook, CliOverridesJson) {
  auto dir = std::filesystem::temp_directory_path() / "sb-phrases";
  std::filesystem::create_directories(dir);
  auto path = dir / "phrases.json";
  std::ofstream(path) << R"({"phrases":{"hey darren":"one"}})";
  auto phrases = soundbubble::build_phrases(path.string(), {"hey darren=two", "hey alex=alex"});
  EXPECT_EQ(phrases["hey darren"], "two");
  EXPECT_EQ(phrases["hey alex"], "alex");
  std::unordered_set<std::string> vocab = {"HEY", "DARREN"};
  auto lines = soundbubble::build_keyword_lines(
      {{"hey darren", "darren"}}, [](const std::string&) { return std::vector<std::string>{"HEY", "DARREN"}; }, vocab);
  ASSERT_EQ(lines.size(), 1u);
  EXPECT_EQ(lines[0].line, "HEY DARREN @PERSON_0");
  EXPECT_THROW(soundbubble::build_keyword_lines({{"nope", "x"}}, [](const std::string&) { return std::vector<std::string>{"<unk>"}; }, vocab),
               std::invalid_argument);
}
