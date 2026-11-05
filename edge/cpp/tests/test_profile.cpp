#include "soundbubble/controller.hpp"
#include "soundbubble/policy.hpp"

#include <gtest/gtest.h>

#include <sstream>
#include <string>

TEST(ProfileLine, MatchesControllerParser) {
  soundbubble::ProfileFields f;
  f.mode = "normal";
  f.capture_age_ms = 3.5;
  f.capture_gap = 0;
  f.capture_drained = 0;
  f.capture_to_frame_ms = 4.2;
  f.dc_ms = 0.11;
  f.level_ms = 0.22;
  f.frame_to_infer_ms = 0.33;
  f.infer_mean_ms = 8.5;
  f.infer_total_ms = 8.5;
  f.steps = 1;
  f.post_ms = 0.4;
  f.e2e_ms = 12.0;
  f.drift_ppm = 1.5;
  f.q = 0;
  f.qcap = 1;
  f.drops_oldest = 2;
  f.out_ahead_ms = 20.0;
  f.dropped = 3;
  f.underrun_frames = 4;
  f.model_ring_ms = 8.0;
  f.model_ring_drops = 5;
  f.model_ring_resets = 6;
  f.model_out_db = -18.0;
  f.output_db = -12.5;
  f.sustainable_steps = 2;
  f.drained_total = 7;
  f.primed = 1;
  f.short_writes = 0;
  auto line = soundbubble::format_profile_line(f);
  EXPECT_NE(line.find("model_out=-18.0dBFS"), std::string::npos) << line;
  EXPECT_NE(line.find("out=-12.5dBFS"), std::string::npos) << line;
  EXPECT_NE(line.find("sustainable=2"), std::string::npos);
  EXPECT_NE(line.find("drained_total=7"), std::string::npos);
  EXPECT_NE(line.find("primed=1"), std::string::npos);
  auto parsed = soundbubble::parse_profile_line(line);
  ASSERT_TRUE(parsed.ok) << line;
  EXPECT_NEAR(parsed.infer_ms, 8.5, 0.02);
  EXPECT_NEAR(parsed.capture_to_frame_ms, 4.2, 0.02);
  EXPECT_EQ(parsed.capture_queue_fill, 0);
  EXPECT_EQ(parsed.capture_queue_capacity, 1);
  EXPECT_EQ(parsed.model_ring_resets, 6u);
}

TEST(MonitorBlock, MatchesControllerParser) {
  soundbubble::MonitorBlockFields m;
  m.levels_dbfs = {-12.3f, -40.1f, -5.0f, -20.0f, -18.2f, -19.0f};
  m.statuses = {"ok", "silence", "FAULT", "ok", "ok", "ok"};
  m.speaker_count = "1";
  m.count_status = "valid";
  m.distance_m = 1.25f;
  m.distance_status = "ok";
  m.model_out_db = -18.0;
  m.output_db = -12.0;
  m.speaker_parts = {"S1: 1.25m (valid)"};
  auto block = soundbubble::format_monitor_block(m);
  soundbubble::MonitorParse acc;
  std::istringstream in(block);
  std::string line;
  while (std::getline(in, line)) {
    soundbubble::parse_monitor_line(line, acc);
  }
  ASSERT_TRUE(acc.any);
  ASSERT_EQ(acc.levels_dbfs.size(), 6u);
  EXPECT_NEAR(acc.levels_dbfs[0], -12.3, 0.05);
  EXPECT_EQ(acc.statuses[1], "silence");
  ASSERT_TRUE(acc.speaker_count.has_value());
  EXPECT_EQ(*acc.speaker_count, "1");
  ASSERT_TRUE(acc.distance_m.has_value());
  EXPECT_NEAR(*acc.distance_m, 1.25, 0.01);
  EXPECT_EQ(acc.distance_status, "ok");
}
