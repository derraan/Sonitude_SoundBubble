#include "soundbubble/limiter.hpp"
#include "soundbubble/processing.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

TEST(DcFilter, MatchesDifferenceEquation) {
  soundbubble::DCOffsetFilter dc(1, 0.5f);
  float x[] = {1.f, 1.f, 1.f};
  auto y = dc.process(x, 3);
  EXPECT_NEAR(y[0], 1.f, 1e-5);
  EXPECT_NEAR(y[1], 0.5f, 1e-5);
  EXPECT_NEAR(y[2], 0.25f, 1e-5);
}

TEST(ChannelMap, ParsesDefault) {
  auto map = soundbubble::parse_channel_map("0,1,2,3,4,5", 8, 6);
  ASSERT_EQ(map.size(), 6u);
  EXPECT_EQ(map[5], 5);
}

TEST(Resampler, IdentityRateCopies) {
  soundbubble::StreamingResampler src(24000, 24000, 2);
  float x[] = {1, 2, 3, 4};
  auto y = src.process(x, 2);
  ASSERT_EQ(y.size(), 4u);
  EXPECT_FLOAT_EQ(y[3], 4.f);
}

TEST(PeakLimiter, InstantAttackNoLookahead) {
  soundbubble::PeakLimiter lim;
  lim.configure({.ceiling = 0.5f, .release_ms = 20.f}, 24000);
  std::vector<float> x = {0.2f, 1.0f, 0.2f};
  lim.process_mono(x);
  EXPECT_NEAR(x[0], 0.2f, 1e-5);
  EXPECT_NEAR(x[1], 0.5f, 1e-5);
  EXPECT_LE(std::fabs(x[2]), 0.5f + 1e-5);
  EXPECT_LT(lim.current_gain(), 1.f);
}

TEST(PeakLimiter, LinkedStereo) {
  soundbubble::PeakLimiter lim;
  lim.configure({.ceiling = 0.8f, .release_ms = 10.f}, 48000);
  std::vector<float> lr = {1.0f, 0.1f};
  lim.process_interleaved_stereo(lr);
  EXPECT_NEAR(lr[0], 0.8f, 1e-5);
  EXPECT_NEAR(lr[1], 0.08f, 1e-5);
}
