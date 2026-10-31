#include "soundbubble/realtime.hpp"

#include <gtest/gtest.h>

TEST(DropOldest, DropsOldestAndPopLatestDrains) {
  soundbubble::DropOldestBuffer<int> buf(2);
  buf.push(1);
  buf.push(2);
  buf.push(3);
  EXPECT_EQ(buf.stats.dropped_oldest, 1u);
  auto [item, drained] = buf.pop_latest(0.01);
  ASSERT_TRUE(item);
  EXPECT_EQ(*item, 3);
  EXPECT_EQ(drained, 1);
  EXPECT_FALSE(buf.pop(0.0));
}

TEST(AudioRing, WrapAndDrop) {
  soundbubble::AudioRingBuffer ring(1, 4);
  float a[] = {1, 2, 3};
  ring.push(a, 1, 3);
  float b[] = {4, 5};
  ring.push(b, 1, 2);
  EXPECT_EQ(ring.stats.dropped_samples, 1u);
  std::vector<float> out;
  ASSERT_TRUE(ring.pop(2, out));
  EXPECT_FLOAT_EQ(out[0], 2.f);
  EXPECT_FLOAT_EQ(out[1], 3.f);
}

TEST(OutputRing, TrimLockstep) {
  soundbubble::OutputRingBuffer ring(2, 8);
  float block[4 * 2] = {1, 1, 2, 2, 3, 3, 4, 4};
  EXPECT_EQ(ring.write_drop_oldest(block, 4), 0);
  EXPECT_EQ(ring.ahead_frames(), 4);
  EXPECT_EQ(ring.drop_oldest_frames(2), 2);
  float out[2 * 2] = {};
  EXPECT_EQ(ring.read_into(out, 2), 2);
  EXPECT_FLOAT_EQ(out[0], 3.f);
  EXPECT_FLOAT_EQ(out[2], 4.f);
}

TEST(Contract, RejectsBadAhead) {
  soundbubble::RealtimeContract c;
  c.target_output_ahead_sec = 1;
  c.max_output_ahead_sec = 1;
  EXPECT_THROW(c.validate(), std::invalid_argument);
}

TEST(RingParams, TargetBelowMax) {
  auto p = soundbubble::compute_output_ring_params(48000, 384, 0.08, 0.04);
  EXPECT_LT(p.ring_target_ahead_frames, p.ring_max_ahead_frames);
  EXPECT_GE(p.ring_n_frames, p.ring_max_ahead_frames * 2);
}
