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

// --- declick behaviour (2026-11-04) ---------------------------------------
#include <cmath>
#include <vector>

TEST(OutputRingDeclick, UnderrunDecaysToZeroNotDcHold) {
  const int K = soundbubble::kRingDeclickFrames;
  soundbubble::OutputRingBuffer ring(1, 4096);
  std::vector<float> one(64, 0.5f);
  ring.write_drop_oldest(one.data(), 64);
  std::vector<float> out(64 + 3 * K, 9.f);
  EXPECT_EQ(ring.read_into(out.data(), static_cast<int>(out.size())), 64);
  EXPECT_FLOAT_EQ(out[63], 0.5f);
  // Strictly decaying, reaching exactly zero within K frames, then silent.
  for (int i = 64; i < 64 + K; ++i) EXPECT_LT(out[i], out[i - 1]);
  for (int i = 64 + K; i < static_cast<int>(out.size()); ++i) EXPECT_FLOAT_EQ(out[i], 0.f);
  EXPECT_EQ(ring.underrun_frames(), static_cast<std::uint64_t>(3 * K));
}

TEST(OutputRingDeclick, ResumeAfterUnderrunFadesIn) {
  const int K = soundbubble::kRingDeclickFrames;
  soundbubble::OutputRingBuffer ring(1, 4096);
  std::vector<float> gap(4 * K);
  ring.read_into(gap.data(), static_cast<int>(gap.size()));  // underrun from empty
  std::vector<float> loud(4 * K, 1.f);
  ring.write_drop_oldest(loud.data(), static_cast<int>(loud.size()));
  std::vector<float> out(4 * K);
  ring.read_into(out.data(), static_cast<int>(out.size()));
  EXPECT_LT(out[0], 0.1f);  // no step from silence to full scale
  for (int i = 1; i < K; ++i) EXPECT_GT(out[i], out[i - 1]);
  EXPECT_FLOAT_EQ(out[2 * K], 1.f);
}

TEST(OutputRingDeclick, TrimSpliceIsContinuous) {
  // Ramp signal: a hard drop of 500 frames would jump by 500 at the splice.
  soundbubble::OutputRingBuffer ring(1, 8192);
  std::vector<float> ramp(2000);
  for (int i = 0; i < 2000; ++i) ramp[i] = static_cast<float>(i);
  ring.write_drop_oldest(ramp.data(), 2000);
  std::vector<float> head(100);
  ring.read_into(head.data(), 100);  // last delivered = 99
  ring.drop_oldest_frames(500);
  std::vector<float> out(200);
  ring.read_into(out.data(), 200);
  float max_step = std::fabs(out[0] - head[99]);
  for (int i = 1; i < 200; ++i) max_step = std::max(max_step, std::fabs(out[i] - out[i - 1]));
  EXPECT_LT(max_step, 20.f);  // smooth vs a 500-unit jump
  EXPECT_FLOAT_EQ(out[150], 750.f);  // past the crossfade: exact new position
}

TEST(OutputRingDeclick, FadeTailRampsNewestToZero) {
  soundbubble::OutputRingBuffer ring(1, 1024);
  std::vector<float> one(200, 1.f);
  ring.write_drop_oldest(one.data(), 200);
  ring.fade_tail(soundbubble::kRingDeclickFrames);
  std::vector<float> out(200);
  ring.read_into(out.data(), 200);
  EXPECT_FLOAT_EQ(out[0], 1.f);
  EXPECT_FLOAT_EQ(out[199], 0.f);
  EXPECT_LT(out[199 - soundbubble::kRingDeclickFrames / 2], 0.6f);
}
