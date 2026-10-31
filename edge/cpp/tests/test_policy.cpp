#include "soundbubble/policy.hpp"

#include <gtest/gtest.h>

TEST(Policy, StaleCaptureSheds) {
  soundbubble::BudgetInput in;
  in.queue_policy = "adaptive";
  in.mode = "normal";
  in.capture_age_ms = 40;
  in.max_capture_age_ms = 24;
  in.capture_discontinuity = true;
  in.infer_ema_ms = 8.5;
  in.block_budget_ms = 16;
  auto d = soundbubble::decide_budget(in);
  EXPECT_EQ(d.mode, "shed");
  EXPECT_LE(d.step_budget, in.hops_per_io);
}

TEST(Policy, RecoverOnlyWhenSustainable) {
  soundbubble::BudgetInput in;
  in.queue_policy = "adaptive";
  in.mode = "normal";
  in.capture_age_ms = 2;
  in.max_capture_age_ms = 24;
  in.out_ahead_ms = 5;
  in.target_ahead_ms = 40;
  in.infer_ema_ms = 8.5;
  in.block_budget_ms = 16;
  in.recover_steps_per_block = 3;
  in.hops_per_io = 2;
  auto d = soundbubble::decide_budget(in);
  EXPECT_NE(d.mode, "recover");
  EXPECT_LE(d.step_budget, 1);
}

TEST(Policy, RecoverWhenBudgetAllows) {
  soundbubble::BudgetInput in;
  in.queue_policy = "adaptive";
  in.capture_age_ms = 1;
  in.max_capture_age_ms = 24;
  in.out_ahead_ms = 1;
  in.target_ahead_ms = 40;
  in.infer_ema_ms = 2;
  in.block_budget_ms = 16;
  in.dc_ema_ms = 0;
  in.level_ema_ms = 0;
  in.frame_to_infer_ema_ms = 0;
  in.post_ema_ms = 0;
  in.recover_steps_per_block = 2;
  in.hops_per_io = 2;
  auto d = soundbubble::decide_budget(in);
  EXPECT_EQ(d.mode, "recover");
  EXPECT_EQ(d.step_budget, 2);
  EXPECT_GT(d.effective_target_ahead_ms, in.target_ahead_ms);
}

TEST(Policy, FreshRingForcesOneHop) {
  soundbubble::BudgetInput in;
  in.queue_policy = "adaptive";
  in.mode = "normal";
  in.capture_age_ms = 4;
  in.max_capture_age_ms = 24;
  in.capture_discontinuity = false;
  in.model_has_chunk = true;
  in.infer_ema_ms = 80;
  in.block_budget_ms = 16;
  in.hops_per_io = 2;
  in.max_steps_per_block = 2;
  auto d = soundbubble::decide_budget(in);
  EXPECT_GE(d.sustainable_steps, 1);
  EXPECT_GE(d.step_budget, 1);
}

TEST(Policy, StaleRingDoesNotForceHop) {
  soundbubble::BudgetInput in;
  in.capture_discontinuity = true;
  in.model_has_chunk = true;
  in.infer_ema_ms = 80;
  in.block_budget_ms = 16;
  in.hops_per_io = 2;
  auto d = soundbubble::decide_budget(in);
  EXPECT_EQ(d.sustainable_steps, 0);
  EXPECT_EQ(d.step_budget, 0);
}

TEST(Policy, UnderrunRaisesCushionThenDecays) {
  soundbubble::BudgetInput in;
  in.target_ahead_ms = 40;
  in.max_ahead_ms = 80;
  in.output_block_ms = 16;
  in.out_ahead_ms = 40;
  in.infer_ema_ms = 2;
  in.block_budget_ms = 16;
  in.dc_ema_ms = 0;
  in.level_ema_ms = 0;
  in.frame_to_infer_ema_ms = 0;
  in.post_ema_ms = 0;
  in.recent_underrun_frames = 768;
  auto raised = soundbubble::decide_budget(in);
  EXPECT_GT(raised.adaptive_extra_ms, 0.0);
  EXPECT_GT(raised.effective_target_ahead_ms, in.target_ahead_ms);
  EXPECT_LE(raised.effective_target_ahead_ms, in.target_ahead_ms * 2.0);
  in.recent_underrun_frames = 0;
  in.adaptive_extra_ms = raised.adaptive_extra_ms;
  auto decayed = soundbubble::decide_budget(in);
  EXPECT_LT(decayed.adaptive_extra_ms, raised.adaptive_extra_ms);
}

TEST(Overload, ParsesDryFill) {
  EXPECT_EQ(soundbubble::parse_overload_policy("dry_fill"), soundbubble::OverloadPolicy::DryFill);
  EXPECT_EQ(soundbubble::parse_overload_policy("silence_fill"), soundbubble::OverloadPolicy::SilenceFill);
  EXPECT_EQ(soundbubble::parse_overload_policy("strict"), soundbubble::OverloadPolicy::Strict);
}
