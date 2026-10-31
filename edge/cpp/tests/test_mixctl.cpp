#include "soundbubble/mixctl.hpp"

#include <gtest/gtest.h>

TEST(MixLine, ParsesAndClamps) {
  auto ok = soundbubble::parse_mix_line("MIX wet=0.35 dry=0.65\n");
  EXPECT_TRUE(ok.ok);
  EXPECT_NEAR(ok.wet, 0.35f, 1e-5);
  EXPECT_NEAR(ok.dry, 0.65f, 1e-5);
  auto bad = soundbubble::parse_mix_line("not a mix line");
  EXPECT_FALSE(bad.ok);
  auto partial = soundbubble::parse_mix_line("MIX wet=0.2");
  EXPECT_FALSE(partial.ok);
  auto hi = soundbubble::parse_mix_line("MIX wet=2 dry=-1");
  EXPECT_TRUE(hi.ok);
  EXPECT_FLOAT_EQ(hi.wet, 1.f);
  EXPECT_FLOAT_EQ(hi.dry, 0.f);
}

TEST(MixControl, ReassemblesPartialLines) {
  soundbubble::MixControl ctl("", 115200, 0.f);
  ctl.ingest_for_test("MIX wet=0.20 dry=");
  EXPECT_FLOAT_EQ(ctl.dry(), 0.f);
  ctl.ingest_for_test("0.80\nMIX garbage\nMIX wet=1.00 dry=0.00\n");
  EXPECT_FLOAT_EQ(ctl.dry(), 0.f);
}
