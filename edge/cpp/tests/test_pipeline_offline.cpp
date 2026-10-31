#include "soundbubble/pipeline.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>

namespace {

class PassthroughInferencer final : public soundbubble::Inferencer {
 public:
  void reset() override {}
  Hop run(const float* frame, int channels, int frame_len) override {
    Hop hop;
    hop.infer_ms = 0.1;
    int n = std::min(192, frame_len);
    hop.samples.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) hop.samples[static_cast<size_t>(i)] = frame[(frame_len - n) + i];
    (void)channels;
    return hop;
  }
};

}  // namespace

TEST(OfflinePipeline, ProcessesOneBlockWithoutOnnx) {
  soundbubble::PipelineConfig cfg;
  cfg.input_channels = 8;
  cfg.output_channels = 2;
  cfg.model_num_ch = 6;
  cfg.input_sr = 24000;
  cfg.model_sr = 24000;
  cfg.output_sr = 24000;
  cfg.model_chunk = 192;
  cfg.model_pad = 96;
  cfg.hops_per_io = 2;
  cfg.max_steps_per_block = 2;
  cfg.recover_steps_per_block = 2;
  cfg.max_capture_blocks = 2;
  cfg.max_output_ahead_sec = 0.10;
  cfg.target_output_ahead_sec = 0.04;
  cfg.silence_dbfs = -120;
  cfg.fault_min_dbfs = 0;
  cfg.fault_max_dbfs = 0;
  cfg.overload_policy = "strict";
  cfg.queue_policy = "fifo";
  cfg.profile = true;
  cfg.profile_interval = 0.0;
  soundbubble::Engine engine(cfg);
  engine.set_inferencer_for_test(std::make_unique<PassthroughInferencer>());
  const int frames = 384;
  soundbubble::CaptureItem item;
  item.channels = 8;
  item.interleaved.assign(static_cast<size_t>(frames * 8), 0.1f);
  item.t_capture = 0;
  item.capture_idx = 0;
  engine.push_capture_for_test(std::move(item));
  ASSERT_TRUE(engine.process_one_for_test(0.05));
  EXPECT_GT(engine.output_ahead_frames(), 0);
}
