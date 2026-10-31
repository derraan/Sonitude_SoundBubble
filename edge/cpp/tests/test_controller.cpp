#include "soundbubble/controller.hpp"
#include "soundbubble/json_mini.hpp"
#include "soundbubble/policy.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>

TEST(ControllerConfig, RejectsUnknownAndBuildsCppArgv) {
  EXPECT_THROW(soundbubble::controller_config_from_json("{\"nope\":1}"), std::invalid_argument);
  auto repo = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
  auto model = repo / "edge" / "model.onnx";
  auto contract = repo / "edge" / "model.runtime.json";
  if (!std::filesystem::exists(model) || !std::filesystem::exists(contract)) GTEST_SKIP() << "model artifacts missing";
  std::string json = std::string("{\"model\":\"") + (repo / "edge" / "model.onnx").generic_string() +
                     "\",\"contract_path\":\"" + (repo / "edge" / "model.runtime.json").generic_string() +
                     "\",\"runtime\":\"cpp\",\"profile\":true}";
  auto cfg = soundbubble::controller_config_from_json(json);
  auto argv = soundbubble::build_pipeline_argv(cfg, repo.string(), "soundbubble-rt");
  ASSERT_FALSE(argv.empty());
  EXPECT_EQ(argv[0], "soundbubble-rt");
  EXPECT_NE(std::find(argv.begin(), argv.end(), "--profile"), argv.end());
  cfg.runtime = "python";
  auto py = soundbubble::build_pipeline_argv(cfg, repo.string(), "soundbubble-rt");
  EXPECT_NE(py[0].find("inference_pipeline.py"), std::string::npos);
}

TEST(ControllerTelemetry, MonitorParseProfileTrialsAndTaps) {
  auto dir = std::filesystem::temp_directory_path() / "sb_ctl_trials_empty";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
#if defined(_WIN32)
  _putenv_s("SOUNDBUBBLE_MEASURE_DIR", dir.string().c_str());
#else
  setenv("SOUNDBUBBLE_MEASURE_DIR", dir.string().c_str(), 1);
#endif
  soundbubble::MonitorParse direct;
  ASSERT_TRUE(soundbubble::parse_monitor_line("ch0:  -12.3 dBFS valid    ch1:  -40.1 dBFS silent", direct));
  ASSERT_EQ(direct.levels_dbfs.size(), 2u);

  soundbubble::ProcessManager mgr(std::filesystem::temp_directory_path().string(), "soundbubble-rt", "python");

  auto tap = mgr.signal_taps("disarm");
  EXPECT_EQ(tap.status, 409);
  EXPECT_NE(tap.body.find("not owned"), std::string::npos);

  auto missing = soundbubble::parse_json(mgr.trials_json("missing-run", ""));
  EXPECT_EQ(missing.find("run_id")->s, "missing-run");
  ASSERT_NE(missing.find("records"), nullptr);
  EXPECT_TRUE(missing.find("records")->a.empty());

  soundbubble::ProfileFields fields;
  fields.mode = "normal";
  fields.capture_to_frame_ms = 4.2;
  fields.dc_ms = 0.11;
  fields.level_ms = 0.22;
  fields.frame_to_infer_ms = 0.33;
  fields.infer_mean_ms = 8.5;
  fields.infer_total_ms = 8.5;
  fields.steps = 1;
  fields.post_ms = 0.4;
  fields.e2e_ms = 12.0;
  fields.drift_ppm = 1.5;
  fields.q = 0;
  fields.qcap = 1;
  fields.drops_oldest = 2;
  fields.out_ahead_ms = 20.0;
  fields.dropped = 3;
  fields.underrun_frames = 4;
  fields.model_ring_ms = 8.0;
  fields.model_ring_drops = 5;
  fields.model_ring_resets = 6;
  mgr.ingest_line(soundbubble::format_profile_line(fields));
  mgr.ingest_line("ch0:  -12.3 dBFS valid    ch1:  -40.1 dBFS silent   ch2:   -5.0 dBFS fault");
  mgr.ingest_line("ch3:  -20.0 dBFS valid    ch4:  -18.2 dBFS valid    ch5:  -19.0 dBFS valid");
  mgr.ingest_line("speaker_count=1 (valid)");
  mgr.ingest_line("distance=1.25m (ok)");

  auto raw = mgr.telemetry_json();
  auto tel = soundbubble::parse_json(raw);
  EXPECT_EQ(tel.find("source")->s, "idle");
  EXPECT_EQ(tel.find("host")->s, "pi_rig");
  EXPECT_EQ(tel.find("runtime")->s, "cpp");
  EXPECT_GE(tel.find("seq")->n, 1);
  const soundbubble::Json* profile = tel.find("last_profile");
  ASSERT_NE(profile, nullptr);
  ASSERT_EQ(profile->type, soundbubble::Json::Type::Object);
  EXPECT_NEAR(profile->find("e2e_ms")->n, 12.0, 0.05);
  EXPECT_NEAR(profile->find("infer_ms")->n, 8.5, 0.05);
  EXPECT_NEAR(profile->find("out_ahead_ms")->n, 20.0, 0.2);
  EXPECT_EQ(profile->find("underrun_frames")->n, 4);
  EXPECT_EQ(profile->find("drops_oldest")->n, 2);
  EXPECT_EQ(profile->find("out_dropped_frames")->n, 3);
  EXPECT_EQ(profile->find("model_ring_drops")->n, 5);
  EXPECT_EQ(profile->find("model_ring_resets")->n, 6);
  const soundbubble::Json* mon = tel.find("last_monitor");
  ASSERT_EQ(mon->type, soundbubble::Json::Type::Object);
  ASSERT_EQ(mon->find("levels_dbfs")->a.size(), 6u) << raw;
  EXPECT_NEAR(mon->find("levels_dbfs")->a[0].n, -12.3, 0.01);
  EXPECT_EQ(mon->find("statuses")->a[1].s, "silent");
  EXPECT_EQ(mon->find("speaker_count")->s, "1");
  EXPECT_NEAR(mon->find("distance_m")->n, 1.25, 0.001);
  EXPECT_EQ(mon->find("distance_status")->s, "ok");
  EXPECT_EQ(mon->find("bubble_radius_m")->type, soundbubble::Json::Type::Null);
  EXPECT_TRUE(tel.find("trials_tail")->a.empty());
  EXPECT_FALSE(tel.find("taps")->find("armed")->b);
  EXPECT_EQ(tel.find("config")->type, soundbubble::Json::Type::Null);
  auto idle_status = soundbubble::parse_json(mgr.status_json());
  EXPECT_EQ(idle_status.find("config")->type, soundbubble::Json::Type::Null);

  soundbubble::MonitorParse pending;
  EXPECT_TRUE(soundbubble::parse_monitor_line("distance=pending", pending));
  EXPECT_FALSE(pending.distance_m.has_value());
  EXPECT_EQ(pending.distance_status, "pending");
  soundbubble::MonitorParse bare_count;
  EXPECT_TRUE(soundbubble::parse_monitor_line("speaker_count=unavailable", bare_count));
  EXPECT_FALSE(bare_count.speaker_count.has_value());

  auto run = dir / "r1";
  std::filesystem::create_directories(run);
  {
    std::ofstream out(run / "telemetry.jsonl");
    out << "{\"seq\":1,\"host\":\"pi_rig\"}\n{\"seq\":2,\"host\":\"pi_rig\"}\n";
  }
  auto filtered = soundbubble::parse_json(mgr.trials_json("r1", "1"));
  ASSERT_EQ(filtered.find("records")->a.size(), 1u);
  EXPECT_EQ(filtered.find("records")->a[0].find("seq")->n, 2);
  auto with_tail = soundbubble::parse_json(mgr.telemetry_json());
  EXPECT_EQ(with_tail.find("trials_tail")->a.size(), 2u);
}

TEST(ControllerTelemetry, ConfigNullUnlessOwned) {
  soundbubble::ProcessManager mgr(std::filesystem::temp_directory_path().string(), "soundbubble-rt", "python");
  auto idle = soundbubble::parse_json(mgr.telemetry_json());
  EXPECT_EQ(idle.find("config")->type, soundbubble::Json::Type::Null);
  EXPECT_EQ(soundbubble::parse_json(mgr.status_json()).find("config")->type, soundbubble::Json::Type::Null);

  mgr.note_owned_config_for_test("{\"bubble_radius\":1.25,\"runtime\":\"cpp\",\"channel_map\":\"0,1,2,3,4,5\"}");
  auto owned = soundbubble::parse_json(mgr.telemetry_json());
  const soundbubble::Json* cfg = owned.find("config");
  ASSERT_NE(cfg, nullptr);
  ASSERT_EQ(cfg->type, soundbubble::Json::Type::Object);
  EXPECT_NEAR(cfg->find("bubble_radius")->n, 1.25, 0.001);
  EXPECT_EQ(cfg->find("runtime")->s, "cpp");
  EXPECT_EQ(cfg->find("channel_map")->s, "0,1,2,3,4,5");
  EXPECT_EQ(cfg->find("input_channels")->n, 8);
  EXPECT_EQ(cfg->find("model")->type, soundbubble::Json::Type::String);
  auto status = soundbubble::parse_json(mgr.status_json());
  ASSERT_EQ(status.find("config")->type, soundbubble::Json::Type::Object);
  EXPECT_NEAR(status.find("config")->find("bubble_radius")->n, 1.25, 0.001);
}

TEST(ControllerConfig, RejectsPathOutsideEdge) {
  auto repo = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
  soundbubble::ControllerConfig cfg;
  cfg.model = repo.string();
  cfg.contract_path.reset();
  EXPECT_THROW(soundbubble::validate_controller_config(cfg, repo.string()), std::invalid_argument);
}
