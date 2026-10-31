#include "soundbubble/controller.hpp"

#include "soundbubble/audio_io.hpp"
#include "soundbubble/json_mini.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace soundbubble {
namespace {

std::string iso_now() {
  auto now = std::chrono::system_clock::now();
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  char buf[64];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

bool under_root(const std::filesystem::path& path, const std::filesystem::path& root) {
  auto p = std::filesystem::weakly_canonical(path);
  auto r = std::filesystem::weakly_canonical(root);
  auto rel = std::filesystem::relative(p, r);
  std::string rel_s = rel.generic_string();
  return !rel.empty() && rel_s.find("..") == std::string::npos;
}

const Json* require_obj_field(const Json& j, const char* key) { return j.find(key); }

void set_if(ControllerConfig& cfg, const Json& j) {
  auto num = [&](const char* k, double& dst) {
    if (const Json* v = j.find(k)) {
      if (v->type != Json::Type::Number) throw std::invalid_argument(std::string(k) + " must be a number");
      dst = v->n;
    }
  };
  auto num_i = [&](const char* k, int& dst) {
    if (const Json* v = j.find(k)) dst = static_cast<int>(std::llround(v->n));
  };
  auto str = [&](const char* k, std::optional<std::string>& dst) {
    if (const Json* v = j.find(k)) {
      if (v->type == Json::Type::Null) dst.reset();
      else if (v->type == Json::Type::String) dst = v->s;
      else throw std::invalid_argument(std::string(k) + " must be a string");
    }
  };
  auto str_req = [&](const char* k, std::string& dst) {
    if (const Json* v = j.find(k)) {
      if (v->type != Json::Type::String) throw std::invalid_argument(std::string(k) + " must be a string");
      dst = v->s;
    }
  };
  auto boolean = [&](const char* k, bool& dst) {
    if (const Json* v = j.find(k)) dst = v->type == Json::Type::Bool ? v->b : v->n != 0;
  };
  static const std::vector<std::string> known = {
      "model", "contract_path", "zoo_dir", "bubble_radius", "input_device", "output_device", "input_channels", "output_channels",
      "channel_map", "monitor_input_channel", "input_sr", "model_sr", "output_sr", "model_chunk", "model_pad", "model_num_ch",
      "hops_per_io", "output_device_block", "duplex_mode", "duplex_block", "latency", "max_capture_blocks", "max_output_ahead_sec", "target_output_ahead_sec", "rt_fifo_priority",
      "output_gain_db", "output_limiter", "limiter_ceiling", "limiter_release_ms",
      "mix_dry", "input_gain_db", "alpha", "silence_dbfs", "silence_window_frames", "fault_min_dbfs", "fault_max_dbfs",
      "max_overrun_ms", "intra_op_threads", "inter_op_threads", "model_ring_ms", "max_steps_per_block", "require_soxr",
      "wake_model_dir", "wake_phrases_json", "wake_phrase", "wake_threshold", "wake_hold_sec", "wake_replay_sec", "wake_replay_rate",
      "wake_release_ms", "profile", "profile_interval", "monitor", "monitor_interval",
      "mix_uart_port", "mix_uart_baud", "runtime"};
  for (const auto& kv : j.o) {
    if (std::find(known.begin(), known.end(), kv.first) == known.end()) throw std::invalid_argument("Unknown config key(s): " + kv.first);
  }
  str("model", cfg.model);
  str("contract_path", cfg.contract_path);
  str("zoo_dir", cfg.zoo_dir);
  if (const Json* v = j.find("bubble_radius")) cfg.bubble_radius = v->type == Json::Type::Null ? std::nullopt : std::optional<double>(v->n);
  if (const Json* v = j.find("input_device")) cfg.input_device = v->type == Json::Type::Null ? std::nullopt : std::optional<int>(static_cast<int>(v->n));
  if (const Json* v = j.find("output_device")) cfg.output_device = v->type == Json::Type::Null ? std::nullopt : std::optional<int>(static_cast<int>(v->n));
  num_i("input_channels", cfg.input_channels);
  num_i("output_channels", cfg.output_channels);
  str_req("channel_map", cfg.channel_map);
  num_i("monitor_input_channel", cfg.monitor_input_channel);
  num_i("input_sr", cfg.input_sr);
  num_i("model_sr", cfg.model_sr);
  num_i("output_sr", cfg.output_sr);
  num_i("model_chunk", cfg.model_chunk);
  num_i("model_pad", cfg.model_pad);
  num_i("model_num_ch", cfg.model_num_ch);
  num_i("hops_per_io", cfg.hops_per_io);
  num_i("output_device_block", cfg.output_device_block);
  str_req("duplex_mode", cfg.duplex_mode);
  num_i("duplex_block", cfg.duplex_block);
  str_req("latency", cfg.latency);
  num_i("max_capture_blocks", cfg.max_capture_blocks);
  num("max_output_ahead_sec", cfg.max_output_ahead_sec);
  num("target_output_ahead_sec", cfg.target_output_ahead_sec);
  num_i("rt_fifo_priority", cfg.rt_fifo_priority);
  num("output_gain_db", cfg.output_gain_db);
  boolean("output_limiter", cfg.output_limiter);
  num("limiter_ceiling", cfg.limiter_ceiling);
  num("limiter_release_ms", cfg.limiter_release_ms);
  num("mix_dry", cfg.mix_dry);
  num("input_gain_db", cfg.input_gain_db);
  num("alpha", cfg.alpha);
  num("silence_dbfs", cfg.silence_dbfs);
  num_i("silence_window_frames", cfg.silence_window_frames);
  num("fault_min_dbfs", cfg.fault_min_dbfs);
  num("fault_max_dbfs", cfg.fault_max_dbfs);
  num("max_overrun_ms", cfg.max_overrun_ms);
  num_i("intra_op_threads", cfg.intra_op_threads);
  num_i("inter_op_threads", cfg.inter_op_threads);
  num("model_ring_ms", cfg.model_ring_ms);
  num_i("max_steps_per_block", cfg.max_steps_per_block);
  boolean("require_soxr", cfg.require_soxr);
  str("wake_model_dir", cfg.wake_model_dir);
  str("wake_phrases_json", cfg.wake_phrases_json);
  if (const Json* v = j.find("wake_phrase"); v && v->type == Json::Type::Array) {
    cfg.wake_phrase.clear();
    for (const auto& item : v->a) {
      if (item.type != Json::Type::String || item.s.find('=') == std::string::npos) {
        throw std::invalid_argument("wake_phrase entries must look like 'phrase=person'");
      }
      cfg.wake_phrase.push_back(item.s);
    }
  }
  num("wake_threshold", cfg.wake_threshold);
  num("wake_hold_sec", cfg.wake_hold_sec);
  num("wake_replay_sec", cfg.wake_replay_sec);
  num("wake_replay_rate", cfg.wake_replay_rate);
  num("wake_release_ms", cfg.wake_release_ms);
  boolean("profile", cfg.profile);
  num("profile_interval", cfg.profile_interval);
  boolean("monitor", cfg.monitor);
  num("monitor_interval", cfg.monitor_interval);
  {
    std::optional<std::string> port;
    str("mix_uart_port", port);
    cfg.mix_uart_port = port.value_or("");
  }
  num_i("mix_uart_baud", cfg.mix_uart_baud);
  str_req("runtime", cfg.runtime);
}

std::string resolve_under(const std::optional<std::string>& text, const std::filesystem::path& repo, bool must_exist, const char* kind,
                          bool allow_repo) {
  if (!text) return {};
  std::filesystem::path p = *text;
  if (p.is_relative()) p = repo / p;
  p = std::filesystem::weakly_canonical(p);
  std::filesystem::path edge = repo / "edge";
  bool ok = under_root(p, edge) || (allow_repo && under_root(p, repo));
  if (!ok) throw std::invalid_argument(std::string(kind) + " must be inside " + edge.string());
  if (must_exist && !std::filesystem::exists(p)) throw std::invalid_argument(std::string(kind) + " does not exist: " + p.string());
  return p.string();
}

double field_after(const std::string& line, const std::string& key, const std::string& unit) {
  auto pos = line.find(key);
  if (pos == std::string::npos) throw std::runtime_error("missing " + key);
  pos += key.size();
  auto end = unit.empty() ? line.find(' ', pos) : line.find(unit, pos);
  if (end == std::string::npos) end = line.size();
  return std::stod(line.substr(pos, end - pos));
}

}  // namespace

ControllerConfig controller_config_from_json(const std::string& json_text) {
  ControllerConfig cfg;
  if (json_text.empty() || json_text == "null") return cfg;
  Json j = parse_json(json_text);
  if (j.type == Json::Type::Null) return cfg;
  if (j.type != Json::Type::Object) throw std::invalid_argument("config must be an object");
  set_if(cfg, j);
  return cfg;
}

ControllerConfig validate_controller_config(ControllerConfig cfg, const std::string& repo_root) {
  if (!cfg.model && !cfg.zoo_dir) throw std::invalid_argument("Provide either model or zoo_dir + bubble_radius");
  if (cfg.zoo_dir && !cfg.bubble_radius) throw std::invalid_argument("zoo_dir requires bubble_radius");
  if (cfg.latency != "low" && cfg.latency != "high") throw std::invalid_argument("latency must be 'low' or 'high'");
  if (cfg.runtime != "cpp" && cfg.runtime != "python") throw std::invalid_argument("runtime must be cpp or python");
  if (cfg.duplex_mode.empty()) cfg.duplex_mode = "off";
  if (cfg.duplex_mode != "off" && cfg.duplex_mode != "worker" && cfg.duplex_mode != "callback") {
    throw std::invalid_argument("duplex_mode must be off, worker, or callback");
  }
  if (cfg.max_output_ahead_sec <= cfg.target_output_ahead_sec) {
    throw std::invalid_argument("max_output_ahead_sec must be greater than target_output_ahead_sec");
  }
  if (!(cfg.mix_dry >= 0 && cfg.mix_dry <= 1)) throw std::invalid_argument("mix_dry must be in [0.0, 1.0]");
  if (!(cfg.output_gain_db >= -60.0 && cfg.output_gain_db <= 24.0)) {
    throw std::invalid_argument("output_gain_db must be in [-60.0, 24.0] for safety");
  }
  // Negative limiter_ceiling means dBFS (e.g. -0.5); convert before range check.
  if (cfg.limiter_ceiling < 0.0) {
    cfg.limiter_ceiling = std::pow(10.0, cfg.limiter_ceiling / 20.0);
  }
  if (!(cfg.limiter_ceiling >= 0.1 && cfg.limiter_ceiling <= 1.0)) {
    throw std::invalid_argument("limiter_ceiling must be in [0.1, 1.0] linear (or negative dBFS)");
  }
  if (!(cfg.limiter_release_ms >= 1.0 && cfg.limiter_release_ms <= 500.0)) {
    throw std::invalid_argument("limiter_release_ms must be in [1.0, 500.0]");
  }
  if (!(cfg.wake_threshold > 0 && cfg.wake_threshold <= 1)) throw std::invalid_argument("wake_threshold must be in (0.0, 1.0]");
  std::filesystem::path repo(repo_root);
  if (cfg.model) cfg.model = resolve_under(cfg.model, repo, true, "model", false);
  if (cfg.contract_path) cfg.contract_path = resolve_under(cfg.contract_path, repo, true, "contract_path", false);
  if (cfg.zoo_dir) cfg.zoo_dir = resolve_under(cfg.zoo_dir, repo, true, "zoo_dir", false);
  if (cfg.wake_model_dir) cfg.wake_model_dir = resolve_under(cfg.wake_model_dir, repo, false, "wake_model_dir", true);
  if (cfg.wake_phrases_json) cfg.wake_phrases_json = resolve_under(cfg.wake_phrases_json, repo, true, "wake_phrases_json", true);
  return cfg;
}

std::vector<std::string> build_pipeline_argv(const ControllerConfig& cfg_in, const std::string& repo_root, const std::string& rt_bin) {
  ControllerConfig cfg = validate_controller_config(cfg_in, repo_root);
  std::vector<std::string> argv;
  if (cfg.runtime == "python") {
    argv.push_back((std::filesystem::path(repo_root) / "edge" / "inference_pipeline.py").string());
  } else {
    argv.push_back(rt_bin.empty() ? std::string("soundbubble-rt") : rt_bin);
  }
  auto add = [&](const std::string& flag, const std::string& value) {
    if (!value.empty()) {
      argv.push_back(flag);
      argv.push_back(value);
    }
  };
  auto add_num = [&](const std::string& flag, double value) {
    argv.push_back(flag);
    std::ostringstream oss;
    oss << value;
    argv.push_back(oss.str());
  };
  if (cfg.model) add("--model", *cfg.model);
  if (cfg.contract_path) add("--contract-path", *cfg.contract_path);
  if (cfg.zoo_dir) add("--zoo-dir", *cfg.zoo_dir);
  if (cfg.bubble_radius) add_num("--bubble-radius", *cfg.bubble_radius);
  if (cfg.input_device) add_num("--input-device", *cfg.input_device);
  if (cfg.output_device) add_num("--output-device", *cfg.output_device);
  add_num("--input-channels", cfg.input_channels);
  add_num("--output-channels", cfg.output_channels);
  add("--channel-map", cfg.channel_map);
  add_num("--monitor-input-channel", cfg.monitor_input_channel);
  add_num("--input-sr", cfg.input_sr);
  add_num("--model-sr", cfg.model_sr);
  add_num("--output-sr", cfg.output_sr);
  add_num("--model-chunk", cfg.model_chunk);
  add_num("--model-pad", cfg.model_pad);
  add_num("--model-num-ch", cfg.model_num_ch);
  add_num("--hops-per-io", cfg.hops_per_io);
  if (cfg.output_device_block > 0) add_num("--output-device-block", cfg.output_device_block);
  if (!cfg.duplex_mode.empty() && cfg.duplex_mode != "off") add("--duplex-mode", cfg.duplex_mode);
  if (cfg.duplex_block > 0) add_num("--duplex-block", cfg.duplex_block);
  add("--latency", cfg.latency);
  add_num("--max-capture-blocks", cfg.max_capture_blocks);
  add_num("--max-output-ahead-sec", cfg.max_output_ahead_sec);
  add_num("--target-output-ahead-sec", cfg.target_output_ahead_sec);
  add_num("--rt-fifo-priority", cfg.rt_fifo_priority);
  add_num("--output-gain-db", cfg.output_gain_db);
  if (!cfg.output_limiter) argv.push_back("--no-output-limiter");
  add_num("--limiter-ceiling", cfg.limiter_ceiling);
  add_num("--limiter-release-ms", cfg.limiter_release_ms);
  add_num("--mix-dry", cfg.mix_dry);
  add_num("--input-gain-db", cfg.input_gain_db);
  add_num("--alpha", cfg.alpha);
  add_num("--silence-dbfs", cfg.silence_dbfs);
  add_num("--silence-window-frames", cfg.silence_window_frames);
  add_num("--fault-min-dbfs", cfg.fault_min_dbfs);
  add_num("--fault-max-dbfs", cfg.fault_max_dbfs);
  add_num("--max-overrun-ms", cfg.max_overrun_ms);
  add_num("--intra-op-threads", cfg.intra_op_threads);
  add_num("--inter-op-threads", cfg.inter_op_threads);
  add_num("--profile-interval", cfg.profile_interval);
  add_num("--monitor-interval", cfg.monitor_interval);
  add_num("--model-ring-ms", cfg.model_ring_ms);
  add_num("--max-steps-per-block", cfg.max_steps_per_block);
  if (cfg.wake_model_dir) add("--wake-model-dir", *cfg.wake_model_dir);
  if (cfg.wake_phrases_json) add("--wake-phrases-json", *cfg.wake_phrases_json);
  add_num("--wake-threshold", cfg.wake_threshold);
  add_num("--wake-hold-sec", cfg.wake_hold_sec);
  add_num("--wake-replay-sec", cfg.wake_replay_sec);
  add_num("--wake-replay-rate", cfg.wake_replay_rate);
  add_num("--wake-release-ms", cfg.wake_release_ms);
  if (cfg.profile) argv.push_back("--profile");
  if (cfg.monitor) argv.push_back("--monitor");
  if (cfg.require_soxr) argv.push_back("--require-soxr");
  if (!cfg.mix_uart_port.empty()) {
    add("--mix-uart-port", cfg.mix_uart_port);
    add_num("--mix-uart-baud", cfg.mix_uart_baud);
  }
  for (const auto& phrase : cfg.wake_phrase) {
    argv.push_back("--wake-phrase");
    argv.push_back(phrase);
  }
  return argv;
}

ProfileParse parse_profile_line(const std::string& line) {
  ProfileParse p;
  auto start = line.find("[profile]");
  if (start == std::string::npos) return p;
  try {
    p.capture_to_frame_ms = field_after(line, "capture->frame=", "ms");
    p.dc_ms = field_after(line, "dc=", "ms");
    p.level_valid_ms = field_after(line, "level-valid=", "ms");
    p.frame_to_infer_ms = field_after(line, "frame->infer=", "ms");
    if (line.find("infer_total=") != std::string::npos) p.infer_ms = field_after(line, "infer_total=", "ms");
    else p.infer_ms = field_after(line, "infer=", "ms");
    p.post_out_ms = field_after(line, "post+out=", "ms");
    p.e2e_ms = field_after(line, "e2e=", "ms");
    p.drift_ppm = field_after(line, "drift=", "ppm");
    auto qpos = line.find("q=");
    auto slash = line.find('/', qpos);
    auto qend = line.find("drops_oldest=", slash);
    p.capture_queue_fill = std::stoi(line.substr(qpos + 2, slash - (qpos + 2)));
    p.capture_queue_capacity = std::stoi(line.substr(slash + 1, qend - (slash + 1)));
    p.drops_oldest = static_cast<std::uint64_t>(field_after(line, "drops_oldest=", " "));
    p.out_ahead_ms = field_after(line, "out_ahead=", "ms");
    p.out_dropped_frames = static_cast<std::uint64_t>(field_after(line, "dropped=", " "));
    p.underrun_frames = static_cast<std::uint64_t>(field_after(line, "underrun_frames=", " "));
    p.model_ring_ms = field_after(line, "model_ring=", "ms");
    p.model_ring_drops = static_cast<std::uint64_t>(field_after(line, "drops=", " "));
    p.model_ring_resets = static_cast<std::uint64_t>(field_after(line, "resets=", ""));
    p.ok = true;
  } catch (...) {
    p.ok = false;
  }
  return p;
}

namespace {

bool is_space_char(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }
bool is_digit_char(char c) { return c >= '0' && c <= '9'; }

void ensure_channel(MonitorParse& acc, size_t idx) {
  if (acc.levels_dbfs.size() <= idx) {
    acc.levels_dbfs.resize(idx + 1, 0);
    acc.statuses.resize(idx + 1);
  }
}

bool parse_channel_tokens(const std::string& line, MonitorParse& acc) {
  bool found = false;
  size_t i = 0;
  while (i + 3 < line.size()) {
    auto pos = line.find("ch", i);
    if (pos == std::string::npos || pos + 3 >= line.size()) break;
    if (pos > 0 && ((line[pos - 1] >= '0' && line[pos - 1] <= '9') || (line[pos - 1] >= 'A' && line[pos - 1] <= 'Z') ||
                    (line[pos - 1] >= 'a' && line[pos - 1] <= 'z'))) {
      i = pos + 2;
      continue;
    }
    size_t j = pos + 2;
    if (!is_digit_char(line[j])) {
      i = pos + 2;
      continue;
    }
    int idx = 0;
    while (j < line.size() && is_digit_char(line[j])) {
      idx = idx * 10 + (line[j] - '0');
      ++j;
    }
    if (j >= line.size() || line[j] != ':') {
      i = pos + 2;
      continue;
    }
    ++j;
    while (j < line.size() && is_space_char(line[j])) ++j;
    size_t ns = j;
    if (j < line.size() && (line[j] == '+' || line[j] == '-')) ++j;
    bool saw_digit = false;
    while (j < line.size() && (is_digit_char(line[j]) || line[j] == '.')) {
      if (is_digit_char(line[j])) saw_digit = true;
      ++j;
    }
    if (!saw_digit) {
      i = pos + 2;
      continue;
    }
    double db = 0;
    try {
      db = std::stod(line.substr(ns, j - ns));
    } catch (...) {
      i = pos + 2;
      continue;
    }
    while (j < line.size() && is_space_char(line[j])) ++j;
    if (j + 3 >= line.size() || line[j] != 'd' || line[j + 1] != 'B' || line[j + 2] != 'F' || line[j + 3] != 'S') {
      i = pos + 2;
      continue;
    }
    j += 4;
    while (j < line.size() && is_space_char(line[j])) ++j;
    size_t ss = j;
    while (j < line.size() && !is_space_char(line[j])) ++j;
    if (j == ss) {
      i = pos + 2;
      continue;
    }
    ensure_channel(acc, static_cast<size_t>(idx));
    // New Python/C++ monitor block always starts with ch0 — drop prior block tails
    // so speakers=/model_out cannot stick after a count=0 hop.
    if (idx == 0) {
      acc.speaker_count.reset();
      acc.distance_m.reset();
      acc.distance_status.clear();
      acc.model_out_db.reset();
      acc.output_db.reset();
      acc.speakers.clear();
    }
    acc.levels_dbfs[static_cast<size_t>(idx)] = db;
    acc.statuses[static_cast<size_t>(idx)] = line.substr(ss, j - ss);
    found = true;
    i = j;
  }
  return found;
}

bool parse_distance_token(const std::string& line, MonitorParse& acc) {
  auto pos = line.find("distance=");
  if (pos == std::string::npos) return false;
  std::string rest = line.substr(pos + 9);
  auto paren = rest.find('(');
  auto mpos = rest.find('m');
  if (mpos != std::string::npos && (paren == std::string::npos || mpos < paren)) {
    try {
      acc.distance_m = std::stod(rest.substr(0, mpos));
    } catch (...) {
      acc.distance_m.reset();
    }
    if (paren != std::string::npos) {
      auto end = rest.find(')', paren);
      if (end != std::string::npos) acc.distance_status = rest.substr(paren + 1, end - paren - 1);
    }
  } else {
    acc.distance_m.reset();
    auto end = rest.find_first_of(" \t\r");
    acc.distance_status = rest.substr(0, end);
  }
  return true;
}

bool parse_speaker_token(const std::string& line, MonitorParse& acc) {
  auto pos = line.find("speaker_count=");
  if (pos == std::string::npos) return false;
  std::string rest = line.substr(pos + 14);
  auto paren = rest.find('(');
  if (paren != std::string::npos && paren > 0 && rest[paren - 1] == ' ') {
    std::string count = rest.substr(0, paren - 1);
    while (!count.empty() && std::isspace(static_cast<unsigned char>(count.back()))) count.pop_back();
    acc.speaker_count = count;
  } else {
    acc.speaker_count.reset();
  }
  return true;
}

bool parse_model_out_token(const std::string& line, MonitorParse& acc) {
  auto pos = line.find("model_out=");
  if (pos == std::string::npos) return false;
  try {
    size_t i = pos + 10;
    while (i < line.size() && is_space_char(line[i])) ++i;
    size_t ns = i;
    if (i < line.size() && (line[i] == '+' || line[i] == '-')) ++i;
    while (i < line.size() && (is_digit_char(line[i]) || line[i] == '.')) ++i;
    acc.model_out_db = std::stod(line.substr(ns, i - ns));
    auto out_pos = line.find("out=", i);
    if (out_pos != std::string::npos) {
      size_t j = out_pos + 4;
      while (j < line.size() && is_space_char(line[j])) ++j;
      size_t os = j;
      if (j < line.size() && (line[j] == '+' || line[j] == '-')) ++j;
      while (j < line.size() && (is_digit_char(line[j]) || line[j] == '.')) ++j;
      acc.output_db = std::stod(line.substr(os, j - os));
    }
  } catch (...) {
    return false;
  }
  return true;
}

bool parse_speakers_list_token(const std::string& line, MonitorParse& acc) {
  auto pos = line.find("speakers=[");
  if (pos == std::string::npos) return false;
  auto end = line.find(']', pos);
  if (end == std::string::npos) return false;
  acc.speakers = line.substr(pos + 9, end - (pos + 9) + 1);  // includes [ … ]
  return true;
}

Json profile_to_json(const ProfileParse& parsed) {
  Json obj;
  obj.type = Json::Type::Object;
  obj.o["infer_ms"] = json_number(parsed.infer_ms);
  obj.o["capture_to_frame_ms"] = json_number(parsed.capture_to_frame_ms);
  obj.o["dc_ms"] = json_number(parsed.dc_ms);
  obj.o["level_valid_ms"] = json_number(parsed.level_valid_ms);
  obj.o["frame_to_infer_ms"] = json_number(parsed.frame_to_infer_ms);
  obj.o["post_out_ms"] = json_number(parsed.post_out_ms);
  obj.o["e2e_ms"] = json_number(parsed.e2e_ms);
  obj.o["drift_ppm"] = json_number(parsed.drift_ppm);
  obj.o["capture_queue_fill"] = json_number(parsed.capture_queue_fill);
  obj.o["capture_queue_capacity"] = json_number(parsed.capture_queue_capacity);
  obj.o["drops_oldest"] = json_number(static_cast<double>(parsed.drops_oldest));
  obj.o["out_ahead_ms"] = json_number(parsed.out_ahead_ms);
  obj.o["out_dropped_frames"] = json_number(static_cast<double>(parsed.out_dropped_frames));
  obj.o["underrun_frames"] = json_number(static_cast<double>(parsed.underrun_frames));
  obj.o["model_ring_ms"] = json_number(parsed.model_ring_ms);
  obj.o["model_ring_drops"] = json_number(static_cast<double>(parsed.model_ring_drops));
  obj.o["model_ring_resets"] = json_number(static_cast<double>(parsed.model_ring_resets));
  return obj;
}

Json monitor_to_json(const MonitorParse& m) {
  Json obj;
  obj.type = Json::Type::Object;
  Json levels;
  levels.type = Json::Type::Array;
  for (double v : m.levels_dbfs) levels.a.push_back(json_number(v));
  Json statuses;
  statuses.type = Json::Type::Array;
  for (const auto& s : m.statuses) statuses.a.push_back(json_string(s));
  obj.o["levels_dbfs"] = std::move(levels);
  obj.o["statuses"] = std::move(statuses);
  obj.o["distance_m"] = m.distance_m ? json_number(*m.distance_m) : Json{};
  obj.o["distance_status"] = json_string(m.distance_status);
  obj.o["speaker_count"] = m.speaker_count ? json_string(*m.speaker_count) : Json{};
  obj.o["bubble_radius_m"] = m.bubble_radius_m ? json_number(*m.bubble_radius_m) : Json{};
  obj.o["model_out_db"] = m.model_out_db ? json_number(*m.model_out_db) : Json{};
  obj.o["output_db"] = m.output_db ? json_number(*m.output_db) : Json{};
  obj.o["speakers"] = m.speakers.empty() ? Json{} : json_string(m.speakers);
  return obj;
}

std::string query_param(const std::string& path, const std::string& key) {
  auto q = path.find('?');
  if (q == std::string::npos) return {};
  std::string qs = path.substr(q + 1);
  std::string prefix = key + "=";
  size_t i = 0;
  while (i < qs.size()) {
    auto amp = qs.find('&', i);
    std::string part = qs.substr(i, amp == std::string::npos ? std::string::npos : amp - i);
    if (part.rfind(prefix, 0) == 0) {
      std::string val = part.substr(prefix.size());
      std::string out;
      for (size_t c = 0; c < val.size(); ++c) {
        if (val[c] == '%' && c + 2 < val.size()) {
          auto hex = val.substr(c + 1, 2);
          out.push_back(static_cast<char>(std::stoi(hex, nullptr, 16)));
          c += 2;
        } else if (val[c] == '+') {
          out.push_back(' ');
        } else {
          out.push_back(val[c]);
        }
      }
      return out;
    }
    if (amp == std::string::npos) break;
    i = amp + 1;
  }
  return {};
}

const char* http_reason(int status) {
  if (status == 200) return "OK";
  if (status == 400) return "Bad Request";
  if (status == 404) return "Not Found";
  if (status == 409) return "Conflict";
  return "ERR";
}

}  // namespace

bool parse_monitor_line(const std::string& line, MonitorParse& acc) {
  bool hit = false;
  if (parse_channel_tokens(line, acc)) hit = true;
  if (parse_distance_token(line, acc)) hit = true;
  if (parse_speaker_token(line, acc)) hit = true;
  if (parse_model_out_token(line, acc)) hit = true;
  if (parse_speakers_list_token(line, acc)) hit = true;
  if (hit) acc.any = true;
  return hit;
}

std::string controller_config_to_json(const ControllerConfig& cfg) {
  Json j;
  j.type = Json::Type::Object;
  auto opt_s = [&](const char* key, const std::optional<std::string>& value) {
    j.o[key] = value ? json_string(*value) : Json{};
  };
  auto opt_n = [&](const char* key, const std::optional<double>& value) {
    j.o[key] = value ? json_number(*value) : Json{};
  };
  auto opt_i = [&](const char* key, const std::optional<int>& value) {
    j.o[key] = value ? json_number(*value) : Json{};
  };
  opt_s("model", cfg.model);
  opt_s("contract_path", cfg.contract_path);
  opt_s("zoo_dir", cfg.zoo_dir);
  opt_n("bubble_radius", cfg.bubble_radius);
  opt_i("input_device", cfg.input_device);
  opt_i("output_device", cfg.output_device);
  j.o["input_channels"] = json_number(cfg.input_channels);
  j.o["output_channels"] = json_number(cfg.output_channels);
  j.o["channel_map"] = json_string(cfg.channel_map);
  j.o["monitor_input_channel"] = json_number(cfg.monitor_input_channel);
  j.o["input_sr"] = json_number(cfg.input_sr);
  j.o["model_sr"] = json_number(cfg.model_sr);
  j.o["output_sr"] = json_number(cfg.output_sr);
  j.o["model_chunk"] = json_number(cfg.model_chunk);
  j.o["model_pad"] = json_number(cfg.model_pad);
  j.o["model_num_ch"] = json_number(cfg.model_num_ch);
  j.o["hops_per_io"] = json_number(cfg.hops_per_io);
  j.o["output_device_block"] = json_number(cfg.output_device_block);
  j.o["duplex_mode"] = json_string(cfg.duplex_mode);
  j.o["duplex_block"] = json_number(cfg.duplex_block);
  j.o["latency"] = json_string(cfg.latency);
  j.o["max_capture_blocks"] = json_number(cfg.max_capture_blocks);
  j.o["max_output_ahead_sec"] = json_number(cfg.max_output_ahead_sec);
  j.o["target_output_ahead_sec"] = json_number(cfg.target_output_ahead_sec);
  j.o["rt_fifo_priority"] = json_number(cfg.rt_fifo_priority);
  j.o["output_gain_db"] = json_number(cfg.output_gain_db);
  j.o["output_limiter"] = json_bool(cfg.output_limiter);
  j.o["limiter_ceiling"] = json_number(cfg.limiter_ceiling);
  j.o["limiter_release_ms"] = json_number(cfg.limiter_release_ms);
  j.o["mix_dry"] = json_number(cfg.mix_dry);
  j.o["input_gain_db"] = json_number(cfg.input_gain_db);
  j.o["alpha"] = json_number(cfg.alpha);
  j.o["silence_dbfs"] = json_number(cfg.silence_dbfs);
  j.o["silence_window_frames"] = json_number(cfg.silence_window_frames);
  j.o["fault_min_dbfs"] = json_number(cfg.fault_min_dbfs);
  j.o["fault_max_dbfs"] = json_number(cfg.fault_max_dbfs);
  j.o["max_overrun_ms"] = json_number(cfg.max_overrun_ms);
  j.o["intra_op_threads"] = json_number(cfg.intra_op_threads);
  j.o["inter_op_threads"] = json_number(cfg.inter_op_threads);
  j.o["model_ring_ms"] = json_number(cfg.model_ring_ms);
  j.o["max_steps_per_block"] = json_number(cfg.max_steps_per_block);
  j.o["require_soxr"] = json_bool(cfg.require_soxr);
  opt_s("wake_model_dir", cfg.wake_model_dir);
  opt_s("wake_phrases_json", cfg.wake_phrases_json);
  Json phrases;
  phrases.type = Json::Type::Array;
  for (const auto& phrase : cfg.wake_phrase) phrases.a.push_back(json_string(phrase));
  j.o["wake_phrase"] = std::move(phrases);
  j.o["wake_threshold"] = json_number(cfg.wake_threshold);
  j.o["wake_hold_sec"] = json_number(cfg.wake_hold_sec);
  j.o["wake_replay_sec"] = json_number(cfg.wake_replay_sec);
  j.o["wake_replay_rate"] = json_number(cfg.wake_replay_rate);
  j.o["wake_release_ms"] = json_number(cfg.wake_release_ms);
  j.o["profile"] = json_bool(cfg.profile);
  j.o["profile_interval"] = json_number(cfg.profile_interval);
  j.o["monitor"] = json_bool(cfg.monitor);
  j.o["monitor_interval"] = json_number(cfg.monitor_interval);
  j.o["mix_uart_port"] = json_string(cfg.mix_uart_port);
  j.o["mix_uart_baud"] = json_number(cfg.mix_uart_baud);
  j.o["runtime"] = json_string(cfg.runtime);
  return dump_json(j);
}

ProcessManager::ProcessManager(std::string repo_root, std::string rt_bin, std::string python_exe)
    : repo_(std::move(repo_root)), rt_bin_(std::move(rt_bin)), python_(std::move(python_exe)) {
  if (const char* env = std::getenv("SOUNDBUBBLE_RUNTIME")) {
    std::string runtime = env;
    if (runtime == "cpp" || runtime == "python") runtime_ = runtime;
  }
  observe_thread_ = std::thread([this] { observe_main(); });
}

ProcessManager::~ProcessManager() {
  observe_stop_.store(true);
#if !defined(_WIN32)
  int jpid = 0;
  {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    jpid = journal_pid_;
  }
  if (jpid > 0) kill(static_cast<pid_t>(jpid), SIGTERM);
#endif
  if (observe_thread_.joinable()) observe_thread_.join();
  stop();
}

void ProcessManager::reap_exited_child() {
#if defined(_WIN32)
  std::lock_guard<std::recursive_mutex> lock(mu_);
  if (!process_) return;
  HANDLE handle = static_cast<HANDLE>(process_);
  DWORD code = 0;
  if (WaitForSingleObject(handle, 0) != WAIT_OBJECT_0) return;
  GetExitCodeProcess(handle, &code);
  CloseHandle(handle);
  returncode_ = static_cast<int>(code);
  process_ = nullptr;
  pid_.reset();
#else
  std::thread out;
  std::thread err;
  {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (!process_) return;
    pid_t child = static_cast<pid_t>(reinterpret_cast<intptr_t>(process_));
    int status = 0;
    pid_t r = waitpid(child, &status, WNOHANG);
    if (r != child) return;
    returncode_ = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    process_ = nullptr;
    pid_.reset();
    out = std::move(out_thread_);
    err = std::move(err_thread_);
  }
  if (out.joinable()) out.join();
  if (err.joinable()) err.join();
#endif
}

bool ProcessManager::is_running() {
  reap_exited_child();
  std::lock_guard<std::recursive_mutex> lock(mu_);
  return process_ != nullptr;
}

void ProcessManager::note_runtime(const std::string& line) {
  auto pos = line.find("runtime=");
  if (pos == std::string::npos) return;
  std::string rest = line.substr(pos + 8);
  if (rest.rfind("python", 0) == 0) runtime_ = "python";
  else if (rest.rfind("cpp", 0) == 0) runtime_ = "cpp";
}

void ProcessManager::ingest_line(const std::string& line) { append(line, "stdout"); }

void ProcessManager::note_owned_config_for_test(const std::string& config_json) {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  ControllerConfig cfg = controller_config_from_json(config_json.empty() ? "{}" : config_json);
  owned_config_json_ = controller_config_to_json(cfg);
  test_owned_ = true;
}

void ProcessManager::append(const std::string& line, const std::string& stream) {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  if (stream == "journal" && process_ != nullptr) return;
  note_runtime(line);
  std::string stamped = "[" + stream + "] " + line;
  logs_.push_back(stamped);
  if (logs_.size() > 5000) logs_.pop_front();
  if (stream == "stderr" || line.find("error") != std::string::npos || line.find("exception") != std::string::npos) {
    errors_.push_back(stamped);
    if (errors_.size() > 300) errors_.pop_front();
  }
  auto parsed = parse_profile_line(line);
  if (parsed.ok) {
    Json obj = profile_to_json(parsed);
    obj.o["received_at"] = json_string(iso_now());
    last_profile_json_ = dump_json(obj);
  }
  if (parse_monitor_line(line, monitor_)) has_monitor_ = true;
}

std::string ProcessManager::status_json() {
  reap_exited_child();
  std::lock_guard<std::recursive_mutex> lock(mu_);
  Json j;
  j.type = Json::Type::Object;
  j.o["running"] = json_bool(process_ != nullptr);
  j.o["pid"] = pid_ ? json_number(static_cast<double>(*pid_)) : Json{};
  j.o["returncode"] = returncode_ ? json_number(*returncode_) : Json{};
  j.o["started_at"] = started_at_.empty() ? Json{} : json_string(started_at_);
  Json args;
  args.type = Json::Type::Array;
  for (const auto& a : argv_) args.a.push_back(json_string(a));
  j.o["args"] = args;
  j.o["last_profile"] = last_profile_json_.empty() ? Json{} : parse_json(last_profile_json_);
  if ((process_ != nullptr || test_owned_) && !owned_config_json_.empty()) j.o["config"] = parse_json(owned_config_json_);
  else j.o["config"] = Json{};
  j.o["runtime"] = json_string(runtime_);
  if (has_monitor_) {
    MonitorParse snap = monitor_;
    if (!snap.bubble_radius_m && bubble_radius_) snap.bubble_radius_m = bubble_radius_;
    j.o["last_monitor"] = monitor_to_json(snap);
  } else {
    j.o["last_monitor"] = Json{};
  }
  return dump_json(j);
}

std::string ProcessManager::logs_json(int tail) const {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  tail = std::max(1, std::min(tail, 5000));
  Json j;
  j.type = Json::Type::Object;
  Json lines;
  lines.type = Json::Type::Array;
  int start = std::max(0, static_cast<int>(logs_.size()) - tail);
  int i = 0;
  for (const auto& line : logs_) {
    if (i++ >= start) lines.a.push_back(json_string(line));
  }
  j.o["lines"] = lines;
  return dump_json(j);
}

std::string ProcessManager::start(const std::string& config_json) {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  if (process_) throw std::runtime_error("Pipeline is already running. Use /restart to replace it.");
  ensure_production_stopped();
  ControllerConfig cfg = controller_config_from_json(config_json.empty() ? "{}" : config_json);
  argv_ = build_pipeline_argv(cfg, repo_, rt_bin_);
  std::vector<std::string> cmd = argv_;
  std::string python_exe = python_;
  if (cfg.runtime == "python") {
    // Match deploy_soundbubble.sh: venv python + PYTHONPATH covering lab `src`.
    if (python_exe.empty() || python_exe == "python") {
      if (const char* env = std::getenv("SOUNDBUBBLE_PYTHON")) {
        if (env[0] != '\0') python_exe = env;
      }
    }
    if (python_exe.empty() || python_exe == "python" || python_exe == "python3") {
#if !defined(_WIN32)
      const char* fallback = "/home/sonitude/Sonitude/Sound_Bubble/.venv/bin/python";
      if (access(fallback, X_OK) == 0) python_exe = fallback;
      else
#endif
        python_exe = python_exe.empty() || python_exe == "python" ? "python3" : python_exe;
    }
    cmd.insert(cmd.begin(), python_exe);
    cmd.insert(cmd.begin() + 1, "-u");
  }
#if defined(_WIN32)
  std::string joined;
  for (const auto& part : cmd) {
    if (!joined.empty()) joined.push_back(' ');
    joined += "\"" + part + "\"";
  }
  STARTUPINFOA si{};
  si.cb = sizeof(si);
  PROCESS_INFORMATION pi{};
  std::vector<char> mutable_cmd(joined.begin(), joined.end());
  mutable_cmd.push_back('\0');
  if (!CreateProcessA(nullptr, mutable_cmd.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, repo_.c_str(), &si, &pi)) {
    throw std::runtime_error("failed to start pipeline");
  }
  CloseHandle(pi.hThread);
  process_ = pi.hProcess;
  pid_ = static_cast<std::intptr_t>(pi.dwProcessId);
#else
  int out_pipe[2];
  int err_pipe[2];
  if (pipe(out_pipe) != 0 || pipe(err_pipe) != 0) throw std::runtime_error("pipe");
  pid_t child = fork();
  if (child < 0) {
    close(out_pipe[0]);
    close(out_pipe[1]);
    close(err_pipe[0]);
    close(err_pipe[1]);
    throw std::runtime_error("fork");
  }
  if (child == 0) {
    dup2(out_pipe[1], STDOUT_FILENO);
    dup2(err_pipe[1], STDERR_FILENO);
    close(out_pipe[0]);
    close(out_pipe[1]);
    close(err_pipe[0]);
    close(err_pipe[1]);
    if (cfg.runtime == "python") {
      // Same layout as edge/rpi5/deploy_soundbubble.sh:
      // ROOT + ROOT/edge for edge.*; SRC_ROOT for lab `src.utils`.
      const char* src_env = std::getenv("SOUNDBUBBLE_SRC_ROOT");
      std::string src_root =
          (src_env && src_env[0] != '\0') ? src_env : "/home/sonitude/Sonitude/Sound_Bubble";
      std::string pypath = repo_ + ":" + repo_ + "/edge:" + src_root;
      if (const char* old = std::getenv("PYTHONPATH")) {
        if (old[0] != '\0') {
          pypath.push_back(':');
          pypath += old;
        }
      }
      setenv("PYTHONPATH", pypath.c_str(), 1);
      setenv("PYTHONUNBUFFERED", "1", 1);
    }
    std::vector<char*> cargv;
    std::vector<std::string> storage = cmd;
    for (auto& s : storage) cargv.push_back(s.data());
    cargv.push_back(nullptr);
    if (chdir(repo_.c_str()) != 0) _exit(127);
    execvp(cargv[0], cargv.data());
    _exit(127);
  }
  close(out_pipe[1]);
  close(err_pipe[1]);
  process_ = reinterpret_cast<void*>(static_cast<intptr_t>(child));
  pid_ = child;
  out_thread_ = std::thread([this, fd = out_pipe[0]] { reader_fd(fd, "stdout"); });
  err_thread_ = std::thread([this, fd = err_pipe[0]] { reader_fd(fd, "stderr"); });
#endif
  started_at_ = iso_now();
  logs_.clear();
  errors_.clear();
  last_profile_json_.clear();
  monitor_ = {};
  has_monitor_ = false;
  taps_armed_ = false;
  taps_select_.clear();
  taps_truncated_ = false;
  runtime_ = cfg.runtime;
  bubble_radius_ = cfg.bubble_radius;
  config_json_ = config_json;
  owned_config_json_ = controller_config_to_json(cfg);
  test_owned_ = false;
  returncode_.reset();
  return status_json();
}

std::string ProcessManager::stop() {
#if defined(_WIN32)
  HANDLE handle = nullptr;
  {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    handle = static_cast<HANDLE>(process_);
  }
  if (!handle) return status_json();
  TerminateProcess(handle, 0);
  WaitForSingleObject(handle, 5000);
  DWORD code = 0;
  GetExitCodeProcess(handle, &code);
  CloseHandle(handle);
  std::lock_guard<std::recursive_mutex> lock(mu_);
  returncode_ = static_cast<int>(code);
  process_ = nullptr;
  pid_.reset();
  owned_config_json_.clear();
  test_owned_ = false;
#else
  pid_t child = 0;
  std::thread out;
  std::thread err;
  {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (!process_) return status_json();
    child = static_cast<pid_t>(reinterpret_cast<intptr_t>(process_));
    out = std::move(out_thread_);
    err = std::move(err_thread_);
  }
  kill(child, SIGTERM);
  int status = 0;
  waitpid(child, &status, 0);
  if (out.joinable()) out.join();
  if (err.joinable()) err.join();
  std::lock_guard<std::recursive_mutex> lock(mu_);
  returncode_ = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  process_ = nullptr;
  pid_.reset();
  owned_config_json_.clear();
  test_owned_ = false;
#endif
  return status_json();
}

std::string ProcessManager::restart(const std::string& config_json) {
  stop();
  return start(config_json);
}

void ProcessManager::reader_fd(int fd, const std::string& stream) {
#if defined(_WIN32)
  (void)fd;
  (void)stream;
#else
  std::string buf;
  char tmp[1024];
  while (true) {
    ssize_t n = ::read(fd, tmp, sizeof(tmp));
    if (n <= 0) break;
    buf.append(tmp, tmp + n);
    size_t pos = 0;
    while ((pos = buf.find('\n')) != std::string::npos) {
      std::string line = buf.substr(0, pos);
      buf.erase(0, pos + 1);
      if (!line.empty() && line.back() == '\r') line.pop_back();
      append(line, stream);
    }
  }
  if (!buf.empty()) append(buf, stream);
  ::close(fd);
#endif
}

bool ProcessManager::unit_is_active() const {
#if defined(_WIN32)
  return false;
#else
  int rc = std::system("systemctl is-active --quiet sonitude-soundbubble");
  return rc == 0;
#endif
}

void ProcessManager::ensure_production_stopped() {
#if defined(_WIN32)
  return;
#else
  if (!unit_is_active()) {
    unit_active_ = false;
    return;
  }
  // Non-interactive: installed via edge/rpi5/install_soundbubble_ctl_sudoers.sh
  int rc = std::system("sudo -n systemctl stop sonitude-soundbubble >/dev/null 2>&1");
  if (rc != 0) {
    throw std::runtime_error(
        "failed to stop sonitude-soundbubble before owned start "
        "(install passwordless sudo: sudo bash edge/rpi5/install_soundbubble_ctl_sudoers.sh)");
  }
  for (int i = 0; i < 50; ++i) {
    if (!unit_is_active()) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if (unit_is_active()) {
    throw std::runtime_error("sonitude-soundbubble still active after stop");
  }
  unit_active_ = false;
  append("[controller] stopped production unit sonitude-soundbubble", "stdout");
#endif
}

void ProcessManager::observe_main() {
#if defined(_WIN32)
  return;
#else
  while (!observe_stop_.load()) {
    if (is_running()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(400));
      continue;
    }
    bool active = unit_is_active();
    {
      std::lock_guard<std::recursive_mutex> lock(mu_);
      unit_active_ = active;
    }
    if (!active) {
      std::this_thread::sleep_for(std::chrono::seconds(2));
      continue;
    }
    follow_journal();
  }
#endif
}

void ProcessManager::follow_journal() {
#if defined(_WIN32)
  return;
#else
  int fds[2];
  if (pipe(fds) != 0) return;
  pid_t child = fork();
  if (child < 0) {
    close(fds[0]);
    close(fds[1]);
    return;
  }
  if (child == 0) {
    dup2(fds[1], STDOUT_FILENO);
    dup2(fds[1], STDERR_FILENO);
    close(fds[0]);
    close(fds[1]);
    execlp("journalctl", "journalctl", "-u", "sonitude-soundbubble", "-f", "-o", "cat", "-n", "50", "--no-pager",
           static_cast<char*>(nullptr));
    _exit(127);
  }
  close(fds[1]);
  {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    journal_pid_ = static_cast<int>(child);
  }
  std::string buf;
  auto last_check = std::chrono::steady_clock::now();
  while (!observe_stop_.load() && !is_running()) {
    pollfd pfd{};
    pfd.fd = fds[0];
    pfd.events = POLLIN;
    int pr = poll(&pfd, 1, 500);
    if (pr < 0) break;
    auto now = std::chrono::steady_clock::now();
    if (now - last_check > std::chrono::seconds(5)) {
      last_check = now;
      bool active = unit_is_active();
      std::lock_guard<std::recursive_mutex> lock(mu_);
      unit_active_ = active;
      if (!active) break;
    }
    if (pr == 0) continue;
    char tmp[1024];
    ssize_t n = ::read(fds[0], tmp, sizeof(tmp));
    if (n <= 0) break;
    buf.append(tmp, tmp + n);
    size_t pos = 0;
    while ((pos = buf.find('\n')) != std::string::npos) {
      std::string line = buf.substr(0, pos);
      buf.erase(0, pos + 1);
      if (!line.empty() && line.back() == '\r') line.pop_back();
      append(line, "journal");
    }
  }
  close(fds[0]);
  kill(child, SIGTERM);
  waitpid(child, nullptr, 0);
  std::lock_guard<std::recursive_mutex> lock(mu_);
  journal_pid_ = 0;
#endif
}

std::string ProcessManager::measure_root() const {
  if (const char* env = std::getenv("SOUNDBUBBLE_MEASURE_DIR")) {
    if (env[0] != '\0') return env;
  }
  return (std::filesystem::path(repo_) / "measure").string();
}

void ProcessManager::load_trial_records(const std::string& run_id, const std::string& since, std::string& resolved_run,
                                        std::vector<std::string>& raw_lines) const {
  resolved_run = run_id;
  raw_lines.clear();
  if (run_id.find("..") != std::string::npos || run_id.find('/') != std::string::npos || run_id.find('\\') != std::string::npos) {
    throw std::invalid_argument("invalid run_id");
  }
  namespace fs = std::filesystem;
  fs::path root(measure_root());
  fs::path file;
  if (!run_id.empty()) {
    file = root / run_id / "telemetry.jsonl";
  } else if (fs::is_directory(root)) {
    file = root / "telemetry.jsonl";
    if (!fs::exists(file)) {
      std::optional<fs::file_time_type> best;
      for (const auto& ent : fs::directory_iterator(root)) {
        if (!ent.is_directory()) continue;
        fs::path cand = ent.path() / "telemetry.jsonl";
        if (!fs::exists(cand)) continue;
        auto mt = fs::last_write_time(cand);
        if (!best || mt >= *best) {
          best = mt;
          file = cand;
          resolved_run = ent.path().filename().string();
        }
      }
    }
  }
  if (file.empty() || !fs::exists(file)) return;
  double since_n = 0;
  bool have_since = false;
  if (!since.empty()) {
    try {
      since_n = std::stod(since);
      have_since = since_n != 0;
    } catch (...) {
      have_since = false;
    }
  }
  std::ifstream in(file);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    if (have_since) {
      try {
        Json row = parse_json(line);
        double key = 0;
        if (const Json* s = row.find("seq")) key = s->n;
        else if (const Json* t = row.find("t")) key = t->n;
        if (key <= since_n) continue;
      } catch (...) {
        continue;
      }
    }
    raw_lines.push_back(line);
  }
}

std::string ProcessManager::trials_json(const std::string& run_id, const std::string& since) const {
  std::string resolved;
  std::vector<std::string> lines;
  load_trial_records(run_id, since, resolved, lines);
  Json records;
  records.type = Json::Type::Array;
  for (const auto& line : lines) {
    try {
      records.a.push_back(parse_json(line));
    } catch (...) {
    }
  }
  Json j;
  j.type = Json::Type::Object;
  j.o["run_id"] = resolved.empty() ? Json{} : json_string(resolved);
  j.o["records"] = std::move(records);
  return dump_json(j);
}

std::string ProcessManager::telemetry_json() {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  ++seq_;
  std::string source = "idle";
  if (process_ != nullptr) source = "owned";
  else if (unit_active_) source = "observe";
  Json j;
  j.type = Json::Type::Object;
  j.o["source"] = json_string(source);
  j.o["runtime"] = json_string(runtime_);
  j.o["host"] = json_string("pi_rig");
  j.o["seq"] = json_number(static_cast<double>(seq_));
  j.o["last_profile"] = last_profile_json_.empty() ? Json{} : parse_json(last_profile_json_);
  if ((process_ != nullptr || test_owned_) && !owned_config_json_.empty()) j.o["config"] = parse_json(owned_config_json_);
  else j.o["config"] = Json{};
  if (has_monitor_) {
    MonitorParse snap = monitor_;
    if (!snap.bubble_radius_m && bubble_radius_) snap.bubble_radius_m = bubble_radius_;
    j.o["last_monitor"] = monitor_to_json(snap);
  } else {
    j.o["last_monitor"] = Json{};
  }
  Json taps;
  taps.type = Json::Type::Object;
  taps.o["armed"] = json_bool(taps_armed_);
  taps.o["select"] = json_string(taps_select_);
  taps.o["truncated"] = json_bool(taps_truncated_);
  j.o["taps"] = std::move(taps);
  Json tail;
  tail.type = Json::Type::Array;
  try {
    std::string resolved;
    std::vector<std::string> lines;
    load_trial_records("", "", resolved, lines);
    size_t begin = lines.size() > 20 ? lines.size() - 20 : 0;
    for (size_t i = begin; i < lines.size(); ++i) {
      try {
        tail.a.push_back(parse_json(lines[i]));
      } catch (...) {
      }
    }
  } catch (...) {
  }
  j.o["trials_tail"] = std::move(tail);
  return dump_json(j);
}

TapResult ProcessManager::signal_taps(const std::string& action) {
  if (action != "arm" && action != "disarm") {
    Json err;
    err.type = Json::Type::Object;
    err.o["error"] = json_string("action must be arm or disarm");
    return TapResult{400, dump_json(err)};
  }
  std::lock_guard<std::recursive_mutex> lock(mu_);
  if (process_ == nullptr) {
    Json err;
    err.type = Json::Type::Object;
    err.o["error"] = json_string("not owned");
    return TapResult{409, dump_json(err)};
  }
#if defined(_WIN32)
  Json err;
  err.type = Json::Type::Object;
  err.o["error"] = json_string("SIGUSR1 is not available on this host");
  return TapResult{400, dump_json(err)};
#else
  pid_t child = static_cast<pid_t>(reinterpret_cast<intptr_t>(process_));
  if (kill(child, SIGUSR1) != 0) {
    Json err;
    err.type = Json::Type::Object;
    err.o["error"] = json_string("signal failed");
    return TapResult{400, dump_json(err)};
  }
  taps_armed_ = !taps_armed_;
  Json ok;
  ok.type = Json::Type::Object;
  ok.o["ok"] = json_bool(true);
  ok.o["action"] = json_string("toggle");
  return TapResult{200, dump_json(ok)};
#endif
}

struct ControllerServer::Impl {
  std::string host;
  int port = 0;
  std::string token;
  ProcessManager* manager = nullptr;
  std::atomic<bool> stop{false};
#if defined(_WIN32)
  SOCKET listen_fd = INVALID_SOCKET;
#else
  int listen_fd = -1;
#endif
};

ControllerServer::ControllerServer(std::string host, int port, std::string token, ProcessManager& manager)
    : impl_(std::make_unique<Impl>()) {
  impl_->host = std::move(host);
  impl_->port = port;
  impl_->token = std::move(token);
  impl_->manager = &manager;
}

ControllerServer::~ControllerServer() { stop(); }

void ControllerServer::stop() {
  if (!impl_) return;
  impl_->stop = true;
#if defined(_WIN32)
  if (impl_->listen_fd != INVALID_SOCKET) closesocket(impl_->listen_fd);
#else
  if (impl_->listen_fd >= 0) ::close(impl_->listen_fd);
#endif
}

void ControllerServer::serve() {
#if defined(_WIN32)
  WSADATA wsa;
  WSAStartup(MAKEWORD(2, 2), &wsa);
  impl_->listen_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
#else
  impl_->listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
#endif
  int yes = 1;
  auto open_listen_socket = [&]() {
#if defined(_WIN32)
    impl_->listen_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    setsockopt(impl_->listen_fd, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof(yes));
#else
    impl_->listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    setsockopt(impl_->listen_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
#endif
  };
  // Tailnet address can appear a few seconds after tailscaled.service. Retry
  // for about 30s, then exit so systemd Restart= tries again. Do not loop forever.
  bool bound = false;
  for (int attempt = 0; attempt < 15 && !impl_->stop.load(); ++attempt) {
    if (attempt == 0) {
#if defined(_WIN32)
      setsockopt(impl_->listen_fd, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof(yes));
#else
      setsockopt(impl_->listen_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
#endif
    } else {
#if defined(_WIN32)
      if (impl_->listen_fd != INVALID_SOCKET) closesocket(impl_->listen_fd);
#else
      if (impl_->listen_fd >= 0) ::close(impl_->listen_fd);
#endif
      open_listen_socket();
      std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(impl_->port));
    if (inet_pton(AF_INET, impl_->host == "0.0.0.0" ? "0.0.0.0" : impl_->host.c_str(), &addr.sin_addr) != 1) {
      std::cerr << "[controller] invalid bind address " << impl_->host << "\n";
      return;
    }
#if defined(_WIN32)
    int rc = bind(impl_->listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
#else
    int rc = ::bind(impl_->listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
#endif
    if (rc == 0) {
      bound = true;
      break;
    }
    std::cerr << "[controller] bind " << impl_->host << ":" << impl_->port << " failed; retry " << (attempt + 1)
              << "/15\n";
  }
  if (!bound) {
    std::cerr << "[controller] bind failed after about 30s; exiting so the service manager can restart\n";
    return;
  }
#if defined(_WIN32)
  listen(impl_->listen_fd, 8);
#else
  listen(impl_->listen_fd, 8);
#endif
  std::cout << "[controller] listening on http://" << impl_->host << ":" << impl_->port << "\n";
  while (!impl_->stop.load()) {
#if defined(_WIN32)
    SOCKET client = accept(impl_->listen_fd, nullptr, nullptr);
    if (client == INVALID_SOCKET) break;
#else
    int client = accept(impl_->listen_fd, nullptr, nullptr);
    if (client < 0) break;
#endif
    // Read full request: headers may arrive without the body, and POST /restart
    // bodies exceed a single 8 KiB segment once the bearer + owned config are included.
    std::string req;
    char buf[4096];
    auto recv_chunk = [&]() -> int {
#if defined(_WIN32)
      return recv(client, buf, sizeof(buf), 0);
#else
      return static_cast<int>(::recv(client, buf, sizeof(buf), 0));
#endif
    };
    while (req.find("\r\n\r\n") == std::string::npos) {
      int n = recv_chunk();
      if (n <= 0) break;
      req.append(buf, buf + n);
      if (req.size() > 1024 * 1024) break;
    }
    size_t content_length = 0;
    {
      auto cl = req.find("Content-Length:");
      if (cl == std::string::npos) cl = req.find("content-length:");
      if (cl != std::string::npos) {
        auto end = req.find("\r\n", cl);
        std::string line = end == std::string::npos ? req.substr(cl) : req.substr(cl, end - cl);
        auto colon = line.find(':');
        if (colon != std::string::npos) {
          try {
            content_length = static_cast<size_t>(std::stoul(line.substr(colon + 1)));
          } catch (...) {
            content_length = 0;
          }
        }
      }
    }
    auto body_pos = req.find("\r\n\r\n");
    while (body_pos != std::string::npos && content_length > 0 &&
           req.size() < body_pos + 4 + content_length) {
      int n = recv_chunk();
      if (n <= 0) break;
      req.append(buf, buf + n);
      if (req.size() > 1024 * 1024) break;
    }
    auto line_end = req.find("\r\n");
    std::string request_line = line_end == std::string::npos ? req : req.substr(0, line_end);
    std::istringstream ls(request_line);
    std::string method, path, version;
    ls >> method >> path >> version;
    std::string path_only = path;
    auto q = path_only.find('?');
    if (q != std::string::npos) path_only = path_only.substr(0, q);
    bool authed = impl_->token.empty();
    auto auth = req.find("Authorization:");
    if (auth != std::string::npos) {
      auto end = req.find("\r\n", auth);
      std::string header = req.substr(auth, end - auth);
      authed = header.find("Bearer " + impl_->token) != std::string::npos;
    }
    std::string body;
    if (body_pos != std::string::npos) {
      body = req.substr(body_pos + 4);
      if (content_length > 0 && body.size() > content_length) body.resize(content_length);
    }
    int status = 200;
    std::string payload;
    try {
      if (method == "GET" && path_only == "/health") {
        payload = "{\"ok\":true}";
      } else if (!authed) {
        status = 400;
        payload = "{\"error\":\"missing or invalid bearer token\"}";
      } else if (method == "GET" && path_only == "/status") {
        payload = impl_->manager->status_json();
      } else if (method == "GET" && path_only == "/logs") {
        int tail = 200;
        std::string tail_s = query_param(path, "tail");
        if (!tail_s.empty()) tail = std::stoi(tail_s);
        payload = impl_->manager->logs_json(tail);
      } else if (method == "GET" && path_only == "/telemetry") {
        payload = impl_->manager->telemetry_json();
      } else if (method == "GET" && path_only == "/trials") {
        payload = impl_->manager->trials_json(query_param(path, "run_id"), query_param(path, "since"));
      } else if (method == "GET" && path_only == "/devices") {
        Json devices;
        devices.type = Json::Type::Object;
        Json arr;
        arr.type = Json::Type::Array;
        try {
          for (const auto& d : list_devices()) {
            Json item;
            item.type = Json::Type::Object;
            item.o["index"] = json_number(d.index);
            item.o["name"] = json_string(d.name);
            item.o["hostapi"] = json_string(d.hostapi);
            item.o["max_input_channels"] = json_number(d.max_input_channels);
            item.o["max_output_channels"] = json_number(d.max_output_channels);
            item.o["default_samplerate"] = json_number(d.default_samplerate);
            arr.a.push_back(item);
          }
        } catch (const std::exception& exc) {
          throw std::runtime_error(exc.what());
        }
        devices.o["devices"] = arr;
        payload = dump_json(devices);
      } else if (method == "POST" && (path_only == "/start" || path_only == "/stop" || path_only == "/restart")) {
        std::string config;
        if (!body.empty()) {
          Json j = parse_json(body);
          if (const Json* c = j.find("config")) config = dump_json(*c);
        }
        if (path_only == "/start") payload = impl_->manager->start(config);
        else if (path_only == "/stop") payload = impl_->manager->stop();
        else payload = impl_->manager->restart(config);
      } else if (method == "POST" && path_only == "/taps") {
        std::string action;
        if (!body.empty()) {
          Json j = parse_json(body);
          if (const Json* a = j.find("action"); a && a->type == Json::Type::String) action = a->s;
        }
        TapResult result = impl_->manager->signal_taps(action);
        status = result.status;
        payload = std::move(result.body);
      } else {
        status = 404;
        payload = "{\"error\":\"not found\"}";
      }
    } catch (const std::exception& exc) {
      status = 400;
      Json err;
      err.type = Json::Type::Object;
      err.o["error"] = json_string(exc.what());
      payload = dump_json(err);
    }
    std::ostringstream resp;
    resp << "HTTP/1.1 " << status << " " << http_reason(status) << "\r\nContent-Type: application/json\r\nContent-Length: "
         << payload.size() << "\r\nConnection: close\r\n\r\n"
         << payload;
    std::string bytes = resp.str();
#if defined(_WIN32)
    send(client, bytes.data(), static_cast<int>(bytes.size()), 0);
    closesocket(client);
#else
    ::send(client, bytes.data(), bytes.size(), 0);
    ::close(client);
#endif
  }
}

}  // namespace soundbubble
