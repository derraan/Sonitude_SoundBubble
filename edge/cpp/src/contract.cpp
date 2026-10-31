#include "soundbubble/contract.hpp"
#include "soundbubble/json_mini.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace soundbubble {
namespace {

std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::optional<double> opt_num(const Json& j, const char* key) {
  const Json* v = j.find(key);
  if (!v || v->type == Json::Type::Null) return std::nullopt;
  if (v->type != Json::Type::Number) throw std::runtime_error(std::string(key) + " must be a number");
  return v->n;
}

std::optional<bool> opt_bool(const Json& j, const char* key) {
  const Json* v = j.find(key);
  if (!v || v->type == Json::Type::Null) return std::nullopt;
  if (v->type == Json::Type::Bool) return v->b;
  throw std::runtime_error(std::string(key) + " must be a bool");
}

std::optional<std::string> opt_str(const Json& j, const char* key) {
  const Json* v = j.find(key);
  if (!v || v->type == Json::Type::Null) return std::nullopt;
  if (v->type != Json::Type::String) throw std::runtime_error(std::string(key) + " must be a string");
  return v->s;
}

int req_int(const Json& j, const char* key) {
  const Json* v = j.find(key);
  if (!v || v->type != Json::Type::Number) throw std::runtime_error(std::string("contract missing ") + key);
  return static_cast<int>(std::llround(v->n));
}

}  // namespace

RuntimeContract load_contract(const std::string& path) {
  Json j = parse_json(read_file(path));
  if (j.type != Json::Type::Object) throw std::runtime_error("contract must be a JSON object");
  RuntimeContract c;
  c.model_sr = req_int(j, "model_sr");
  c.model_num_ch = req_int(j, "model_num_ch");
  c.model_chunk = req_int(j, "model_chunk");
  c.model_pad = req_int(j, "model_pad");
  c.frame_len = req_int(j, "frame_len");
  c.expected_input_channels = req_int(j, "expected_input_channels");
  const Json* map = j.find("expected_channel_map");
  if (!map || map->type != Json::Type::Array) throw std::runtime_error("expected_channel_map");
  c.expected_channel_map.clear();
  for (const Json& item : map->a) c.expected_channel_map.push_back(static_cast<int>(std::llround(item.n)));
  if (auto notes = opt_str(j, "notes")) c.notes = *notes;
  c.model_class = opt_str(j, "model_class");
  c.dis_threshold = opt_num(j, "dis_threshold");
  if (const Json* radii = j.find("supported_radii"); radii && radii->type == Json::Type::Array) {
    for (const Json& item : radii->a) c.supported_radii.push_back(item.n);
  }
  c.source_config = opt_str(j, "source_config");
  c.source_run_dir = opt_str(j, "source_run_dir");
  if (auto ep = opt_num(j, "trained_epochs")) c.trained_epochs = static_cast<int>(std::llround(*ep));
  c.distance_head_trained = opt_bool(j, "distance_head_trained");
  c.distance_valid_threshold = opt_num(j, "distance_valid_threshold");
  c.count_head_trained = opt_bool(j, "count_head_trained");
  c.count_min_confidence = opt_num(j, "count_min_confidence");
  return c;
}

ZooPick resolve_zoo_model(const std::string& zoo_dir, double bubble_radius) {
  namespace fs = std::filesystem;
  if (!fs::is_directory(zoo_dir)) throw std::runtime_error("--zoo-dir does not exist or is not a directory: " + zoo_dir);
  struct Entry {
    std::vector<double> offered;
    fs::path contract;
    fs::path onnx;
  };
  std::vector<Entry> found;
  std::vector<fs::path> contracts;
  for (const auto& ent : fs::directory_iterator(zoo_dir)) {
    if (!ent.is_regular_file()) continue;
    auto name = ent.path().filename().string();
    const std::string suffix = ".runtime.json";
    if (name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
      contracts.push_back(ent.path());
    }
  }
  std::sort(contracts.begin(), contracts.end());
  for (const auto& contract_path : contracts) {
    RuntimeContract c;
    try {
      c = load_contract(contract_path.string());
    } catch (const std::exception& exc) {
      continue;
    }
    std::string stem = contract_path.filename().string();
    stem.resize(stem.size() - std::string(".runtime.json").size());
    fs::path onnx = contract_path.parent_path() / (stem + ".onnx");
    if (!fs::is_regular_file(onnx)) continue;
    std::vector<double> offered;
    if (c.dis_threshold) offered.push_back(*c.dis_threshold);
    for (double r : c.supported_radii) offered.push_back(r);
    if (offered.empty()) continue;
    found.push_back(Entry{std::move(offered), contract_path, onnx});
  }
  std::vector<double> available;
  for (const auto& e : found) available.insert(available.end(), e.offered.begin(), e.offered.end());
  std::sort(available.begin(), available.end());
  available.erase(std::unique(available.begin(), available.end()), available.end());
  if (found.empty()) throw std::runtime_error("No valid zoo entries found in " + zoo_dir);
  const Entry* pick = nullptr;
  for (const auto& e : found) {
    if (e.offered.size() == 1 && std::find(e.offered.begin(), e.offered.end(), bubble_radius) != e.offered.end()) {
      pick = &e;
      break;
    }
  }
  if (!pick) {
    for (const auto& e : found) {
      if (e.offered.size() > 1 && std::find(e.offered.begin(), e.offered.end(), bubble_radius) != e.offered.end()) {
        pick = &e;
        break;
      }
    }
  }
  if (!pick) throw std::runtime_error("--bubble-radius not offered by any zoo entry");
  return ZooPick{pick->onnx.string(), pick->contract.string(), available};
}

}  // namespace soundbubble
