#include "soundbubble/pipeline.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#if defined(SOUNDBUBBLE_HAS_ORT)
#if defined(_WIN32) && !defined(_MSC_VER)
#ifndef _In_
#define _In_
#define _In_z_
#define _In_opt_
#define _In_opt_z_
#define _Out_
#define _Outptr_
#define _Out_opt_
#define _Inout_
#define _Inout_opt_
#define _Frees_ptr_opt_
#define _Ret_maybenull_
#define _Ret_notnull_
#define _Check_return_
#define _Outptr_result_maybenull_
#define _In_reads_(X)
#define _Inout_updates_(X)
#define _Out_writes_(X)
#define _Inout_updates_all_(X)
#define _Out_writes_bytes_all_(X)
#define _Out_writes_all_(X)
#define _Success_(X)
#define _Outptr_result_buffer_maybenull_(X)
#endif
#endif
#include <onnxruntime_cxx_api.h>
#endif

namespace soundbubble {
namespace {

#if defined(SOUNDBUBBLE_HAS_ORT)
const std::unordered_map<double, std::array<float, 3>> kDisEmbed = {
    {1.0, {0.f, 0.f, 1.f}},
    {1.5, {0.f, 1.f, 0.f}},
    {2.0, {1.f, 0.f, 0.f}},
};

class OnnxInferencer final : public Inferencer {
 public:
  OnnxInferencer(const std::string& path, int intra, int inter) : env_(ORT_LOGGING_LEVEL_WARNING, "soundbubble") {
    Ort::SessionOptions opt;
    opt.SetIntraOpNumThreads(intra);
    opt.SetInterOpNumThreads(inter);
    opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#if defined(_WIN32)
    std::wstring wpath(path.begin(), path.end());
    session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), opt);
#else
    session_ = std::make_unique<Ort::Session>(env_, path.c_str(), opt);
#endif
    Ort::AllocatorWithDefaultOptions alloc;
    size_t n_in = session_->GetInputCount();
    for (size_t i = 0; i < n_in; ++i) {
      auto name = session_->GetInputNameAllocated(i, alloc);
      input_names_.emplace_back(name.get());
    }
    size_t n_out = session_->GetOutputCount();
    for (size_t i = 0; i < n_out; ++i) {
      auto name = session_->GetOutputNameAllocated(i, alloc);
      output_names_.emplace_back(name.get());
    }
    bool has_mix = false;
    bool has_out = false;
    for (const auto& n : input_names_) {
      if (n == "mixture") has_mix = true;
      if (n == "dis_embed") accepts_dis_ = true;
      if (n.rfind("state_", 0) == 0) state_inputs_.push_back(n);
    }
    for (const auto& n : output_names_) {
      if (n == "output") has_out = true;
      if (n.rfind("next_state_", 0) == 0) state_outputs_.push_back(n);
    }
    if (!has_mix || !has_out) throw std::runtime_error("ONNX graph must expose mixture/output");
    std::sort(state_inputs_.begin(), state_inputs_.end(), [](const std::string& a, const std::string& b) {
      return std::stoi(a.substr(6)) < std::stoi(b.substr(6));
    });
    std::sort(state_outputs_.begin(), state_outputs_.end(), [](const std::string& a, const std::string& b) {
      return std::stoi(a.substr(11)) < std::stoi(b.substr(11));
    });
    if (state_inputs_.size() != state_outputs_.size()) throw std::runtime_error("state input/output count mismatch");
    for (size_t i = 0; i < n_in; ++i) {
      if (input_names_[i].rfind("state_", 0) != 0) continue;
      // TypeInfo owns the tensor-info pointer. Keep it alive across GetShape.
      auto type_info = session_->GetInputTypeInfo(i);
      auto info = type_info.GetTensorTypeAndShapeInfo();
      auto shape = info.GetShape();
      std::vector<int64_t> dims;
      size_t count = 1;
      for (auto d : shape) {
        int64_t v = d > 0 ? d : 1;
        dims.push_back(v);
        count *= static_cast<size_t>(v);
      }
      state_[input_names_[i]] = std::vector<float>(count, 0.f);
      state_dims_[input_names_[i]] = dims;
    }
    auto find_out = [&](const char* name) -> int {
      auto it = std::find(output_names_.begin(), output_names_.end(), std::string(name));
      if (it == output_names_.end()) return -1;
      return static_cast<int>(it - output_names_.begin());
    };
    distance_m_idx_ = find_out("distance_m");
    distance_valid_idx_ = find_out("distance_valid");
    count_probs_idx_ = find_out("count_probs");
    has_distance_ = distance_m_idx_ >= 0 && distance_valid_idx_ >= 0;
    has_count_ = count_probs_idx_ >= 0;
    held_names_.reserve(input_names_.size());
  }

  bool accepts_dis_embed() const override { return accepts_dis_; }

  void set_radius(double radius) override {
    if (!accepts_dis_) return;
    auto it = kDisEmbed.find(radius);
    if (it == kDisEmbed.end()) throw std::invalid_argument("bubble radius not in training set");
    dis_ = it->second;
    have_dis_ = true;
  }

  void set_distance_runtime_config(bool trained, float valid_threshold, bool gate_on_count) override {
    distance_trained_ = trained;
    distance_valid_threshold_ = valid_threshold;
    gate_distance_on_count_ = gate_on_count;
  }

  void set_count_runtime_config(bool trained, float min_confidence) override {
    count_trained_ = trained;
    count_min_confidence_ = min_confidence;
  }

  void reset() override {
    for (auto& kv : state_) std::fill(kv.second.begin(), kv.second.end(), 0.f);
  }

  Hop run(const float* frame_nct, int channels, int frame_len) override {
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> mix_shape = {1, channels, frame_len};
    std::vector<float> mix(frame_nct, frame_nct + static_cast<size_t>(channels * frame_len));
    std::vector<Ort::Value> feeds;
    std::vector<const char*> feed_names;
    auto push_tensor = [&](const std::string& name, float* data, const std::vector<int64_t>& shape) {
      feed_names.push_back(name.c_str());
      feeds.push_back(Ort::Value::CreateTensor<float>(mem, data, static_cast<size_t>(std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>())), shape.data(), shape.size()));
    };
    held_names_.clear();
    held_names_.push_back("mixture");
    push_tensor(held_names_.back(), mix.data(), mix_shape);
    std::array<float, 3> dis_buf{};
    if (accepts_dis_) {
      if (!have_dis_) throw std::runtime_error("dis_embed radius was not set");
      dis_buf = dis_;
      held_names_.push_back("dis_embed");
      push_tensor(held_names_.back(), dis_buf.data(), {1, 3});
    }
    for (const auto& name : state_inputs_) {
      held_names_.push_back(name);
      push_tensor(held_names_.back(), state_[name].data(), state_dims_[name]);
    }
    std::vector<const char*> out_names;
    out_names.reserve(output_names_.size());
    for (const auto& n : output_names_) out_names.push_back(n.c_str());
    auto t0 = std::chrono::steady_clock::now();
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, feed_names.data(), feeds.data(), feeds.size(), out_names.data(), out_names.size());
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    for (size_t i = 0; i < state_inputs_.size(); ++i) {
      auto it = std::find(output_names_.begin(), output_names_.end(), state_outputs_[i]);
      if (it == output_names_.end()) continue;
      size_t idx = static_cast<size_t>(it - output_names_.begin());
      float* src = outputs[idx].GetTensorMutableData<float>();
      auto& dst = state_[state_inputs_[i]];
      std::copy(src, src + dst.size(), dst.begin());
    }
    auto out_it = std::find(output_names_.begin(), output_names_.end(), std::string("output"));
    size_t out_idx = static_cast<size_t>(out_it - output_names_.begin());
    float* y = outputs[out_idx].GetTensorMutableData<float>();
    auto info = outputs[out_idx].GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    int hop = 0;
    if (shape.size() == 3) hop = static_cast<int>(shape.back());
    else hop = static_cast<int>(info.GetElementCount());
    Hop hop_out;
    hop_out.infer_ms = ms;
    hop_out.samples.assign(y, y + hop);
    decode_heads(outputs, hop_out);
    return hop_out;
  }

 private:
  void decode_heads(std::vector<Ort::Value>& outputs, Hop& hop_out) {
    hop_out.distance_status = "unavailable";
    hop_out.distance_m.reset();
    hop_out.count_status = "unavailable";
    hop_out.speaker_count.reset();
    hop_out.speaker_distances.clear();

    std::vector<std::pair<float, float>> distance_slots;
    if (has_distance_ && distance_trained_) {
      auto& d_ten = outputs[static_cast<size_t>(distance_m_idx_)];
      auto& v_ten = outputs[static_cast<size_t>(distance_valid_idx_)];
      auto d_info = d_ten.GetTensorTypeAndShapeInfo();
      auto v_info = v_ten.GetTensorTypeAndShapeInfo();
      auto d_shape = d_info.GetShape();
      float* d_data = d_ten.GetTensorMutableData<float>();
      float* v_data = v_ten.GetTensorMutableData<float>();
      if (d_shape.size() == 3) {
        int last = static_cast<int>(d_shape[1]);
        int slots = static_cast<int>(d_shape[2]);
        int base = std::max(0, last - 1) * slots;
        for (int i = 0; i < slots; ++i) distance_slots.emplace_back(d_data[base + i], v_data[base + i]);
      } else if (d_shape.size() == 2) {
        int last = static_cast<int>(d_shape[1]);
        int idx = std::max(0, last - 1);
        distance_slots.emplace_back(d_data[idx], v_data[idx]);
      } else {
        size_t count = d_info.GetElementCount();
        size_t vcount = v_info.GetElementCount();
        if (count > 0 && vcount > 0) distance_slots.emplace_back(d_data[count - 1], v_data[vcount - 1]);
      }
      bool has_valid = false;
      for (const auto& slot : distance_slots) {
        float d_val = slot.first;
        float v_val = slot.second;
        if (std::isfinite(d_val) && std::isfinite(v_val) && v_val >= distance_valid_threshold_ && d_val > 0.f) {
          hop_out.distance_m = d_val;
          hop_out.distance_status = "valid";
          has_valid = true;
          break;
        }
      }
      if (!has_valid) hop_out.distance_status = "inactive";
    }

    if (has_count_ && count_trained_) {
      auto& c_ten = outputs[static_cast<size_t>(count_probs_idx_)];
      auto c_info = c_ten.GetTensorTypeAndShapeInfo();
      auto c_shape = c_info.GetShape();
      float* c_data = c_ten.GetTensorMutableData<float>();
      float probs[3] = {0.f, 0.f, 0.f};
      if (c_shape.size() == 3 && c_shape[2] >= 3) {
        int last = static_cast<int>(c_shape[1]);
        int base = std::max(0, last - 1) * 3;
        probs[0] = c_data[base];
        probs[1] = c_data[base + 1];
        probs[2] = c_data[base + 2];
      } else {
        size_t count = c_info.GetElementCount();
        if (count >= 3) {
          probs[0] = c_data[count - 3];
          probs[1] = c_data[count - 2];
          probs[2] = c_data[count - 1];
        }
      }
      int pred_idx = 0;
      float pred_conf = probs[0];
      for (int i = 1; i < 3; ++i) {
        if (probs[i] > pred_conf) {
          pred_conf = probs[i];
          pred_idx = i;
        }
      }
      if (pred_conf >= count_min_confidence_) {
        hop_out.count_status = "valid";
        static const char* labels[] = {"0", "1", "2+"};
        hop_out.speaker_count = labels[pred_idx];
      } else {
        hop_out.count_status = "low_confidence";
      }
    }

    if (gate_distance_on_count_ && hop_out.count_status == "valid" && hop_out.speaker_count && *hop_out.speaker_count != "1") {
      hop_out.distance_status = "inactive";
      hop_out.distance_m.reset();
    }

    if (hop_out.count_status == "valid" && hop_out.speaker_count) {
      int n_speakers = 0;
      if (*hop_out.speaker_count == "1") n_speakers = 1;
      else if (*hop_out.speaker_count == "2+") n_speakers = 2;
      for (int idx = 0; idx < n_speakers; ++idx) {
        SpeakerDistance row;
        row.speaker_index = idx + 1;
        row.status = "unavailable";
        if (static_cast<int>(distance_slots.size()) > idx) {
          float d_val = distance_slots[static_cast<size_t>(idx)].first;
          float v_val = distance_slots[static_cast<size_t>(idx)].second;
          if (std::isfinite(d_val) && std::isfinite(v_val) && v_val >= distance_valid_threshold_ && d_val > 0.f) {
            row.status = "valid";
            row.distance_m = d_val;
          } else {
            row.status = "inactive";
          }
        } else if (idx == 0 && (hop_out.distance_status == "valid" || hop_out.distance_status == "inactive")) {
          row.status = hop_out.distance_status;
          row.distance_m = hop_out.distance_m;
        }
        hop_out.speaker_distances.push_back(row);
      }
    }
  }

  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<std::string> state_inputs_;
  std::vector<std::string> state_outputs_;
  std::vector<std::string> held_names_;
  std::unordered_map<std::string, std::vector<float>> state_;
  std::unordered_map<std::string, std::vector<int64_t>> state_dims_;
  bool accepts_dis_ = false;
  bool have_dis_ = false;
  std::array<float, 3> dis_{};
  int distance_m_idx_ = -1;
  int distance_valid_idx_ = -1;
  int count_probs_idx_ = -1;
  bool has_distance_ = false;
  bool has_count_ = false;
  bool distance_trained_ = false;
  float distance_valid_threshold_ = 0.5f;
  bool gate_distance_on_count_ = false;
  bool count_trained_ = false;
  float count_min_confidence_ = 0.6f;
};
#endif

}  // namespace

std::unique_ptr<Inferencer> make_onnx_inferencer(const std::string& model_path, int intra, int inter, const RuntimeContract* contract,
                                                 std::optional<double> radius) {
#if defined(SOUNDBUBBLE_HAS_ORT)
  auto inf = std::make_unique<OnnxInferencer>(model_path, intra, inter);
  if (inf->accepts_dis_embed()) {
    std::optional<double> chosen = radius;
    if (!chosen && contract && contract->dis_threshold) chosen = contract->dis_threshold;
    if (!chosen && contract && !contract->supported_radii.empty()) chosen = contract->supported_radii.front();
    if (!chosen) throw std::runtime_error("distance-aware ONNX requires --bubble-radius");
    inf->set_radius(*chosen);
  }
  if (contract) {
    bool dist_trained = contract->distance_head_trained.value_or(false);
    float dist_th = static_cast<float>(contract->distance_valid_threshold.value_or(0.5));
    inf->set_distance_runtime_config(dist_trained, dist_th, false);
    bool count_trained = contract->count_head_trained.value_or(false);
    float count_th = static_cast<float>(contract->count_min_confidence.value_or(0.6));
    inf->set_count_runtime_config(count_trained, count_th);
  }
  return inf;
#else
  (void)model_path;
  (void)intra;
  (void)inter;
  (void)contract;
  (void)radius;
  throw std::runtime_error("This build has no ONNX Runtime. Reconfigure with SOUNDBUBBLE_WITH_ORT=ON.");
#endif
}

}  // namespace soundbubble
