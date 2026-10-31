#include "soundbubble/audio_io.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

#if defined(SOUNDBUBBLE_HAS_PORTAUDIO)
#include <portaudio.h>
#endif

namespace soundbubble {

#if defined(SOUNDBUBBLE_HAS_PORTAUDIO)
namespace {
struct PaInit {
  PaInit() {
    PaError err = Pa_Initialize();
    if (err != paNoError) throw std::runtime_error(Pa_GetErrorText(err));
  }
  ~PaInit() { Pa_Terminate(); }
};
PaInit& pa_init() {
  static PaInit init;
  return init;
}
}  // namespace
#endif

std::vector<DeviceInfo> list_devices() {
  std::vector<DeviceInfo> out;
#if defined(SOUNDBUBBLE_HAS_PORTAUDIO)
  pa_init();
  int n = Pa_GetDeviceCount();
  for (int i = 0; i < n; ++i) {
    const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
    if (!info) continue;
    const PaHostApiInfo* host = Pa_GetHostApiInfo(info->hostApi);
    DeviceInfo d;
    d.index = i;
    d.name = info->name ? info->name : "";
    d.hostapi = host && host->name ? host->name : "";
    d.max_input_channels = info->maxInputChannels;
    d.max_output_channels = info->maxOutputChannels;
    d.default_samplerate = info->defaultSampleRate;
    out.push_back(d);
  }
#else
  throw std::runtime_error("PortAudio is not linked");
#endif
  return out;
}

void print_devices() {
  auto devices = list_devices();
  std::cout << "Audio devices:\n\n";
  for (const auto& d : devices) {
    std::cout << "[" << d.index << "] " << d.name << " | hostapi=" << d.hostapi << " | in=" << d.max_input_channels
              << " out=" << d.max_output_channels << " default_sr=" << d.default_samplerate << "\n";
  }
}

void preflight_devices(const AudioFormat& input, const AudioFormat& output) {
#if defined(SOUNDBUBBLE_HAS_PORTAUDIO)
  pa_init();
  PaStreamParameters in{};
  in.device = input.device.value_or(Pa_GetDefaultInputDevice());
  in.channelCount = input.channels;
  in.sampleFormat = paFloat32;
  in.suggestedLatency = Pa_GetDeviceInfo(in.device) ? Pa_GetDeviceInfo(in.device)->defaultLowInputLatency : 0.02;
  PaStreamParameters out{};
  out.device = output.device.value_or(Pa_GetDefaultOutputDevice());
  out.channelCount = output.channels;
  out.sampleFormat = paFloat32;
  out.suggestedLatency = Pa_GetDeviceInfo(out.device) ? Pa_GetDeviceInfo(out.device)->defaultLowOutputLatency : 0.02;
  PaError err = Pa_IsFormatSupported(&in, nullptr, input.sample_rate);
  if (err != paFormatIsSupported) throw std::runtime_error(std::string("input format: ") + Pa_GetErrorText(err));
  err = Pa_IsFormatSupported(nullptr, &out, output.sample_rate);
  if (err != paFormatIsSupported) throw std::runtime_error(std::string("output format: ") + Pa_GetErrorText(err));
#else
  (void)input;
  (void)output;
  throw std::runtime_error("PortAudio is not linked");
#endif
}

void preflight_duplex(const AudioFormat& input, const AudioFormat& output) {
#if defined(SOUNDBUBBLE_HAS_PORTAUDIO)
  pa_init();
  if (input.sample_rate != output.sample_rate) {
    throw std::runtime_error("full-duplex requires matching input/output sample rates");
  }
  PaStreamParameters in{};
  in.device = input.device.value_or(Pa_GetDefaultInputDevice());
  in.channelCount = input.channels;
  in.sampleFormat = paFloat32;
  const PaDeviceInfo* in_info = Pa_GetDeviceInfo(in.device);
  in.suggestedLatency = in_info ? in_info->defaultLowInputLatency : 0.02;
  PaStreamParameters out{};
  out.device = output.device.value_or(Pa_GetDefaultOutputDevice());
  out.channelCount = output.channels;
  out.sampleFormat = paFloat32;
  const PaDeviceInfo* out_info = Pa_GetDeviceInfo(out.device);
  out.suggestedLatency = out_info ? out_info->defaultLowOutputLatency : 0.02;
  if (in.device != out.device) {
    throw std::runtime_error("full-duplex requires the same PortAudio device index for input and output");
  }
  PaError err = Pa_IsFormatSupported(&in, &out, input.sample_rate);
  if (err != paFormatIsSupported) {
    throw std::runtime_error(std::string("duplex format: ") + Pa_GetErrorText(err));
  }
#else
  (void)input;
  (void)output;
  throw std::runtime_error("PortAudio is not linked");
#endif
}

struct DuplexSession::Impl {
#if defined(SOUNDBUBBLE_HAS_PORTAUDIO)
  PaStream* input = nullptr;
  PaStream* output = nullptr;
  InputCb in_cb;
  OutputCb out_cb;
  StatusCb status_cb;
  int in_ch = 0;
  int out_ch = 0;
  static int in_thunk(const void* input, void*, unsigned long frames, const PaStreamCallbackTimeInfo*, PaStreamCallbackFlags flags, void* user) {
    auto* self = static_cast<Impl*>(user);
    if (flags && self->status_cb) self->status_cb(true, "status");
    if (self->in_cb && input) self->in_cb(static_cast<const float*>(input), static_cast<int>(frames), self->in_ch);
    return paContinue;
  }
  static int out_thunk(const void*, void* output, unsigned long frames, const PaStreamCallbackTimeInfo*, PaStreamCallbackFlags flags, void* user) {
    auto* self = static_cast<Impl*>(user);
    if (flags && self->status_cb) self->status_cb(false, "status");
    if (self->out_cb && output) self->out_cb(static_cast<float*>(output), static_cast<int>(frames), self->out_ch);
    return paContinue;
  }
#endif
};

DuplexSession::DuplexSession(AudioFormat input, AudioFormat output, int input_block, int output_block, const std::string& latency,
                             InputCb in_cb, OutputCb out_cb, StatusCb status_cb)
    : impl_(std::make_unique<Impl>()) {
#if defined(SOUNDBUBBLE_HAS_PORTAUDIO)
  pa_init();
  impl_->in_cb = std::move(in_cb);
  impl_->out_cb = std::move(out_cb);
  impl_->status_cb = std::move(status_cb);
  impl_->in_ch = input.channels;
  impl_->out_ch = output.channels;
  PaStreamParameters in{};
  in.device = input.device.value_or(Pa_GetDefaultInputDevice());
  in.channelCount = input.channels;
  in.sampleFormat = paFloat32;
  const PaDeviceInfo* in_info = Pa_GetDeviceInfo(in.device);
  in.suggestedLatency = in_info ? (latency == "high" ? in_info->defaultHighInputLatency : in_info->defaultLowInputLatency) : 0.02;
  PaStreamParameters out{};
  out.device = output.device.value_or(Pa_GetDefaultOutputDevice());
  out.channelCount = output.channels;
  out.sampleFormat = paFloat32;
  const PaDeviceInfo* out_info = Pa_GetDeviceInfo(out.device);
  double out_default = out_info
                           ? (latency == "high" ? out_info->defaultHighOutputLatency : out_info->defaultLowOutputLatency)
                           : 0.02;
  double out_from_block = std::max(0.004, 2.0 * static_cast<double>(std::max(1, output_block)) / static_cast<double>(output.sample_rate));
  out.suggestedLatency = latency == "high" ? out_default : std::min(out_default, out_from_block);
  PaError err = Pa_OpenStream(&impl_->input, &in, nullptr, input.sample_rate, static_cast<unsigned long>(input_block), paNoFlag, &Impl::in_thunk, impl_.get());
  if (err != paNoError) throw std::runtime_error(Pa_GetErrorText(err));
  err = Pa_OpenStream(&impl_->output, nullptr, &out, output.sample_rate, static_cast<unsigned long>(output_block), paNoFlag, &Impl::out_thunk, impl_.get());
  if (err != paNoError) throw std::runtime_error(Pa_GetErrorText(err));
#else
  (void)input;
  (void)output;
  (void)input_block;
  (void)output_block;
  (void)latency;
  (void)in_cb;
  (void)out_cb;
  (void)status_cb;
  throw std::runtime_error("PortAudio is not linked");
#endif
}

DuplexSession::~DuplexSession() { stop(); }

void DuplexSession::start() {
#if defined(SOUNDBUBBLE_HAS_PORTAUDIO)
  PaError err = Pa_StartStream(impl_->output);
  if (err != paNoError) throw std::runtime_error(Pa_GetErrorText(err));
  err = Pa_StartStream(impl_->input);
  if (err != paNoError) throw std::runtime_error(Pa_GetErrorText(err));
#endif
}

void DuplexSession::stop() {
#if defined(SOUNDBUBBLE_HAS_PORTAUDIO)
  if (!impl_) return;
  if (impl_->input) {
    Pa_StopStream(impl_->input);
    Pa_CloseStream(impl_->input);
    impl_->input = nullptr;
  }
  if (impl_->output) {
    Pa_StopStream(impl_->output);
    Pa_CloseStream(impl_->output);
    impl_->output = nullptr;
  }
#endif
}

bool DuplexSession::active() const {
#if defined(SOUNDBUBBLE_HAS_PORTAUDIO)
  if (!impl_ || !impl_->input || !impl_->output) return false;
  return Pa_IsStreamActive(impl_->input) == 1 && Pa_IsStreamActive(impl_->output) == 1;
#else
  return false;
#endif
}

struct DuplexSyncSession::Impl {
#if defined(SOUNDBUBBLE_HAS_PORTAUDIO)
  PaStream* stream = nullptr;
  DuplexCb cb;
  StatusCb status_cb;
  int in_ch = 0;
  int out_ch = 0;
  static int thunk(const void* input, void* output, unsigned long frames, const PaStreamCallbackTimeInfo*,
                   PaStreamCallbackFlags flags, void* user) {
    auto* self = static_cast<Impl*>(user);
    if (flags && self->status_cb) self->status_cb("status");
    if (self->cb) {
      self->cb(static_cast<const float*>(input), static_cast<float*>(output), static_cast<int>(frames), self->in_ch,
               self->out_ch);
    } else if (output) {
      std::fill(static_cast<float*>(output),
                static_cast<float*>(output) + static_cast<size_t>(frames) * static_cast<size_t>(self->out_ch), 0.f);
    }
    return paContinue;
  }
#endif
};

DuplexSyncSession::DuplexSyncSession(AudioFormat input, AudioFormat output, int frames_per_buffer, const std::string& latency,
                                     DuplexCb cb, StatusCb status_cb)
    : impl_(std::make_unique<Impl>()) {
#if defined(SOUNDBUBBLE_HAS_PORTAUDIO)
  pa_init();
  if (input.sample_rate != output.sample_rate) {
    throw std::runtime_error("DuplexSyncSession requires matching sample rates");
  }
  impl_->cb = std::move(cb);
  impl_->status_cb = std::move(status_cb);
  impl_->in_ch = input.channels;
  impl_->out_ch = output.channels;

  PaStreamParameters in{};
  in.device = input.device.value_or(Pa_GetDefaultInputDevice());
  in.channelCount = input.channels;
  in.sampleFormat = paFloat32;
  const PaDeviceInfo* in_info = Pa_GetDeviceInfo(in.device);
  double in_default = in_info ? (latency == "high" ? in_info->defaultHighInputLatency : in_info->defaultLowInputLatency) : 0.02;
  double from_block = std::max(0.002, static_cast<double>(std::max(1, frames_per_buffer)) / static_cast<double>(input.sample_rate));
  in.suggestedLatency = latency == "high" ? in_default : std::min(in_default, from_block);

  PaStreamParameters out{};
  out.device = output.device.value_or(Pa_GetDefaultOutputDevice());
  out.channelCount = output.channels;
  out.sampleFormat = paFloat32;
  const PaDeviceInfo* out_info = Pa_GetDeviceInfo(out.device);
  double out_default = out_info ? (latency == "high" ? out_info->defaultHighOutputLatency : out_info->defaultLowOutputLatency) : 0.02;
  out.suggestedLatency = latency == "high" ? out_default : std::min(out_default, from_block);

  if (in.device != out.device) {
    throw std::runtime_error("DuplexSyncSession requires the same device for input and output");
  }

  PaError err = Pa_OpenStream(&impl_->stream, &in, &out, input.sample_rate,
                              static_cast<unsigned long>(std::max(1, frames_per_buffer)), paNoFlag, &Impl::thunk,
                              impl_.get());
  if (err != paNoError) throw std::runtime_error(Pa_GetErrorText(err));
#else
  (void)input;
  (void)output;
  (void)frames_per_buffer;
  (void)latency;
  (void)cb;
  (void)status_cb;
  throw std::runtime_error("PortAudio is not linked");
#endif
}

DuplexSyncSession::~DuplexSyncSession() { stop(); }

void DuplexSyncSession::start() {
#if defined(SOUNDBUBBLE_HAS_PORTAUDIO)
  PaError err = Pa_StartStream(impl_->stream);
  if (err != paNoError) throw std::runtime_error(Pa_GetErrorText(err));
#endif
}

void DuplexSyncSession::stop() {
#if defined(SOUNDBUBBLE_HAS_PORTAUDIO)
  if (!impl_ || !impl_->stream) return;
  Pa_StopStream(impl_->stream);
  Pa_CloseStream(impl_->stream);
  impl_->stream = nullptr;
#endif
}

bool DuplexSyncSession::active() const {
#if defined(SOUNDBUBBLE_HAS_PORTAUDIO)
  if (!impl_ || !impl_->stream) return false;
  return Pa_IsStreamActive(impl_->stream) == 1;
#else
  return false;
#endif
}

}  // namespace soundbubble
