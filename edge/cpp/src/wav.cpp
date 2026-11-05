#include "soundbubble/wav.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace soundbubble {
namespace {

std::uint32_t read_u32(std::istream& in) {
  unsigned char b[4];
  in.read(reinterpret_cast<char*>(b), 4);
  return static_cast<std::uint32_t>(b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24));
}
std::uint16_t read_u16(std::istream& in) {
  unsigned char b[2];
  in.read(reinterpret_cast<char*>(b), 2);
  return static_cast<std::uint16_t>(b[0] | (b[1] << 8));
}
void write_u32(std::ostream& out, std::uint32_t v) {
  unsigned char b[4] = {static_cast<unsigned char>(v), static_cast<unsigned char>(v >> 8), static_cast<unsigned char>(v >> 16),
                        static_cast<unsigned char>(v >> 24)};
  out.write(reinterpret_cast<char*>(b), 4);
}
void write_u16(std::ostream& out, std::uint16_t v) {
  unsigned char b[2] = {static_cast<unsigned char>(v), static_cast<unsigned char>(v >> 8)};
  out.write(reinterpret_cast<char*>(b), 2);
}

}  // namespace

WavData read_wav(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open wav " + path);
  char riff[4];
  in.read(riff, 4);
  if (std::string(riff, 4) != "RIFF") throw std::runtime_error("not a RIFF wav");
  read_u32(in);
  char wave[4];
  in.read(wave, 4);
  if (std::string(wave, 4) != "WAVE") throw std::runtime_error("not a WAVE");
  int channels = 0;
  int rate = 0;
  int bits = 0;
  std::vector<char> pcm;
  while (in && !in.eof()) {
    char id[4];
    if (!in.read(id, 4)) break;
    std::uint32_t size = read_u32(in);
    std::string tag(id, 4);
    if (tag == "fmt ") {
      int format = read_u16(in);
      channels = read_u16(in);
      rate = static_cast<int>(read_u32(in));
      read_u32(in);
      read_u16(in);
      bits = read_u16(in);
      if (size > 16) in.seekg(size - 16, std::ios::cur);
      if (format != 1 && format != 3) throw std::runtime_error("wav must be PCM or float");
    } else if (tag == "data") {
      pcm.resize(size);
      in.read(pcm.data(), size);
      break;
    } else {
      in.seekg(size, std::ios::cur);
    }
  }
  if (channels <= 0 || rate <= 0 || pcm.empty()) throw std::runtime_error("incomplete wav");
  WavData wav;
  wav.sample_rate = rate;
  wav.channels = channels;
  size_t frames = 0;
  if (bits == 16) {
    frames = pcm.size() / (static_cast<size_t>(channels) * 2);
    wav.interleaved.resize(frames * static_cast<size_t>(channels));
    for (size_t i = 0; i < wav.interleaved.size(); ++i) {
      std::int16_t s;
      std::memcpy(&s, pcm.data() + i * 2, 2);
      wav.interleaved[i] = static_cast<float>(s) / 32768.f;
    }
  } else if (bits == 32) {
    frames = pcm.size() / (static_cast<size_t>(channels) * 4);
    wav.interleaved.resize(frames * static_cast<size_t>(channels));
    std::memcpy(wav.interleaved.data(), pcm.data(), wav.interleaved.size() * sizeof(float));
  } else {
    throw std::runtime_error("unsupported wav bit depth");
  }
  return wav;
}

void write_wav(const std::string& path, int sample_rate, int channels, const std::vector<float>& interleaved) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot write wav " + path);
  std::uint32_t data_bytes = static_cast<std::uint32_t>(interleaved.size() * sizeof(float));
  out.write("RIFF", 4);
  write_u32(out, 36 + data_bytes);
  out.write("WAVE", 4);
  out.write("fmt ", 4);
  write_u32(out, 16);
  write_u16(out, 3);
  write_u16(out, static_cast<std::uint16_t>(channels));
  write_u32(out, static_cast<std::uint32_t>(sample_rate));
  write_u32(out, static_cast<std::uint32_t>(sample_rate * channels * 4));
  write_u16(out, static_cast<std::uint16_t>(channels * 4));
  write_u16(out, 32);
  out.write("data", 4);
  write_u32(out, data_bytes);
  out.write(reinterpret_cast<const char*>(interleaved.data()), data_bytes);
}

void write_wav_atomic(const std::string& path, int sample_rate, int channels, const std::vector<float>& interleaved) {
  namespace fs = std::filesystem;
  const fs::path dest(path);
  const fs::path partial = dest.string() + ".partial";
  write_wav(partial.string(), sample_rate, channels, interleaved);
  std::error_code ec;
  fs::remove(dest, ec);
  fs::rename(partial, dest, ec);
  if (ec) {
    // Fallback: copy then remove partial if rename fails (e.g. cross-device).
    fs::copy_file(partial, dest, fs::copy_options::overwrite_existing, ec);
    fs::remove(partial, ec);
    if (!fs::exists(dest)) throw std::runtime_error("atomic wav rename failed for " + path);
  }
}

}  // namespace soundbubble
