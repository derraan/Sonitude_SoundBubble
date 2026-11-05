#include "soundbubble/mixctl.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <iostream>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#endif

namespace soundbubble {
namespace {

#if defined(_WIN32)
std::string normalize_win_port(const std::string& port) {
  if (port.size() >= 4 && (port[0] == '\\' || port[0] == '/')) return port;
  if (port.size() >= 4) {
    std::string upper = port;
    for (char& c : upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    if (upper.rfind("COM", 0) == 0) {
      try {
        int n = std::stoi(upper.substr(3));
        if (n >= 10) return "\\\\.\\" + port;
      } catch (...) {
      }
    }
  }
  return port;
}
#endif

}  // namespace

MixLine parse_mix_line(std::string_view line) {
  MixLine out;
  while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || std::isspace(static_cast<unsigned char>(line.back())))) {
    line.remove_suffix(1);
  }
  size_t i = 0;
  while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) ++i;
  if (line.size() < i + 3 || line.substr(i, 3) != "MIX") return out;
  i += 3;
  if (i < line.size() && !std::isspace(static_cast<unsigned char>(line[i]))) return out;
  bool have_wet = false;
  bool have_dry = false;
  auto parse_field = [&](const char* key, float& dst, bool& flag) {
    std::string token(key);
    auto pos = line.find(token, i);
    if (pos == std::string_view::npos) return;
    pos += token.size();
    if (pos >= line.size()) return;
    size_t end = pos;
    while (end < line.size() && (std::isdigit(static_cast<unsigned char>(line[end])) || line[end] == '.' || line[end] == '-' || line[end] == '+')) {
      ++end;
    }
    if (end == pos) return;
    try {
      dst = std::stof(std::string(line.substr(pos, end - pos)));
      dst = std::clamp(dst, 0.f, 1.f);
      flag = true;
    } catch (...) {
      flag = false;
    }
  };
  parse_field("wet=", out.wet, have_wet);
  parse_field("dry=", out.dry, have_dry);
  out.ok = have_wet && have_dry;
  return out;
}

TapUartCmd parse_tap_uart_line(std::string_view line) {
  while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || std::isspace(static_cast<unsigned char>(line.back())))) {
    line.remove_suffix(1);
  }
  size_t i = 0;
  while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) ++i;
  if (line.size() < i + 3) return TapUartCmd::None;
  auto head = line.substr(i, 3);
  if (!(head[0] == 'T' || head[0] == 't') || !(head[1] == 'A' || head[1] == 'a') || !(head[2] == 'P' || head[2] == 'p')) {
    return TapUartCmd::None;
  }
  i += 3;
  while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) ++i;
  if (i >= line.size()) return TapUartCmd::Toggle;
  std::string arg;
  while (i < line.size() && !std::isspace(static_cast<unsigned char>(line[i]))) {
    arg.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(line[i++]))));
  }
  if (arg.empty() || arg == "toggle") return TapUartCmd::Toggle;
  if (arg == "on" || arg == "arm") return TapUartCmd::On;
  if (arg == "off" || arg == "disarm") return TapUartCmd::Off;
  if (arg == "flush") return TapUartCmd::Flush;
  return TapUartCmd::None;
}

MixControl::MixControl(std::string port, int baud, float initial_dry)
    : port_(std::move(port)), baud_(baud), dry_(std::clamp(initial_dry, 0.f, 1.f)) {}

MixControl::~MixControl() { stop(); }

void MixControl::ingest_for_test(std::string_view bytes) { consume(bytes); }

TapUartCmd MixControl::take_tap_cmd() {
  int expected = tap_cmd_.exchange(static_cast<int>(TapUartCmd::None));
  return static_cast<TapUartCmd>(expected);
}

void MixControl::consume(std::string_view chunk) {
  pending_.append(chunk.data(), chunk.size());
  size_t nl;
  while ((nl = pending_.find('\n')) != std::string::npos) {
    // Own the line: substr() returns a temporary, so a string_view of it dangles
    // (ASan heap-use-after-free in parse_mix_line).
    const std::string line = pending_.substr(0, nl);
    pending_.erase(0, nl + 1);
    auto parsed = parse_mix_line(line);
    if (parsed.ok) {
      dry_.store(parsed.dry);
      continue;
    }
    auto tap = parse_tap_line(line);
    if (tap != TapUartCmd::None) tap_cmd_.store(static_cast<int>(tap));
  }
}

void MixControl::start() {
  if (started_ || port_.empty()) return;
  started_ = true;
  stop_ = false;
  thread_ = std::thread([this] { run(); });
}

void MixControl::stop() {
  stop_ = true;
  if (thread_.joinable()) thread_.join();
  started_ = false;
}

void MixControl::run() {
  int backoff_ms = 200;
  bool logged_open = false;
  bool logged_fail = false;
  while (!stop_.load()) {
#if defined(_WIN32)
    std::string path = normalize_win_port(port_);
    HANDLE h = CreateFileA(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
      if (!logged_fail) {
        std::cerr << "[mix] open failed " << path << "\n";
        logged_fail = true;
      }
      logged_open = false;
      std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
      backoff_ms = std::min(5000, backoff_ms * 2);
      continue;
    }
    DCB dcb{};
    dcb.DCBlength = sizeof(dcb);
    GetCommState(h, &dcb);
    dcb.BaudRate = static_cast<DWORD>(baud_);
    dcb.ByteSize = 8;
    dcb.Parity = NOPARITY;
    dcb.StopBits = ONESTOPBIT;
    SetCommState(h, &dcb);
    COMMTIMEOUTS timeouts{};
    timeouts.ReadIntervalTimeout = 50;
    timeouts.ReadTotalTimeoutConstant = 200;
    timeouts.ReadTotalTimeoutMultiplier = 0;
    SetCommTimeouts(h, &timeouts);
    if (!logged_open) {
      std::cerr << "[mix] open " << path << " baud=" << baud_ << "\n";
      logged_open = true;
    }
    logged_fail = false;
    backoff_ms = 200;
    char buf[256];
    while (!stop_.load()) {
      DWORD n = 0;
      if (!ReadFile(h, buf, sizeof(buf), &n, nullptr)) break;
      if (n > 0) consume(std::string_view(buf, n));
    }
    CloseHandle(h);
#else
    int fd = ::open(port_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0) {
      if (!logged_fail) {
        std::cerr << "[mix] open failed " << port_ << "\n";
        logged_fail = true;
      }
      logged_open = false;
      std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
      backoff_ms = std::min(5000, backoff_ms * 2);
      continue;
    }
    termios tio{};
    if (tcgetattr(fd, &tio) == 0) {
      cfmakeraw(&tio);
      speed_t speed = B115200;
      if (baud_ == 9600) speed = B9600;
      else if (baud_ == 57600) speed = B57600;
      else if (baud_ == 115200) speed = B115200;
      cfsetispeed(&tio, speed);
      cfsetospeed(&tio, speed);
      tio.c_cflag |= (CLOCAL | CREAD);
      tcsetattr(fd, TCSANOW, &tio);
    }
    if (!logged_open) {
      std::cerr << "[mix] open " << port_ << " baud=" << baud_ << "\n";
      logged_open = true;
    }
    logged_fail = false;
    backoff_ms = 200;
    char buf[256];
    while (!stop_.load()) {
      ssize_t n = ::read(fd, buf, sizeof(buf));
      if (n > 0) consume(std::string_view(buf, static_cast<size_t>(n)));
      else if (n == 0 || (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      } else {
        break;
      }
    }
    ::close(fd);
#endif
  }
}

}  // namespace soundbubble
