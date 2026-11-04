#include "soundbubble/pipeline.hpp"

#include <atomic>
#include <csignal>
#include <iostream>
#include <string>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {
std::atomic<soundbubble::Engine*> g_engine{nullptr};
std::atomic<bool> g_stdin_done{false};

void on_signal(int sig) {
  auto* engine = g_engine.load();
  if (!engine) return;
#if !defined(_WIN32)
  if (sig == SIGUSR1) {
    engine->toggle_taps();
    return;
  }
  if (sig == SIGUSR2) {
    engine->request_tap_flush();
    return;
  }
#endif
#if defined(_WIN32) && defined(SIGBREAK)
  if (sig == SIGBREAK) {
    engine->request_stop("signal:SIGBREAK");
    return;
  }
#endif
  const char* name = sig == SIGINT ? "SIGINT" : "SIGTERM";
  engine->request_stop(std::string("signal:") + name);
}

#if defined(_WIN32)
BOOL WINAPI on_console_ctrl(DWORD type) {
  auto* engine = g_engine.load();
  if (!engine) return FALSE;
  if (type == CTRL_C_EVENT) {
    engine->request_stop("signal:CTRL_C");
    return TRUE;
  }
  if (type == CTRL_BREAK_EVENT) {
    engine->request_stop("signal:CTRL_BREAK");
    return TRUE;
  }
  if (type == CTRL_CLOSE_EVENT) {
    engine->request_stop("signal:CTRL_CLOSE");
    return TRUE;
  }
  return FALSE;
}
#endif

void stdin_control_loop() {
  std::string line;
  while (std::getline(std::cin, line)) {
    auto* engine = g_engine.load();
    if (!engine) break;
    try {
      engine->handle_control_line(line);
    } catch (const std::exception& exc) {
      std::cerr << "[ctrl] " << exc.what() << "\n";
    }
    if (engine->stopped()) break;
  }
  g_stdin_done.store(true);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    auto cfg = soundbubble::parse_args(argc, argv);
    bool offline = !cfg.input_wav.empty();
    bool list_devices = cfg.list_devices;
    bool preflight = cfg.preflight_only;
    soundbubble::Engine engine(std::move(cfg));
    g_engine = &engine;
    std::signal(SIGINT, on_signal);
#if defined(_WIN32)
    SetConsoleCtrlHandler(on_console_ctrl, TRUE);
#if defined(SIGBREAK)
    std::signal(SIGBREAK, on_signal);
#endif
#else
    std::signal(SIGTERM, on_signal);
    std::signal(SIGUSR1, on_signal);
    std::signal(SIGUSR2, on_signal);
#endif

    std::thread stdin_thread;
    const bool live = !list_devices && !preflight && !offline;
    if (live) {
      stdin_thread = std::thread(stdin_control_loop);
    }

    int code = 0;
    if (list_devices || preflight || !offline) code = engine.run_live();
    else code = engine.run_offline_wav();

    g_engine = nullptr;
    if (stdin_thread.joinable()) {
      // Closing stdin is the parent's job; if still open, detach so we do not hang.
      if (g_stdin_done.load()) {
        stdin_thread.join();
      } else {
        stdin_thread.detach();
      }
    }
    return code;
  } catch (const std::exception& exc) {
    std::cerr << "[fatal] " << exc.what() << "\n";
    return 1;
  }
}
