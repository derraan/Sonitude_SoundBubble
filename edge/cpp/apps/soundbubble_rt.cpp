#include "soundbubble/pipeline.hpp"

#include <atomic>
#include <csignal>
#include <iostream>

namespace {
std::atomic<soundbubble::Engine*> g_engine{nullptr};

void on_signal(int sig) {
  auto* engine = g_engine.load();
  if (!engine) return;
  const char* name = sig == SIGINT ? "SIGINT" : "SIGTERM";
  engine->request_stop(std::string("signal:") + name);
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
#if !defined(_WIN32)
    std::signal(SIGTERM, on_signal);
#endif
    int code = 0;
    if (list_devices || preflight || !offline) code = engine.run_live();
    else code = engine.run_offline_wav();
    g_engine = nullptr;
    return code;
  } catch (const std::exception& exc) {
    std::cerr << "[fatal] " << exc.what() << "\n";
    return 1;
  }
}
