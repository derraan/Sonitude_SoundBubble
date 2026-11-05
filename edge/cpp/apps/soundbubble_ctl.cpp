#include "soundbubble/controller.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  std::string host = "0.0.0.0";
  int port = 8000;
  bool allow_no_token = false;
  std::string repo = ".";
  std::string rt = "soundbubble-rt";
  std::string python = "python3";
  if (const char* env = std::getenv("SOUNDBUBBLE_ROOT")) repo = env;
  if (const char* env = std::getenv("SOUNDBUBBLE_RT_BIN")) rt = env;
  if (const char* env = std::getenv("SOUNDBUBBLE_PYTHON")) python = env;
  if (const char* env = std::getenv("SOUND_BUBBLE_CONTROLLER_HOST")) host = env;
  if (const char* env = std::getenv("SOUND_BUBBLE_CONTROLLER_PORT")) port = std::atoi(env);
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&]() {
      if (i + 1 >= argc) throw std::runtime_error("missing value");
      return std::string(argv[++i]);
    };
    if (a == "--host") host = need();
    else if (a == "--port") port = std::stoi(need());
    else if (a == "--repo") repo = need();
    else if (a == "--rt-bin") rt = need();
    else if (a == "--python") python = need();
    else if (a == "--allow-no-token") allow_no_token = true;
    else {
      std::cerr << "unknown argument " << a << "\n";
      return 2;
    }
  }
  const char* token_env = std::getenv("SOUND_BUBBLE_CONTROLLER_TOKEN");
  std::string token = token_env ? token_env : "";
  if (token.empty() && !allow_no_token) {
    std::cerr << "Set SOUND_BUBBLE_CONTROLLER_TOKEN or pass --allow-no-token for local testing.\n";
    return 2;
  }
  soundbubble::ProcessManager manager(repo, rt, python);
  soundbubble::ControllerServer server(host, port, token, manager);
  server.serve();
  manager.stop();
  return 0;
}
