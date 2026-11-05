#include "soundbubble/mixctl.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: soundbubble-mix-listen PORT [BAUD]\n"
                 "  Windows: COM7   Pi: /dev/ttyACM0\n"
                 "  Prints the dry mix published by MIX lines. Unplug/replug is tolerated.\n";
    return 2;
  }
  int baud = argc > 2 ? std::atoi(argv[2]) : 115200;
  soundbubble::MixControl ctl(argv[1], baud, 0.f);
  ctl.start();
  float last = -1.f;
  while (true) {
    float dry = ctl.dry();
    if (dry != last) {
      std::cout << "dry=" << dry << " wet=" << (1.f - dry) << "\n";
      last = dry;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}
