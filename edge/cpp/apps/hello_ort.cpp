#include "soundbubble/pipeline.hpp"

#include <iostream>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: soundbubble-hello-ort MODEL.onnx\n";
    return 2;
  }
  try {
    auto inf = soundbubble::make_onnx_inferencer(argv[1], 1, 1, nullptr, std::nullopt);
    std::cout << "loaded " << argv[1] << " dis_embed=" << (inf->accepts_dis_embed() ? "yes" : "no") << "\n";
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << exc.what() << "\n";
    return 1;
  }
}
