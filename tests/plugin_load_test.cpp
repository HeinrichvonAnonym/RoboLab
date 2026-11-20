#include "plugin_test_support.h"
#include <iostream>
int main(int argc, char** argv) {
  try {
    for (int i = 1; i < argc; ++i) {
      TestPlugin module(argv[i]);
      std::cout << "Loaded factory: " << argv[i] << '\n';
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
