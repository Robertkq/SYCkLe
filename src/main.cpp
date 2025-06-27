#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  auto platforms = platform::get_platforms();

  for (const auto &plat : platforms) {
    std::cout << "Platform: " << plat.get_info<info::platform::name>() << "\n";

    for (const auto &dev : plat.get_devices()) {
      std::cout << "  Device: " << dev.get_info<info::device::name>() << "\n";
      std::cout << "    Vendor: " << dev.get_info<info::device::vendor>()
                << "\n";
      std::cout << "    Type: ";
      switch (dev.get_info<info::device::device_type>()) {
      case info::device_type::cpu:
        std::cout << "CPU\n";
        break;
      case info::device_type::gpu:
        std::cout << "GPU\n";
        break;
      case info::device_type::accelerator:
        std::cout << "Accelerator\n";
        break;
      default:
        std::cout << "Other\n";
        break;
      }
    }
  }

  return 0;
}
