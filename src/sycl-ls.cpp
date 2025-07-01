#include <iostream>
#include <sycl/sycl.hpp>

void sycl_ls(std::ostream &os) {
  auto platforms = sycl::platform::get_platforms();

  for (const auto &plat : platforms) {
    os << "Platform: " << plat.get_info<sycl::info::platform::name>() << "\n";

    for (const auto &dev : plat.get_devices()) {
      os << "  Device: " << dev.get_info<sycl::info::device::name>() << "\n";
      os << "    Vendor: " << dev.get_info<sycl::info::device::vendor>()
         << "\n";
      os << "    Type: ";
      switch (dev.get_info<sycl::info::device::device_type>()) {
      case sycl::info::device_type::cpu:
        os << "CPU\n";
        break;
      case sycl::info::device_type::gpu:
        os << "GPU\n";
        break;
      case sycl::info::device_type::accelerator:
        os << "Accelerator\n";
        break;
      default:
        os << "Other\n";
        break;
      }

      os << "    Backend: ";
      sycl::backend be = dev.get_backend();
      switch (be) {
      case sycl::backend::opencl:
        os << "OpenCL\n";
        break;
      case sycl::backend::ext_oneapi_cuda:
        os << "CUDA\n";
        break;
      case sycl::backend::ext_oneapi_level_zero:
        os << "Level Zero\n";
        break;
      case sycl::backend::ext_oneapi_native_cpu:
        os << "Native CPU\n";
        break;
      case sycl::backend::ext_oneapi_hip:
        os << "HIP\n";
        break;
      default:
        os << "Unknown\n";
        break;
      }
    }
  }
}
