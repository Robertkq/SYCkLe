#pragma once

#include <iostream>
#include <sycl/sycl.hpp>

void sycl_ls(std::ostream &os);

void vector_operation(const std::string &input, const std::string &output,
                      const std::string &device);

bool blur_image(const std::string &input_image, const std::string &output_image,
                uint32_t blur_radius, const std::string &device);

inline sycl::device get_sycl_device(const std::string &device) {
  std::cout << "Using SYCL device: " << device << std::endl;
  if (device == "gpu") {
    return sycl::device{sycl::gpu_selector_v};
  } else if (device == "cpu") {
    return sycl::device{sycl::cpu_selector_v};
  } else {
    return sycl::device{sycl::default_selector_v};
  }
}
