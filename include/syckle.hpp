#pragma once

#include <iostream>

void sycl_ls(std::ostream &os);

bool blur_image(const std::string &input_image, const std::string &output_image,
                uint32_t blur_radius = 2.0);