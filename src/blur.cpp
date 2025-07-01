#include <chrono>
#include <cstdint>
#include <format>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <sycl/sycl.hpp>

bool blur_image(const std::string &input_image, const std::string &output_image,
                uint32_t blur_radius = 2) {
  sycl::queue queue{sycl::gpu_selector_v,
                    sycl::property::queue::enable_profiling{}};
  if (!queue.get_device().is_gpu()) {
    std::cerr << "No GPU found. Exiting." << std::endl;
    return false;
  } else {
    std::cout << "Using GPU: "
              << queue.get_device().get_info<sycl::info::device::name>()
              << std::endl;
  }

  cv::Mat input = cv::imread(input_image);
  if (input.empty()) {
    std::cerr << "Error reading input image: " << input_image << std::endl;
    return false;
  }

  const int height = input.rows;
  const int width = input.cols;
  const int channels = input.channels();

  std::cout << std::format("Image: {}x{} with {} channels\n", width, height,
                           channels);

  // Copy OpenCV data to contiguous array
  std::vector<uint8_t> input_data(input.data,
                                  input.data + (height * width * channels));
  std::vector<uint8_t> output_data(height * width * channels);

  // Create 2D buffer: [height][width * channels]
  sycl::buffer<uint8_t, 2> input_buffer(
      input_data.data(), sycl::range<2>(height, width * channels));
  sycl::buffer<uint8_t, 2> output_buffer(
      output_data.data(), sycl::range<2>(height, width * channels));

  try {
    sycl::event blur_event = queue.submit([&](sycl::handler &h) {
      auto input_acc = input_buffer.get_access<sycl::access::mode::read>(h);
      auto output_acc = output_buffer.get_access<sycl::access::mode::write>(h);

      // Simple copy: process EVERY pixel
      h.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> idx) {
        const int y = idx[0];
        const int x = idx[1];

        // Copy all channels for this pixel
        for (int c = 0; c < channels; c++) {
          output_acc[y][x * channels + c] = input_acc[y][x * channels + c];
        }
      });
    });

    blur_event.wait();

    std::cout << std::format("========== Profiling Results ==========\n");
    auto start_time =
        blur_event
            .get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end_time =
        blur_event
            .get_profiling_info<sycl::info::event_profiling::command_end>();
    auto duration = std::chrono::nanoseconds(end_time - start_time);
    std::cout << std::format(
        "Blur operation took: \n{} nanoseconds\n{} microseconds\n"
        "{} milliseconds\n",
        duration.count(),
        std::chrono::duration_cast<std::chrono::microseconds>(duration).count(),
        std::chrono::duration_cast<std::chrono::milliseconds>(duration)
            .count());
    std::cout << std::format("=======================================\n");

  } catch (const sycl::exception &e) {
    std::cerr << "SYCL Exception: " << e.what() << std::endl;
    std::cerr << "Error code: " << e.code() << std::endl;
    return false;
  } catch (const std::exception &e) {
    std::cerr << "Standard Exception: " << e.what() << std::endl;
    return false;
  } catch (...) {
    std::cerr << "Unknown exception occurred!" << std::endl;
    return false;
  }

  cv::Mat output(height, width, input.type(), output_data.data());

  cv::imwrite(output_image, output);
  std::cout << std::format("Blurred image saved to: {}\n", output_image);
  cv::imshow("Blurred Image", output);
  cv::imshow("Original Image", input);
  cv::waitKey(0);

  return true;
}