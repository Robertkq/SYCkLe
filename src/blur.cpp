#include "syckle.hpp"
#include <chrono>
#include <cstdint>
#include <format>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

bool blur_image(const std::string &input_image, const std::string &output_image,
                uint32_t blur_radius, const std::string &device) {
  sycl::queue queue{get_sycl_device(device),
                    sycl::property::queue::enable_profiling{}};

  std::cout << "Using Device: "
            << queue.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  cv::Mat input = cv::imread(input_image);
  if (input.empty()) {
    std::cerr << "Error reading input image: " << input_image << std::endl;
    return false;
  }

  const int height = input.rows;
  const int width = input.cols;
  const int channels = input.channels();

  std::cout << std::format("Image: {}x{} with {} channels, blur radius: {}\n",
                           width, height, channels, blur_radius);

  // Copy OpenCV data to contiguous array
  std::vector<uint8_t> input_data(input.data,
                                  input.data + (height * width * channels));
  std::vector<uint8_t> output_data(height * width * channels);

  {
    try {

      // Create 2D buffer: [height][width * channels]
      sycl::buffer<uint8_t, 2> input_buffer(
          input_data.data(), sycl::range<2>(height, width * channels));
      sycl::buffer<uint8_t, 2> output_buffer(
          output_data.data(), sycl::range<2>(height, width * channels));

      // Record start time using chrono
      auto start_time = std::chrono::high_resolution_clock::now();

      sycl::event blur_event = queue.submit([&](sycl::handler &h) {
        auto input_acc = input_buffer.get_access<sycl::access::mode::read>(h);
        auto output_acc =
            output_buffer.get_access<sycl::access::mode::write>(h);

        h.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> idx) {
          const int y = idx[0];
          const int x = idx[1];
          const int blur_r = static_cast<int>(blur_radius);

          for (int c = 0; c < channels; c++) {
            int sum = 0;
            int count = 0;

            for (int dy = -blur_r; dy <= blur_r; dy++) {
              for (int dx = -blur_r; dx <= blur_r; dx++) {
                int ny = y + dy;
                int nx = x + dx;

                if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                  sum += input_acc[ny][nx * channels + c];
                  count++;
                }
              }
            }

            if (count > 0) {
              output_acc[y][x * channels + c] =
                  static_cast<uint8_t>(sum / count);
            } else {
              output_acc[y][x * channels + c] = input_acc[y][x * channels + c];
            }
          }
        });
      });

      blur_event.wait();

      auto end_time = std::chrono::high_resolution_clock::now();

      auto wall_duration = end_time - start_time;
      auto wall_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(wall_duration)
              .count();
      auto wall_us =
          std::chrono::duration_cast<std::chrono::microseconds>(wall_duration)
              .count();
      auto wall_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(wall_duration)
              .count();

      // Get SYCL profiling info (device execution time)
      auto device_start =
          blur_event
              .get_profiling_info<sycl::info::event_profiling::command_start>();
      auto device_end =
          blur_event
              .get_profiling_info<sycl::info::event_profiling::command_end>();
      auto device_duration_ns = device_end - device_start;

      // Convert using std::chrono
      auto device_chrono_ns = std::chrono::nanoseconds(device_duration_ns);
      auto device_duration_us =
          std::chrono::duration_cast<std::chrono::microseconds>(
              device_chrono_ns)
              .count();
      auto device_duration_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              device_chrono_ns)
              .count();

      // Print profiling results
      std::cout << std::format("\n=== BLUR PROFILING RESULTS ===\n");
      std::cout << std::format(
          "Wall Clock Time (Host + Device + Transfers):\n");
      std::cout << std::format("  {} nanoseconds\n", wall_ns);
      std::cout << std::format("  {} microseconds\n", wall_us);
      std::cout << std::format("  {} milliseconds\n", wall_ms);

      std::cout << std::format("\nDevice Execution Time (Kernel Only):\n");
      std::cout << std::format("  {} nanoseconds\n", device_duration_ns);
      std::cout << std::format("  {} microseconds\n", device_duration_us);
      std::cout << std::format("  {} milliseconds\n", device_duration_ms);

      std::cout << std::format("\nImage Size: {}x{} ({} pixels)\n", width,
                               height, width * height);
      std::cout << std::format("Blur Radius: {}\n", blur_radius);
      std::cout << std::format("===============================\n");

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
  }

  cv::Mat output(height, width, input.type(), output_data.data());

  cv::imwrite(output_image, output);
  std::cout << std::format("Blurred image saved to: {}\n", output_image);
  cv::imshow("Blurred Image", output);
  cv::imshow("Original Image", input);
  cv::waitKey(0);

  return true;
}