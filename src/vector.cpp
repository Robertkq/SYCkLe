#include "syckle.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

void vector_add(const std::vector<int> &a, const std::vector<int> &b,
                std::vector<int> &c, const std::string &device);

void vector_operation(const std::string &input, const std::string &output,
                      const std::string &device) {
  std::ifstream infile(input);
  if (!infile.is_open()) {
    std::cerr << "Error opening input file: " << input << std::endl;
    return;
  }
  std::vector<int> a, b;
  int value;

  std::string line;
  if (std::getline(infile, line)) {
    std::stringstream ss(line);
    while (ss >> value) {
      a.push_back(value);
    }
  }

  if (std::getline(infile, line)) {
    std::stringstream ss(line);
    while (ss >> value) {
      b.push_back(value);
    }
  }

  infile.close();

  if (a.size() != b.size()) {
    std::cerr << "Vectors must be of the same size!" << std::endl;
    return;
  }

  std::cout << "Vector a: ";
  for (int val : a)
    std::cout << val << " ";
  std::cout << "\nVector b: ";
  for (int val : b)
    std::cout << val << " ";
  std::cout << std::endl;

  std::vector<int> c(a.size());
  c.assign(a.size(), 0);

  vector_add(a, b, c, device);

  std::cout << "Result vector c: ";
  for (int val : c)
    std::cout << val << " ";
  std::cout << std::endl;

  std::ofstream outfile(output);
  if (!outfile.is_open()) {
    std::cerr << "Error opening output file: " << output << std::endl;
    return;
  }
  for (int val : c) {
    outfile << val << " ";
  }
}

void vector_add(const std::vector<int> &a, const std::vector<int> &b,
                std::vector<int> &c, const std::string &device) {
  if (a.size() != b.size() || a.size() != c.size()) {
    std::cerr << "Vectors must be of the same size!" << std::endl;
    return;
  }

  try {
    sycl::queue queue{get_sycl_device(device),
                      sycl::property::queue::enable_profiling{}};

    std::cout << "Using Device: "
              << queue.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    sycl::buffer<int> a_buf(a.data(), sycl::range<1>(a.size()));
    sycl::buffer<int> b_buf(b.data(), sycl::range<1>(b.size()));
    sycl::buffer<int> c_buf(c.data(), sycl::range<1>(c.size()));

    auto start_time = std::chrono::high_resolution_clock::now();

    sycl::event kernel_event = queue.submit([&](sycl::handler &h) {
      auto a_acc = a_buf.get_access<sycl::access::mode::read>(h);
      auto b_acc = b_buf.get_access<sycl::access::mode::read>(h);
      auto c_acc = c_buf.get_access<sycl::access::mode::write>(h);

      h.parallel_for<class VectorAdd>(
          sycl::range<1>(a.size()),
          [=](sycl::id<1> idx) { c_acc[idx] = a_acc[idx] + b_acc[idx]; });
    });

    kernel_event.wait();

    // Record end time using chrono
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate wall clock time using chrono
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
        kernel_event
            .get_profiling_info<sycl::info::event_profiling::command_start>();
    auto device_end =
        kernel_event
            .get_profiling_info<sycl::info::event_profiling::command_end>();
    auto device_duration_ns = device_end - device_start;

    // Convert using std::chrono
    auto device_chrono_ns = std::chrono::nanoseconds(device_duration_ns);
    auto device_duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(device_chrono_ns)
            .count();
    auto device_duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(device_chrono_ns)
            .count();

    // Print profiling results
    std::cout << "\n=== PROFILING RESULTS ===" << std::endl;
    std::cout << "Wall Clock Time (Host + Device + Transfers):" << std::endl;
    std::cout << "  " << wall_ns << " nanoseconds" << std::endl;
    std::cout << "  " << wall_us << " microseconds" << std::endl;
    std::cout << "  " << wall_ms << " milliseconds" << std::endl;

    std::cout << "\nDevice Execution Time (Kernel Only):" << std::endl;
    std::cout << "  " << device_duration_ns << " nanoseconds" << std::endl;
    std::cout << "  " << device_duration_us << " microseconds" << std::endl;
    std::cout << "  " << device_duration_ms << " milliseconds" << std::endl;

    std::cout << "\nVector Size: " << a.size() << " elements" << std::endl;
    std::cout << "=========================" << std::endl;

  } catch (const sycl::exception &e) {
    std::cerr << "SYCL Exception: " << e.what() << std::endl;
    std::cerr << "Error code: " << e.code() << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Standard Exception: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "Unknown exception occurred!" << std::endl;
  }
}