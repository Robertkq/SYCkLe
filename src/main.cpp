#include "syckle.hpp"
#include <CLI/CLI.hpp>
#include <format>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sycl/sycl.hpp>

int main(int argc, char **argv) {
  CLI::App syckle("SYCkLe - SYCL-based tool for device capabilities, image "
                  "processing, and neural networks",
                  "syckle");

  // Global options
  bool verbose = false;

  std::string device;

  syckle.add_flag("-v,--verbose", verbose, "Enable verbose output");
  syckle
      .add_option(
          "-d,--device", device,
          "Specify SYCL device (e.g., 'auto', 'gpu', 'cpu', 'accelerator')")
      ->check(CLI::IsMember({"auto", "gpu", "cpu", "accelerator"}))
      ->default_str("gpu");

  if (!verbose) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
  }

  // Subcommand 1: sycl-ls (list devices)
  auto ls_cmd = syckle.add_subcommand(
      "ls", "List available SYCL devices and their capabilities");

  auto vector_cmd =
      syckle.add_subcommand("vector", "Perform vector operations using SYCL");

  std::string input_vector;
  std::string output_vector = "output_vector.txt";

  vector_cmd
      ->add_option("-i,--input", input_vector, "First input vector file path")
      ->required()
      ->check(CLI::ExistingFile);

  vector_cmd
      ->add_option("-o,--output", output_vector,
                   "Output vector file path (default: output_vector.txt)")
      ->default_str("output_vector.txt");

  auto blur_cmd =
      syckle.add_subcommand("blur", "Apply GPU-accelerated blur to an image");
  std::string input_image;
  std::string output_image = "output_blurred.png";
  int blur_radius = 2;

  blur_cmd->add_option("-i,--input", input_image, "Input image file path")
      ->required()
      ->check(CLI::ExistingFile);
  blur_cmd->add_option("-o,--output", output_image, "Output image file path");
  blur_cmd->add_option("-r,--radius", blur_radius, "Blur radius (default: 2.0)")
      ->check(CLI::Range(1, 20))
      ->default_val(2);

  auto nn_cmd = syckle.add_subcommand("nn", "Run neural network inference");
  std::string model_path;
  std::string input_data;
  std::string output_file = "nn_output.txt";
  std::string framework = "custom";
  int batch_size = 1;

  nn_cmd
      ->add_option("-m,--model", model_path,
                   "Path to neural network model file")
      ->required()
      ->check(CLI::ExistingFile);
  nn_cmd->add_option("-i,--input", input_data, "Input data file or image")
      ->required()
      ->check(CLI::ExistingFile);
  nn_cmd->add_option("-o,--output", output_file, "Output file for results");
  nn_cmd
      ->add_option("-f,--framework", framework,
                   "Framework (custom/onnx/tensorflow)")
      ->check(CLI::IsMember({"custom", "onnx", "tensorflow"}));
  nn_cmd->add_option("-b,--batch-size", batch_size, "Batch size for inference")
      ->check(CLI::Range(1, 1024));

  // Parse command line arguments
  try {
    syckle.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return syckle.exit(e);
  }

  // Execute based on selected subcommand
  if (*ls_cmd) {
    std::cout << "Executing SYCL device listing...\n";

    sycl_ls(std::cout);
  } else if (*vector_cmd) {
    std::cout << "Executing vector operations...\n";

    vector_operation(input_vector, output_vector, device);

  } else if (*blur_cmd) {
    std::cout << std::format(
        "Applying blur with radius {} to image '{}' and saving to '{}'\n",
        blur_radius, input_image, output_image);

    blur_image(input_image, output_image, blur_radius, device);

  } else if (*nn_cmd) {
    // TODO: Call your neural network implementation
    // run_neural_network(model_path, input_data, output_file, framework,
    // batch_size, device_type, verbose);

  } else {
    // No subcommand specified, show help
    std::cout << syckle.help() << std::endl;
  }

  return 0;
}
