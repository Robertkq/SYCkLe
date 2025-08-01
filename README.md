# SYCkLe — A SYCL‑Based Computing Tool

**SYCkLe** is a command-line utility designed to explore heterogeneous computing via SYCL on both CPU and GPU. It supports vector operations, image processing via OpenCV, and neural network inference—accessible through a clean CLI interface powered by CLI11.

---

## Prerequisites & GPU Support

- Install the **Intel oneAPI Base Toolkit**, which includes the DPC++ compiler: [Download from Intel](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
- To use **NVIDIA GPUs**, first install the **CUDA Toolkit**: [Download from NVIDIA](https://developer.nvidia.com/cuda-toolkit)
  Then install **oneAPI for NVIDIA GPUs** plugin by Codeplay: [Download plugin](https://developer.codeplay.com/products/oneapi/nvidia/home)

Before building or running SYCkLe, source Intel environment variables using the provided `setvars.sh` script (or `setvars.bat` on Windows).

---

## Build Instructions

Build SYCkLe using CMake with Ninja and Intel DPC++ (`icx`):

```bash
mkdir build
cd build
cmake -G Ninja -DCMAKE_CXX_COMPILER=icx ..
cmake --build .
```

Ensure you have available:

- Intel oneAPI Base Toolkit with DPC++ (`icx`, `icpx`)  
- OpenCV development libraries for image I/O  
- CLI11 header-only library  
- C++20-compatible SYCL toolchain

---

## Tool Overview & Usage

Run the tool with no arguments to view usage:

```bash
./syckle
```

```
SYCkLe — SYCL-based tool for device capabilities, image processing, and neural networks

Usage: syckle [OPTIONS] [SUBCOMMANDS]

OPTIONS:
  -h, --help                     Show help message
  -v, --verbose                  Enable verbose output
  -d, --device {auto,gpu,cpu,accelerator} [gpu]
                                 Choose SYCL device

SUBCOMMANDS:
  ls       List available SYCL devices
  vector   Perform SYCL vector operations
  blur     Apply GPU‑accelerated blur to an image
  nn       Run neural network inference
```

---

## Commands

### `ls` — List SYCL Devices

```bash
./syckle ls
```

Displays available platforms and devices, with vendor, type, and backend details.

### `vector` — Vector Computations

```bash
./syckle vector --input input.txt [--output output_vector.txt]
```

- `--input`: path to input vector file (required)  
- `--output`: output file path (defaults to `output_vector.txt`)

### `blur` — GPU‑Accelerated Image Blur

```bash
./syckle blur --input image.png [--output result.png] [--radius N]
```

- `--input`: path to image (required)  
- `--output`: path to save blurred image  
- `--radius`: blur radius (range 1–20, default is 2)

### `nn` — Neural Network Inference

*Not implemented yet.*

---

## Example Usage

```bash
# List available SYCL devices
./syckle ls

# Perform vector operations
./syckle vector --input data.txt --verbose

# Apply a blur filter
./syckle blur --input input.jpg --output blurred.jpg --radius 5
```

---

I hope this projects stands to help someone discover how to run programs on CPU/GPU using SYCL.

Happy experimenting with SYCL-based acceleration!  
