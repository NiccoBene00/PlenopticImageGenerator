
# Plenoptic Image Generator (PIG)

## Installation

### Prerequisites

- Visual Studio + CMake
- CUDA 13.1
- OpenCV 4.12.0
- Eigen 3.4.1
- nlohmann::json (automatically installed)

### Project Structure

Create a top-level project folder containing `lib/` and `code/` folders:

```ascii
project/
├── lib/
│   ├── opencv/
│   └── eigen/
└── code/
    ├── config/
    └── datasets/
    └── ...
```

- `lib/` contains all the libraries required by the software (currently OpenCV and Eigen).  
- `code/` contains all the source code.

---

#### Visual Studio Installation

> Visual Studio is an IDE made by Microsoft with its own Windows C++ compiler.

It is the easiest way to code for Windows. You can use other editors like VS Code, but you will lose the quick compilation and building features.

1. Download the installer from [Visual Studio](https://visualstudio.microsoft.com/fr/downloads/).  
2. Run the installer and follow the steps until the workload selection screen.  
3. Select the following workloads:
   - **Desktop Development with C++**
   - **CMake**
   
   > This installs the compiler, Windows SDK, and all tools needed.  
4. Complete the installation and launch Visual Studio.  
5. Select **Open Project Folder** and choose the `code/` folder.

---

#### CUDA Installation

> CUDA is a programming framework for NVIDIA GPUs.

Check if CUDA is already installed:

```bash
nvcc --version

C:\Users\Frbre>nvcc --version
	nvcc: NVIDIA (R) Cuda compiler driver
	Copyright (c) 2005-2025 NVIDIA Corporation
	Built on Tue_Dec_16_19:27:18_Pacific_Standard_Time_2025
	Cuda compilation tools, release 13.1, V13.1.115
	Build cuda_13.1.r13.1/compiler.37061995_0
```

The output shows that the version installed is: "Cuda compilation tools, release 13.1, V13.1.115" -> 13.1.

If the version is not correct, download and install CUDA 13.1 from [NVIDIA CUDA Toolkit 13.1](https://developer.nvidia.com/cuda-13-1-0-download-archive).

---

#### OpenCV Installation

> **OpenCV** is a library for image processing used to load, manipulate and save images.

1. Download the ***OpenCV 4.12.0*** from the [OpenCV Releases](https://opencv.org/releases/) page.
2. Extract the files into `lib/opencv-4.12.0`.
3. Set the following environment variable:
	- `OpenCV_DIR = C:\path-to-project\lib\opencv-4.12.0\build\x64\vc16\lib`
	- `OPENCV_IO_ENABLE_OPENEXR = 1`
		- This is required to enable OpenEXR support in OpenCV.
4. Add the OpenCV `bin` directory to your system `PATH`:
	- `C:\path-to-project\lib\opencv-4.12.0\build\x64\vc16\bin`

---
	
#### Eigen Installation

> **Eigen** is a mathematical library used for fast linear algebra and other mathematical operations.

1. Download ***Eigen 3.4.1*** from [Eigen GitLab](https://gitlab.com/libeigen/eigen/-/releases/3.4.1).
2. Extract files into `lib/eigen-3.4.1`.
3. You need to build and install the library. Open a `cmd` and run the commands:
```bash
cd C:\Eigen
mkdir build
cd .\build
cmake ..
cmake --build . --target install
```

After this, CMake should automatically detect your Eigen installation.

## Running PIG

Once all libraries are installed, Visual Studio will be able to build and compile PIG.

To run PIG, you need to provide some arguments:
```bash
C:\path-to-project\code\out\build\x64-release\pig.exe --system_spec <path-to-system-spec>.json --dataset <path-to-dataset>.json --config <path-to-config>.json --output <path-to-output>.png
```

To automate this, you can set up these arguments in Visual Studio so that you can click **Run**(Hollow green button). Follow these steps:
1. Open Visual Studio in the project folder.
2. Go to **Run and Debug → Edit Configuration**. This will open the file `launch.vs.json`.
3. Add the following configuration:
```json
{
  "version": "0.2.1",
  "defaults": {},
  "configurations": [
    {
      "type": "default",
      "project": "CMakeLists.txt",
      "projectTarget": "pig_cpp.exe",
      "name": "Run PIG",
      "args": [
        "--system_spec",
        "specifications/prototype_spec.json",
        "--dataset",
        "datasets/ball.json",
        "--config",
        "config/pig_default.json",
        "--output",
        "C:\\Users\\Frbre\\Documents\\GitHub\\pig_cpp\\results\\plenoptic.png"
      ]
    }
}
```

Whenever you want to change any of these runtime parameters, open this file and update the values accordingly.