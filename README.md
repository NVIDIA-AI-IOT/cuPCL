# cuda-pcl
<a><img src="https://img.shields.io/badge/-Documentation-bright"/></a>

cuda-pcl has some libraries used to process points cloud with CUDA and some samples for their usage.
The are several subfolders in the project and every subfolder has:
1. lib for segmentation implemented by CUDA
2. Sample code can show the lib usage and also be used to check perf
   and accuracy by comparing its output with PCL

To get started, follow the instructions below.  If you run into any issues please [let us know](../../issues).

## Getting Started

To get started with trt_pose, follow these steps.

### Step 1 - Install Dependencies

1. Install Jetpack4.4.1 by SDKManager
2. install PCL (Eigen included)

```
$sudo apt-get update
$sudo apt-get install libpcl-dev
```

### Step 2 - Build
Enter any subfolder and then
```
$ make
```
### Step 3 - Run
1. Please boost CPU and GPU firstly
```
$ sudo nvpmodel -m 0
$ sudo jetson_clocks 
```

2. Usage:
```
$ ./demo [*.pcd]
```

## How to check the Version of the Lib
```
$ strings lib* | grep version | grep lib
lib* version: 1.0 Jun  2 2019 09:30:19
```
## Test Enviroment
```
Jetson Xavier AGX 8GB
Jetpack 4.4.1
CUDA 10.2
PCL 1.8
Eigen 3
```
