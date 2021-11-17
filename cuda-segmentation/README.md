## Description
This package provides:<br>
1. Segmentation and Cluster lib implemented by CUDA
2. Sample code can show the lib usage and also be used to check perf
   and accuracy by comparing its output with PCL

NOTE:
Now Segmentation Just support: SAC_RANSAC + SACMODEL_PLANE
Segmentation can be used to remove road plane form points cloud.
Cluster can be used to extract objects from points cloud after road plane was removed by Segmentation.

## Prerequisites

### 1. Install Jetpack4.4.1 by SDKManager
### 2. install PCL (Eigen included)
```
$sudo apt-get update
$sudo apt-get install libpcl-dev
```
## Build
$ make

## Run
Please boost CPU and GPU firstly

```
sudo nvpmodel -m 0
sudo jetson_clocks 
```
Usage:<br>
```
./demo [*.pcd]
```
## How to check output
We can get output as below:
```
-------------- cudaSegmentation -----------
CUDA segment by Time: 16.2376 ms.
CUDA modelCoefficients: -0.00915323 -0.87662 -0.481097 -1.17334
CUDA find points: 283505
-------------- cudaExtractCluster -----------
CUDA extract by Time: 24.8037 ms.
PointCloud representing the Cluster: 165687 data points.
PointCloud representing the Cluster: 7098 data points.
PointCloud representing the Cluster: 1263 data points.
PointCloud representing the Cluster: 257 data points.


-------------- PCL(CPU) SACSegmentation -----------
PCL(CPU) segment by Time: 117.612 ms.
Model coefficients: -0.00914787 -0.876947 -0.480501 -1.17268
Model inliers: 283504
-------------- PCL(CPU) EuclideanClusterExtraction -----------
PointCloud representing the planar component: 283504 data points.
PointCloud representing the planar component: 176896 data points.
PCL(CPU) cluster kd-tree by Time: 110 ms.
PCL(CPU) cluster extracted by Time: 4128.99 ms.
PointCloud cluster_indices: 4.
PointCloud representing the Cluster: 166789 data points.
PointCloud representing the Cluster: 7410 data points.
PointCloud representing the Cluster: 1318 data points.
PointCloud representing the Cluster: 427 data points.

```
## Perforamnce table
```
                             GPU         CPU
Segmentation cost time(ms)  16.2376 	117.612
Cluster      cost time(ms)  30.1213 	4075.14
```
**How To Check the Version of the Lib**
-------------------------------------------------
```
$ strings lib* | grep version | grep lib<br>
lib* version: 1.0 Jun  2 2019 09:30:19<br>
```
**Test Enviroment**
-------------------------------------------------
Jetson Xavier AGX 8GB<br>
Jetpack 4.4.1<br>
CUDA 10.2<br>
PCL 1.8<br>
Eigen 3<br>


