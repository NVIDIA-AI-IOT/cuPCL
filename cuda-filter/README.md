## Description
The are two folders in the package
1. lib for segmentation implemented by CUDA
2. Sample code can show the lib usage and also be used to check perf
   and accuracy by comparing its output with PCL
3. A point clound: sample.pcd which has 119978 points

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
CUDA Loaded 119978 data points from PCD file with the following fields: x y z

------------checking CUDA PassThrough ---------------- 
CUDA PassThrough by Time: 0.580568 ms.
Points selected: 15860


PCL(CPU) Loaded 119978 data points from PCD file with the following fields: x y z

------------checking PCL(CPU) PassThrough ---------------- 
PCL(CPU) PassThrough by Time: 2.72104 ms.
PointCloud after filtering: 15860 data points (x y z).

```


**Perforamnce table**
-------------------------------------------------
```
                        GPU 		CPU
count of points cloud   11w+        11w+
dim                     Z	    	Z
down,up FilterLimits	(0.0,1.0)   (0.0,1.0)
limitsNegative          false       false
Points selected         15860       15860
cost time(ms)           0.589752    2.82811
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

