**Summary**
-------------------------------------------------
This package provides:<br>
1. Segmentation lib implemented by CUDA
2. Sample code can show the lib usage and also be used to check perf
   and accuracy by comparing its output with PCL

NOTE:
Now Just support: SAC_RANSAC + SACMODEL_PLANE

**How to Compile**
-------------------------------------------------
Please install cuda toolkit, Boost, EIGEN and PCL firstly<br>
$ make

**How to Run the Sample**
-------------------------------------------------
Please boost CPU and GPU firstly
Usage:<br>
>./$(App) *.pcd<br>

**How to check output**

$ ./demo test-0.pcd
-------------------------------------------------
We can get output like below:
-------------------------------------------------

-------------------------
CUDA segment by Time: 14.5712 ms.
CUDA modelCoefficients: -0.00269913 0.0424975 0.999093 2.10639
CUDA find points: 7519
-------------------------
PCL segment by Time: 75.2655 ms.
Model coefficients: -0.0026991 0.0424981 0.999093 2.10639
Model inliers: 7519

**Perforamnce table**
-------------------------------------------------
 	                    GPU 		CPU 		<br>
count of points cloud 	11w+ 		11w+ 		<br>
Points selected			7519   	    7519		<br>
cost time(ms) 	        14.5712 	75.2655 	<br>

**How To Check the Version of the Lib**
-------------------------------------------------
$ strings lib* | grep version | grep lib<br>
lib* version: 1.0 Jun  2 2019 09:30:19<br>

**Test Enviroment**
-------------------------------------------------
TX Xavier AGX 8GB<br>
Jetpack 4.4.1<br>
CUDA 10.2<br>
PCL 1.8<br>
Eigen 3<br>


