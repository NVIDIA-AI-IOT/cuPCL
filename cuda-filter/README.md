**Source Code**
-------------------------------------------------
The are two folders in the package
*  lib: lib and headfiles for CUDA function.<br>
  More details about the API canbe found in the header file.<br>
*  the sample code about how to use the function<br>
  and test<br>

**How to Compile**
-------------------------------------------------
Please install CUDA toolkit, EIGEN and PCL firstly<br>
$ make

**How to Run the Sample**
-------------------------------------------------
Usage:<br>
>./$(App) test-0.pcd<br>

**How to check output**
-------------------------------------------------
We can get output like below:
-------------------------------------------------
------------checking CUDA ---------------- 
CUDA Loaded 119978 data points from src.pcd with the following fields: x y z

------------checking CUDA PassThrough ---------------- 
CUDA PassThrough by Time: 0.589752 ms.countLeft: 15860


------------checking PCL ---------------- 
PCL Loaded 119978 data points from src.pcd with the following fields: x y z

------------checking PCL PassThrough ---------------- 
PCL PassThrough by Time: 2.82811 ms.
PointCloud before filtering: 119978 data points (x y z).
PointCloud after filtering: 15860 data points (x y z).


-------------------------------------------------


**How To Check the Version of the Lib**
-------------------------------------------------
$ strings lib* | grep version | grep lib<br>
lib* version: 1.0 Jun  2 2019 09:30:19<br>

**Perforamnce table**
-------------------------------------------------
 	                    GPU 		CPU 		<br>
count of points cloud 	11w+ 		11w+ 		<br>
dim                     Z	    	Z 	    	<br>
down,up FilterLimits	(0.0,1.0)   (0.0,1.0)	<br>
limitsNegative			false	    false       <br>
Points selected			15860   	15860		<br>
cost time(ms) 	        0.589752 	2.82811 	<br>

**Test Enviroment**
-------------------------------------------------
TX Xavier AGX 8GB<br>
Jetpack 4.4.1<br>
CUDA 10.2<br>
PCL 1.8<br>
Eigen 3<br>

