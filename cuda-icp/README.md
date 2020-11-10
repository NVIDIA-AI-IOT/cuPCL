**Summary**
-------------------------------------------------
This project provides:<br>
1. ICP implement by CUDA
2. Sample code can show the function usage, which can also be used to check perf
   and output verification by comparing its output with ICP and GICP from PCL
3. Generate the customer release package by one command - "$ make release"

**Source Code**
-------------------------------------------------
*  lib: the implementation of the convolution function.<br>
  More details about the API canbe found in the header file.<br>
*  sample: the sample code about how to use the function<br>
  and perf test<br>

**CUDA ICP LIB Dependency**
-------------------------------------------------
Please install cuda toolkit, EIGEN<br>

**How to Compile**
-------------------------------------------------
The sample depends PCL to load data and show result, please install PCL firstly<br>
$ make<br>


**How to Run the Sample**
-------------------------------------------------
Usage:<br>
>./$(App)<br>

**How to check output**
-------------------------------------------------
We can get output like below:<br>

    CUDA ICP by Time: 2071.81 ms.<br>
    CUDA ICP fitness_score: 6.5795e-13<br>
    matrix_icp calculated Matrix by Class ICP <br>
    Rotation matrix :<br>
        | 0.930060 -0.367142 -0.013954 | <br>
    R = | 0.367313 0.930010 0.012710 | <br>
        | 0.008311 -0.016946 0.999821 | <br>
    Translation vector :<br>
    t = < 0.004626, -0.006548, 0.201614 > <br>

"Time" is the time ICP costed.<br>
"fitness_score" is the score of the ICP transform, the less the better.<br>


**How To Check the Version of the Lib**
-------------------------------------------------
$ strings lib* | grep version | grep lib<br>
lib* version: 1.0 Jun  2 2019 09:30:19<br>

**Perforamnce table for default parameters**
-------------------------------------------------

GPU 	GICP 	ICP<br>
count of points cloud 	7000 	7000 	7000<br>
maximum of iterations 	20 	    20 	    20<br>
cost time(ms) 	        43.3    652.8   7746.0<br>
fitness_score           0.514   0.525   0.643<br>

**Test Enviroment**
-------------------------------------------------
Xavier<br>
jetpack 4.4<br>
cuda 10.2<br>
Eigen 3<br>
PCL 1.8<br>
