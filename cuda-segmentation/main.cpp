/*
 * Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include <fstream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/filters/random_sample.h>
//#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <chrono>
#include "cuda_runtime.h"
#include "lib/cudaSegmentation.h"

void Getinfo(void)
{
    cudaDeviceProp prop;

    int count = 0;
    cudaGetDeviceCount(&count);
    printf("\nGPU has cuda devices: %d\n", count);
    for (int i = 0; i < count; ++i) {
        cudaGetDeviceProperties(&prop, i);
        printf("----device id: %d info----\n", i);
        printf("  GPU : %s \n", prop.name);
        printf("  Capbility: %d.%d\n", prop.major, prop.minor);
        printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
        printf("  Const memory: %luKB\n", prop.totalConstMem  >> 10);
        printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
        printf("  warp size: %d\n", prop.warpSize);
        printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
        printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
    printf("\n");
}

void testCUDA(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, 1000>> time_span =
     std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);

  cudaStream_t stream = NULL;
  cudaStreamCreate (&stream);

  int nCount = cloud->width * cloud->height;
  float *inputData = (float *)cloud->points.data();

  float *input = NULL;
  cudaMallocManaged(&input, sizeof(float) * 4 * nCount, cudaMemAttachHost);
  cudaStreamAttachMemAsync (stream, input);
  //cudaMemcpyAsync(input, inputData, sizeof(float) * 4 * nCount, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);

  int *index = NULL;
  // index should >= nCount of maximum inputdata,
  // index can be used for multi-inputs, be allocated and freed just at beginning and end
  cudaMallocManaged(&index, sizeof(int) * nCount, cudaMemAttachHost);
  cudaStreamAttachMemAsync (stream, index);
  cudaStreamSynchronize(stream);
  // modelCoefficients can be used for multi-inputs, be allocated and freed just at beginning and end
  float *modelCoefficients = NULL;
  int modelSize = 4;
  cudaMallocManaged(&modelCoefficients, sizeof(float) * modelSize, cudaMemAttachHost);
  cudaStreamAttachMemAsync (stream, modelCoefficients);
  cudaStreamSynchronize(stream);

  //Now Just support: SAC_RANSAC + SACMODEL_PLANE
  cudaSegmentation cudaSeg(SACMODEL_PLANE, SAC_RANSAC, stream);

  double threshold = 0.01;
  bool optimizeCoefficients = true;
  std::vector<int> indexV;
  t1 = std::chrono::steady_clock::now();
  segParam_t setP;
  setP.distanceThreshold = threshold; 
  setP.maxIterations = 50;
  setP.probability = 0.99;
  setP.optimizeCoefficients = optimizeCoefficients;
  cudaSeg.set(setP);
  cudaMemcpyAsync(input, inputData, sizeof(float) * 4 * nCount, cudaMemcpyHostToDevice, stream);
  cudaSeg.segment(input, nCount, index, modelCoefficients);

  for(int i = 0; i < nCount; i++)
  {
    if(index[i] == 1) 
    indexV.push_back(i);
  }

  t2 = std::chrono::steady_clock::now();
  time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  std::cout << "CUDA segment by Time: " << time_span.count() << " ms."<< std::endl;

  //std::cout << "CUDA index Size : " <<indexV.size()<< std::endl;

  std::cout << "CUDA modelCoefficients: " << modelCoefficients[0]
    <<" "<< modelCoefficients[1]
    <<" "<< modelCoefficients[2]
    <<" "<< modelCoefficients[3]
    << std::endl;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudDst(new pcl::PointCloud<pcl::PointXYZ>);
  cloudDst->width  = nCount;
  cloudDst->height = 1;
  cloudDst->points.resize (cloudDst->width * cloudDst->height);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudNew(new pcl::PointCloud<pcl::PointXYZ>);
  cloudNew->width  = nCount;
  cloudNew->height = 1;
  cloudNew->points.resize (cloudDst->width * cloudDst->height);

  int check = 0;
  for (std::size_t i = 0; i < nCount; ++i)
  {
    if (index[i] == 1)
    {
      cloudDst->points[i].x = input[i*4+0];
      cloudDst->points[i].y = input[i*4+1];
      cloudDst->points[i].z = input[i*4+2];
      check++;
    }
    else if (index[i] != 1)
    {
      cloudNew->points[i].x = input[i*4+0];
      cloudNew->points[i].y = input[i*4+1];
      cloudNew->points[i].z = input[i*4+2];
    }
  }
  pcl::io::savePCDFileASCII ("after-seg-cuda.pcd", *cloudDst);
  pcl::io::savePCDFileASCII ("after-seg-cudaNew.pcd", *cloudNew);

  std::cout << "CUDA find points: " << check << std::endl;

  cudaFree(input);
  cudaFree(index);
  cudaFree(modelCoefficients);
}

void testPCL(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{

  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.01);

  seg.setMaxIterations(50);
  seg.setProbability(0.99);
  seg.setInputCloud (cloud);

  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, 1000>> time_span =
     std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  t1 = std::chrono::steady_clock::now();
  seg.segment (*inliers, *coefficients);
  t2 = std::chrono::steady_clock::now();
  time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  std::cout << "PCL(CPU) segment by Time: " << time_span.count() << " ms."<< std::endl;

  if (inliers->indices.size () == 0)
  {
    PCL_ERROR ("Could not estimate a planar model for the given dataset.");
  }

  std::cerr << "Model coefficients: " << coefficients->values[0] << " " 
                                      << coefficients->values[1] << " "
                                      << coefficients->values[2] << " " 
                                      << coefficients->values[3] << std::endl;

  std::cerr << "Model inliers: " << inliers->indices.size () << std::endl;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudDst(new pcl::PointCloud<pcl::PointXYZ>);
  cloudDst->width  = inliers->indices.size ();
  cloudDst->height = 1;
  cloudDst->points.resize (inliers->indices.size ());
  for (std::size_t i = 0; i < inliers->indices.size (); ++i)
  {
    cloudDst->points[i] = cloud->points[inliers->indices[i]];
  }
  pcl::io::savePCDFileASCII ("after-seg-pcl.pcd", *cloudDst);

}

int main(int argc, const char **argv)
{
  std::string file = "./sample.pcd";
  if(argc > 1) file = (argv[1]);

  Getinfo();

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudSrc(new pcl::PointCloud<pcl::PointXYZ>);

  if (pcl::io::loadPCDFile<pcl::PointXYZ>(file.c_str(), *cloudSrc )== -1)
  {
    std::cout << "Error:can not open the file: "<< file.c_str() << std::endl;
    return(-1);
  }

  std::cout << "-------------------------"<< std::endl;
  testCUDA(cloudSrc);

  std::cout << "-------------------------"<< std::endl;
  testPCL(cloudSrc);

  return 0;
}
