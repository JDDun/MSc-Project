#pragma once


#include <iostream>
#include "kernels.cuh"

// CUDA error checks
#define WIN32_LEAN_AND_MEAN
#include <helper_cuda.h>
#include <helper_functions.h>


class gpuHazardClass
{
private:
	int rows, cols;

	float *d_heightMap, *d_gaussMap, *d_slope, *d_rough;
	unsigned char *d_lum;
	int *d_preGrassScore, *d_score;

	float sWeight, rWeight, shWeight;

public:
	__host__ gpuHazardClass();
	__host__ gpuHazardClass(float *DEM, const int ydim, const int xdim, unsigned char *lum, const float slopeWeight, const float roughWeight, const float shadowWeight);

	__host__ ~gpuHazardClass();

	// Apply gaussian filter.
	__host__ void gaussFilter();

	// Methods to create each map.
	__host__ void mapSlope();
	__host__ void mapRough();
	__host__ void mapHazards();
	__host__ void grassfire();

	__host__ int createHazardMap(); // Linking function, calls the others with error checking!

	__host__ void copyScores();
	
	// Get methods to retrieve each of the map components.
	__host__ float* getSlopeMap();
	__host__ float* getRoughMap();
	__host__ int* getPreGrassMap();	
	__host__ int* getHazardMap();	

	// Check the success of a kernel launch, call after each launch.
	__host__ void kernelCheck();

	// Max finding method, 2 kernel launches required.
	__host__ void findMax(float* in_data, float* max);
};


