#include <cuda_runtime.h>

// Compute slope from input DEM.
__global__ void slopeKernel(float *d_DEM, const int rows, const int cols, float *d_deg);
// Compute a measure for roughness from root mean square of average(height - avgHeight).
__global__ void roughnessKernel(float *d_DEM, const int rows, const int cols, float *d_rough);
// Computes hazard score, taking into account roughness, slope & shadow.
__global__ void hazardKernel(float *slope, float *rough, unsigned char *img, const int rows, const int cols, int *d_score,
	float* slopeMax, float* roughMax, float slopeWeight, float roughWeight, float shadowWeight);

// Applies grassfire algorithm in four stages.
__global__ void rowRasterKernel(int *d_score, const int rows, const int cols);
__global__ void colRasterKernel(int *d_score, const int rows, const int cols);
__global__ void rowAntiRasterKernel(int *d_score, const int rows, const int cols);
__global__ void colAntiRasterKernel(int *d_score, const int rows, const int cols);

// Max kernels
__global__ void maxFirst(float* in_array, float* block_max, const int size);
__global__ void maxSecond(float* block_max, float* max);

// Gaussian smoothing kernel
__global__ void gaussKernel(float* in_array, float* out_array, const int rows, const int cols);