#include "gpuHazardClass.cuh"

/////////////////////////
// --- CONSTRUCTOR --- //
/////////////////////////

__host__ gpuHazardClass::gpuHazardClass()
{
	rows = 0;
	cols = 0;

	d_heightMap = nullptr;
	d_gaussMap = nullptr;
	d_slope = nullptr;
	d_rough = nullptr;
	d_lum = nullptr;
	d_score = nullptr;

	sWeight = 1;
	rWeight = 1;
	shWeight = 1;

}

__host__ gpuHazardClass::gpuHazardClass(float *DEM, const int ydim, const int xdim, unsigned char *lum, const float slopeWeight, const float roughWeight, const float shadowWeight)
{
	rows = ydim;
	cols = xdim;

	std::cout << "Allocating Device Memory\n";
	// Allocating memory for device pointers.
	checkCudaErrors(cudaMalloc(&d_heightMap, sizeof(float) * cols * rows));
	checkCudaErrors(cudaMalloc(&d_gaussMap, sizeof(float) * cols * rows));
	checkCudaErrors(cudaMalloc(&d_slope, sizeof(float) * cols * rows));
	checkCudaErrors(cudaMalloc(&d_rough, sizeof(float) * cols * rows));
	checkCudaErrors(cudaMalloc(&d_preGrassScore, sizeof(int) * cols * rows));
	checkCudaErrors(cudaMalloc(&d_score, sizeof(int) * cols * rows));
	checkCudaErrors(cudaMalloc(&d_lum, sizeof(unsigned char) * cols * rows));

	std::cout << "Copying DEM from host to device.\n";
	// Copy host data over to device.
	checkCudaErrors(cudaMemcpy(d_heightMap, DEM, sizeof(float) * rows * cols, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_lum, lum, sizeof(unsigned char) * rows * cols, cudaMemcpyHostToDevice));


	// Hazard weightings.
	sWeight = slopeWeight;
	rWeight = roughWeight;
	shWeight = shadowWeight;

}

////////////////////////
// --- DESTRUCTOR --- //
////////////////////////

__host__ gpuHazardClass::~gpuHazardClass()
{
	checkCudaErrors(cudaFree(d_heightMap));
	checkCudaErrors(cudaFree(d_gaussMap));
	checkCudaErrors(cudaFree(d_slope));
	checkCudaErrors(cudaFree(d_rough));
	checkCudaErrors(cudaFree(d_preGrassScore));		
	checkCudaErrors(cudaFree(d_score));
	checkCudaErrors(cudaFree(d_lum));
}


/////////////////////////////
// --- MAPPING METHODS --- //
/////////////////////////////

__host__ void gpuHazardClass::mapSlope()
{
	// Kernel dimensions.
	const dim3 blockSize(16, 16, 1);
	const dim3 gridSize((cols / 16 + 1), (rows / 16 + 1), 1);

	// Call kernel.
	slopeKernel << <gridSize, blockSize >> >(d_gaussMap, rows, cols, d_slope);			// Using gaussian blurred map instead.
	// Exit if the kernel call failed.
	kernelCheck();
}


__host__ void gpuHazardClass::mapRough()
{
	// Kernel dimensions.
	const dim3 blockSize(16, 16, 1);
	const dim3 gridSize((cols / 16 + 1), (rows / 16 + 1), 1);

	// Call kernel.
	roughnessKernel << <gridSize, blockSize >> >(d_gaussMap, rows, cols, d_rough);		// Using gaussian blurred map instead.
	// Exit if the kernel call failed.
	kernelCheck();
}


// Compute a hazard map, uses max finding kernel.
__host__ void gpuHazardClass::mapHazards()
{
	float *d_slopeMax, *d_roughMax;
	float *d_blockMaxSlope, *d_blockMaxRough;
	const int size = rows*cols;

	// Allocate required memory
	checkCudaErrors(cudaMalloc(&d_slopeMax, sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_roughMax, sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_blockMaxSlope, sizeof(float) * 256));
	checkCudaErrors(cudaMalloc(&d_blockMaxRough, sizeof(float) * 256));


	// Determine the max slope.
	maxFirst << <256, 256 >> >(d_slope, d_blockMaxSlope, size);
	// Exit if the kernel call failed.					// Inline or function this...
	kernelCheck();

	maxSecond << <1, 256 >> >(d_blockMaxSlope, d_slopeMax);
	// Exit if the kernel call failed.
	kernelCheck();


	// Determine the max roughness.
	maxFirst << <256, 256 >> >(d_rough, d_blockMaxRough, size);
	// Exit if the kernel call failed.	
	kernelCheck();

	maxSecond << <1, 256 >> >(d_blockMaxRough, d_roughMax);
	// Exit if the kernel call failed.
	kernelCheck();


	// Kernel dimensions.
	const dim3 blockSize(16, 16, 1);
	const dim3 gridSize((cols / 16 + 1), (rows / 16 + 1), 1);

	// Call kernel.
	hazardKernel << <gridSize, blockSize >> >(d_slope, d_rough, d_lum, rows, cols, d_preGrassScore, d_slopeMax, d_roughMax, sWeight, rWeight, shWeight);
	// Exit if the kernel call failed.
	kernelCheck();

	// de-allocating temporary pointers
	cudaFree(d_slopeMax);
	cudaFree(d_roughMax);
	cudaFree(d_blockMaxSlope);
	cudaFree(d_blockMaxRough);
}

///////////////////////
// --- GRASSFIRE --- //
///////////////////////

__host__ void gpuHazardClass::grassfire()
{
	int maxThreads = 512;
	dim3 bDim(maxThreads);
	// Number of blocks is the number of threads required to cover all rows/cols.
	int numBlocks = max(rows, cols) / maxThreads;
	
	if (max(rows, cols) % maxThreads > 0)
		numBlocks++;

	dim3 gDim(numBlocks);

	// Calling each of the 4 kernels in turn, sync at each stage.
	rowRasterKernel << <gDim, bDim >> >(d_score, rows, cols);
	// Exit if the kernel call failed.
	kernelCheck();

	checkCudaErrors(cudaDeviceSynchronize());

	colRasterKernel << <gDim, bDim >> >(d_score, rows, cols);
	// Exit if the kernel call failed.
	kernelCheck();

	checkCudaErrors(cudaDeviceSynchronize());

	rowAntiRasterKernel << <gDim, bDim >> >(d_score, rows, cols);
	// Exit if the kernel call failed.
	kernelCheck();

	checkCudaErrors(cudaDeviceSynchronize());

	colAntiRasterKernel << <gDim, bDim >> >(d_score, rows, cols);
	// Exit if the kernel call failed.
	kernelCheck();
}

////////////////////////////
// --- LINKING METHOD --- //
////////////////////////////

// Calls each stage of hazard map generation in turn, returns error code if data wasnt initialized.
__host__ int gpuHazardClass::createHazardMap()
{
	// Check the class has been initialized.
	if (rows == 0 || cols == 0)
		return 1;
	if (d_heightMap == nullptr)
		return 2;

	std::cout << "Applying 5x5 gaussian filter to DEM.\n";
	gaussFilter();

	std::cout << "Creating slope map.\n";
	mapSlope();

	std::cout << "Creating roughness map.\n";
	mapRough();

	std::cout << "Creating hazard map.\n";
	mapHazards();

	// New
	copyScores();
	// End New

	std::cout << "Applying grassfire algorithm.\n";
	grassfire();

	return 0;
}

////////////////////////////
// --- RETURN METHODS --- //
////////////////////////////

// Slope map.
__host__ float* gpuHazardClass::getSlopeMap()
{
	float *temp = new float[rows*cols];
	checkCudaErrors(cudaMemcpy(temp, d_slope, sizeof(float)*rows*cols, cudaMemcpyDeviceToHost));
	return temp;
}

// Roughness map.
__host__ float* gpuHazardClass::getRoughMap()
{
	float *temp = new float[rows*cols];
	checkCudaErrors(cudaMemcpy(temp, d_rough, sizeof(float)*rows*cols, cudaMemcpyDeviceToHost));
	return temp;
}

// Pre-Grassfire hazard map.
__host__ int* gpuHazardClass::getPreGrassMap()
{
	int *temp = new int[rows*cols];
	checkCudaErrors(cudaMemcpy(temp, d_preGrassScore, sizeof(int)*rows*cols, cudaMemcpyDeviceToHost));
	return temp;
}

// Hazard map.
__host__ int* gpuHazardClass::getHazardMap()
{
	int *temp = new int[rows*cols];
	checkCudaErrors(cudaMemcpy(temp, d_score, sizeof(int)*rows*cols, cudaMemcpyDeviceToHost));
	return temp;
}


// Copy the data from preGrassScores to Scores.
void gpuHazardClass::copyScores()
{
	checkCudaErrors(cudaMemcpy(d_score, d_preGrassScore, sizeof(float)*rows*cols, cudaMemcpyDeviceToDevice));
}


__host__ void gpuHazardClass::kernelCheck()
{
	if (cudaSuccess != cudaPeekAtLastError())
	{
		std::cout << "Kernel call failed.\n";
		std::cout << cudaGetErrorString(cudaPeekAtLastError());
		exit(-1);		// Exit the program.
	}
}


// Gaussian Filter
__host__ void gpuHazardClass::gaussFilter()
{

	// Kernel dimensions.
	const dim3 blockSize(16, 16, 1);
	const dim3 gridSize((cols / 16 + 1), (rows / 16 + 1), 1);

	// Call kernel.
	gaussKernel << <gridSize, blockSize >> >(d_heightMap, d_gaussMap, rows, cols);
	// Exit if the kernel call failed.
	kernelCheck();
}
