#include "kernels.cuh"

//////////////
/// SLOPE ////
//////////////
// Derives the slope at a given point through comparison with neighbours.
__global__ void slopeKernel(float *d_DEM, const int rows, const int cols, float *d_deg)
{
	// Setting x,y coords from thread index
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;


	// Return if either index is out of bounds.
	if (x >= cols || y >= rows)
		return;

	// Absolute 1-D index.
	int absIdx = x + y*cols;
	float centre = d_DEM[absIdx];
	float dist = 0.0f, diff = 0.0f;

	// Variable to hold index of next cell to compare with.
	int u, v, absUV;
	//int count = 0;
	float temp;

	for (int i = -1; i <= 1; i++)
	{
		u = x + i;
		if (u >= 0 && u < cols)
		{
			for (int j = -1; j <= 1; j++)
			{
				v = y + j;
				if (v >= 0 && v < rows)
				{
					absUV = u + v * rows;
					// Iterate the running tally of valid comparisons.
					//count++;

					// Calculate distance between the two points.
					temp = fabsf(centre - d_DEM[absUV]);
					// if new maximum difference in height, then set it as the new diff and calculate dist using trig.
					if (temp > diff)
					{
						diff = temp;
						dist = sqrtf(float(abs(i) + abs(j)));
					}
				}
			}
		}
	}
	// Radians
	float slopeDeg = atan(diff / dist);

	d_deg[absIdx] = slopeDeg;
}

//////////////
// ROUGHNESS /
//////////////
// Using Root Mean Square as a measure for local roughness.
__global__ void roughnessKernel(float *d_DEM, const int rows, const int cols, float *d_rough)
{
	// Setting x,y coords from thread index
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;


	// Return if either index is out of bounds.
	if (x >= cols || y >= rows)
		return;

	// Absolute 1-D index.
	int absIdx = x + y*cols;

	// Height of centre cell.
	float centre = d_DEM[absIdx];

	int u, v, absUV;		// x,y,1-D index for neighbouring cells.

	int count = 0;
	float avg = 0.0f;

	for (int i = -1; i <= 1; i++)
	{
		u = x + i;
		if (u >= 0 && u < cols)
		{
			for (int j = -1; j <= 1; j++)
			{
				v = y + j;
				if (v >= 0 && v < rows && !(i == 0 && j == 0))		// Discount the centre cell.
				{
					absUV = u + v * rows;

					count++;

					// Calculate distance between the two points.
					avg += powf(d_DEM[absUV] - centre, 2);
				}
			}
		}
	}

	if (count != 0)
	{
		avg = sqrtf(avg / float(count));
		d_rough[absIdx] = avg;
	}
}

//////////////
/// HAZARD ///
//////////////
// Derives hazard score.
// Shadow is factored in within this kernel, trivial.
__global__ void hazardKernel(float *d_slope, float *d_rough, unsigned char *d_source, const int rows, const int cols, int *d_score,
	float* slopeMax, float* roughMax, float slopeWeight, float roughWeight, float shadowWeight)
{
	const int hazardMax = 1000;			// Arbitary maximum hazard score.

	// Setting x,y coords from thread index
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;


	// Return if either index is out of bounds.
	if (x >= cols || y >= rows)
		return;

	// Absolute 1-D index.
	int absIdx = x + y*cols;

	// Roughness & Slope
	float r = d_rough[absIdx];
	float s = d_slope[absIdx];
	//unsigned int sh = unsigned int(255 - d_source[absIdx]);

	// Testing
	 int sh = 0;
	 if (d_source[absIdx] < 15)
	   sh = 1;
	// End Testing

	float normalScore = ((slopeWeight * s / slopeMax[0]) + (roughWeight * r / roughMax[0]) + (shadowWeight * sh)) / (slopeWeight + roughWeight + shadowWeight);
	// Scale the normalized score and round the resulting float.
	normalScore = rintf(normalScore * hazardMax);

	// Convert the float to an integer and store in array.
	d_score[absIdx] = float2int(normalScore);
}

///////////////
// GRASSFIRE //
///////////////

//////////////// Raster /////////////////
__global__ void rowRasterKernel(int *d_score, const int rows, const int cols)
{
	// Fix y-coordinate.
	int y = threadIdx.x + blockIdx.x * blockDim.x;

	if (y < rows - 1)
	{
		// max(current, left cell - 1)
		for (int x = 1; x < cols; x++)
			d_score[x + y * cols] = max(d_score[x + y * cols], d_score[x - 1 + y * cols] - 1);
	}
}

__global__ void colRasterKernel(int *d_score, const int rows, const int cols)
{
	// Fix x-coordinate.
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	if (x < cols - 1)
	{
		// max(current, cell above - 1)
		for (int y = 1; y < rows; y++)
			d_score[x + y*cols] = max(d_score[x + y*cols], d_score[x + (y - 1) * cols] - 1);
	}
}


///////////// Anti - Raster /////////////
__global__ void rowAntiRasterKernel(int *d_score, const int rows, const int cols)
{
	// Fix y-coordinate.
	int y = threadIdx.x + blockIdx.x * blockDim.x;

	if (y < rows - 1)
	{
		{
			// max(current, right cell - 1)
			for (int x = cols - 2; x >= 0; x--)
				d_score[x + y * cols] = max(d_score[x + y * cols], d_score[(x + 1) + y * cols] - 1);
		}
	}
}


__global__ void colAntiRasterKernel(int *d_score, const int rows, const int cols)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	if (x < cols - 1)
	{
		// max(current, cell below - 1)
		for (int y = rows - 2; y >= 0; y--)
			d_score[x + y*cols] = max(d_score[x + y*cols], d_score[x + (y + 1) * cols] - 1);
	}
}

/////////////////
// Max kernels //
/////////////////


__global__ void maxFirst(float *in_array, float *blockMax, const int n)
{
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = stride;

	// Shared memory array for the block. Holds each threads maximum.
	__shared__ float threadMax[256];

	// Set initial max to the first element for each thread.
	float temp = in_array[index];

	// Each thread performs a comparison until all elements have been exhausted.
	while (index + offset < n)
	{
		// If new max store in temp variable.
		temp = fmaxf(temp, in_array[index + offset]);

		// Shift to the next element to compare.
		offset += stride;
	}

	// Set max for this thread in the shared array.
	threadMax[threadIdx.x] = temp;

	__syncthreads();


	// Find the max of the block through reduction.
	// At each step half number of threads performing a comparison until only thread 0 remains, with the final result.
	unsigned int i = blockDim.x / 2;
	while (i != 0)
	{
		if (threadIdx.x < i)
			threadMax[threadIdx.x] = fmaxf(threadMax[threadIdx.x], threadMax[threadIdx.x + i]);

		__syncthreads();
		i /= 2;
	}
	// threadMax[0] now contains the block maximum.
	__syncthreads();

	// The first thread writes the final result to the output array of block maximums.
	if (threadIdx.x == 0)
		blockMax[blockIdx.x] = threadMax[0];
}

// Final stage of max finding kernels.
__global__ void maxSecond(float *blockMax, float* max)
{
	// Perform a reduction on the input array of block maximums.
	unsigned int i = blockDim.x / 2;
	while (i != 0)
	{
		if (threadIdx.x < i)
			blockMax[threadIdx.x] = fmaxf(blockMax[threadIdx.x], blockMax[threadIdx.x + i]);

		__syncthreads();
		i /= 2;
	}

	__syncthreads();

	// Write the final result.
	if (threadIdx.x == 0)
		max[0] = blockMax[0];
}


//////////////////
// Gauss Kernel //
//////////////////

__global__ void gaussKernel(float* in_array, float* out_array, const int rows, const int cols)
{

	// Hardcoded 5x5 Gaussian filter weights.
	float weights[25]
	{
		0.003765, 0.015019, 0.023792, 0.015019, 0.003765,
			0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
			0.023792, 0.094907, 0.150342, 0.094907, 0.023792,
			0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
			0.003765, 0.015019, 0.023792, 0.015019, 0.003765
	};
	
	__shared__ float filter[25];
	
	if (threadIdx.x == 0)
	{
		for (int i = 0; i < 25; i++)
		{
			filter[i] = weights[i];
		}
	}
	
	__syncthreads();


	// Setting x,y coords from thread index
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	
	// Return if either index is out of bounds.
	if (x >= cols || y >= rows)
		return;

	// Absolute 1-D index.
	int absIdx = x + y*cols;

	int filterWidth = 5;	// Width of the gaussian filter, hardcoded to 5.
	int half = filterWidth / 2;			
	float blur = 0.f;								// will contained blurred value
	int	width = cols - 1;
	int	height = rows - 1;

	for (int i = -half; i <= half; ++i)					// rows
	{
		for (int j = -half; j <= half; ++j)				// columns
		{
			// Clamp filter to the image border
			int		w = min(max(x + j, 0), width);
			int		h = min(max(y + i, 0), height);

			// Blur is a product of current pixel value and weight of that pixel.
			// Remember that sum of all weights equals to 1, so we are averaging sum of all pixels by their weight.
			int		idx = w + cols * h;											// current pixel index
			float	value = in_array[idx];

			idx = (i + half) * filterWidth + j + half;
			float	weight = filter[idx];

			blur += value * weight;
		}
	}

	out_array[absIdx] = blur;
	
}
