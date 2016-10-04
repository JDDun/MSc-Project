#include "cpuHazardClass.h"

/////////////////////////
// --- CONSTRUCTOR --- //
/////////////////////////

// Default constructor, unused in practice.
cpuHazardClass::cpuHazardClass() : rows(0), cols(0), score(nullptr), preGrassScore(nullptr), heightMap(nullptr),
								   gaussMap(nullptr), slope(nullptr), rough(nullptr), lum(nullptr), sWeight(1),
								   rWeight(1), shWeight(1) {}


// Typical constructor.
cpuHazardClass::cpuHazardClass(float *DEM, const int ydim, const int xdim, unsigned char *image_lum, const float slopeWeight, const float roughWeight, const float shadowWeight)
{
	rows = ydim;
	cols = xdim;

	heightMap = DEM;
	lum = image_lum;

	// Reserve memory at each pointer.
	gaussMap = new float[rows*cols];
	score = new int[rows*cols];
	preGrassScore = new int[rows*cols];
	slope = new float[rows*cols];
	rough = new float[rows*cols];

	sWeight = slopeWeight;
	rWeight = roughWeight;
	shWeight = shadowWeight;
}

////////////////////////
// --- DESTRUCTOR --- //
////////////////////////

// De-allocate all the pointers and Mat once the object leaves scope.
cpuHazardClass::~cpuHazardClass()
{
	delete[]score;
	delete[]heightMap;
	delete[]gaussMap;
	delete[]slope;
	delete[]rough;
	delete[]preGrassScore;
}

/////////////////////////////
// --- MAPPING METHODS --- //
/////////////////////////////

void cpuHazardClass::mapSlope()
{
	int x, y, u, v, absUV;

	// Loop parallelization setup.
	const int size = rows * cols;
#pragma loop(hint_parallel(0))
#pragma loop(ivdep)
	for (int n = 0; n < size; n++)
	{
		// The current cell.
		float centre = gaussMap[n];
		// Calculate x,y coordinates for current cell.
		x = n % cols;
		y = n / cols;

		float dist = 0.0f;
		float diff = 0.0f;

		// Run through neighbouring cells.
		for (int i = -1; i <= 1; i++)
		{
			u = x + i;
			// Verify valid x-coordinate.
			if (u >= 0 && u < cols)
			{
				for (int j = -1; j <= 1; j++)
				{
					v = y + j;
					// Verify valid y-coordinate.
					if (v >= 0 && v < rows)
					{
						absUV = u + v * rows;
						dist = sqrt(abs(i) + abs(j));
						float temp = abs(centre - gaussMap[absUV]) / dist;		// Difference in height divided by the distance to between the cells.

						// New max difference.
						if (temp > diff)
							diff = temp;
					}
				}
			}
		}
		// Compute slope in radians.
		slope[n] = atan(diff);
	}
}


void cpuHazardClass::mapRough()
{
	int x, y, u, v, absUV;

	// Loop parallelization setup.
	const int size = rows * cols;
#pragma loop(hint_parallel(0))
#pragma loop(ivdep)
	for (int n = 0; n < size; n++)
	{
		// The current cell.
		float centre = gaussMap[n];
		// Calculate x,y coordinates for current cell.
		x = n % cols;
		y = n / cols;

		float count = 0;		// Number of neighbouring cells.
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

						// Add squared height difference to running average.
						avg += pow(gaussMap[absUV] - centre, 2);
					}
				}
			}
		}
		if (count != 0)
		{
			avg = sqrt(avg / float(count));
			rough[n] = avg;
		}
	}
}


void cpuHazardClass::mapHazards()
{
	const int hazardMax = 1000;			// Arbitrary maximum hazard score.

	// Determine the max values for roughness and slope for normalisation.
	std::cout << "Determining maximum roughness & slope.\n";
	float slopeMax = slope[0], roughMax = rough[0];


	for (int i = 1; i < rows * cols; i++)
	{
		if (rough[i] > roughMax)
			roughMax = rough[i];
		if (slope[i] > slopeMax)
			slopeMax = slope[i];
	}
	std::cout << "\n\nSlope Max: " << slopeMax << "\nRough Max: " << roughMax << "\n\n";

	// Loop parallelization setup.
	const int size = rows * cols;
#pragma loop(hint_parallel(0))
#pragma loop(ivdep)
	for (int n = 0; n < size; n++)
	{
		// Roughness & Slope of current cell.
		float r = rough[n];
		float s = slope[n];

		//unsigned int sh = unsigned int(255 - lum[n]);	// Inverted because shadowed areas have LOWER intensity values.
		int sh = 0;
		if (lum[n] < 15)
			sh = 1;

		float normalScore = ((sWeight * s / slopeMax) + (rWeight * r / roughMax) + (shWeight * sh)) / (sWeight + rWeight + shWeight);
		
		// Scale the normalized score and truncate the resulting float.
		preGrassScore[n] = int(normalScore * hazardMax);
	}
}

///////////////////////
// --- GRASSFIRE --- //
///////////////////////

// Compare left.
void cpuHazardClass::grassRowRaster()
{
	// Loop parallelization setup.
	const int size = rows;
#pragma loop(hint_parallel(0))			// Number of threads to use - max possible.
#pragma loop(ivdep)						// Explicitly stating there is no data dependency.
	for (int y = 0; y < size; y++)
	{
		for (int x = 1; x < cols; x++)
			score[x + y * cols] = std::max(score[x + y * cols], score[x - 1 + y * cols] - 1);
	}
}

// Compare up.
void cpuHazardClass::grassColRaster()
{
	// Loop parallelization setup.
	const int size = cols;
#pragma loop(hint_parallel(0))		
#pragma loop(ivdep)					
	for (int x = 0; x < size; x++)
	{
		for (int y = 1; y < rows; y++)
			score[x + y * cols] = std::max(score[x + y * cols], score[x + (y - 1) * cols] - 1);
	}
}

// Compare right.
void cpuHazardClass::grassRowAntiRaster()
{
	// Loop parallelization setup.
	const int size = rows - 1;
#pragma loop(hint_parallel(0))
#pragma loop(ivdep)
	for (int y = size; y >= 0; y--)
	{
		for (int x = cols - 2; x >= 0; x--)
			score[x + y * cols] = std::max(score[x + y * cols], score[(x + 1) + y * cols] - 1);
	}
}

// Compare down.
void cpuHazardClass::grassColAntiRaster()
{

	// Loop parallelization setup.
	const int size = cols - 1;
#pragma loop(hint_parallel(0))
#pragma loop(ivdep)
	for (int x = size; x >= 0; x--)
	{
		for (int y = rows - 2; y >= 0; y--)
			score[x + y*cols] = std::max(score[x + y*cols], score[x + (y + 1) * cols] - 1);
	}
}

//////////////////////////////
// --- LINKING FUNCTION --- //
//////////////////////////////

int cpuHazardClass::createHazardMap()
{
	if (rows == 0 || cols == 0)
		return 1;
	if (heightMap == nullptr)
		return 2;

	std::cout << "Applying 5x5 gaussian filter.\n";
	gaussFilter();

	std::cout << "Creating slope map.\n";
	mapSlope();

	std::cout << "Creating roughness map.\n";
	mapRough();

	std::cout << "Creating hazard map.\n";
	mapHazards();

	// Copy the scores over from preGrass to scores!
	copyScores();

	std::cout << "Applying grassfire algorithm.\n";
	grassRowRaster();
	grassColRaster();
	grassRowAntiRaster();
	grassColAntiRaster();

	std::cout << "CPU computation of hazard scores complete.\n";
	return 0;	// Return 0 on success.
}

void cpuHazardClass::copyScores()
{
	for (int i = 0; i < rows*cols; i++)
		score[i] = preGrassScore[i];
}


/////////////////////////
// --- GET METHODS --- //
/////////////////////////

// Returns a pointer to a copy of the pre-grassfire hazard map.
int* cpuHazardClass::getPreGrassMap()
{
	int *temp = new int[rows*cols];
	for (int i = 0; i < rows*cols; i++)
		temp[i] = preGrassScore[i];

	return temp;
}

// Return a pointer to a copy of the hazard scores post-grassfire.
int* cpuHazardClass::getHazardMap()
{
	int *temp = new int[rows*cols];
	for (int i = 0; i < rows*cols; i++)
		temp[i] = score[i];

	return temp;
}

// Returns a pointer to a copy of the slope values.
float* cpuHazardClass::getSlopeMap()
{
	float *temp = new float[rows*cols];
	for (int i = 0; i < rows*cols; i++)
		temp[i] = slope[i];

	return temp;
}

// Returns a pointer to a copy of the roughness values.
float* cpuHazardClass::getRoughMap()
{
	float *temp = new float[rows*cols];
	for (int i = 0; i < rows*cols; i++)
		temp[i] = rough[i];
	
	return temp;
}


///////////////////////////
// --- Gaussian Blur --- //
///////////////////////////

//void cpuHazardClass::gaussFilter()
//{
//	// Hardcoded 5x5 Gaussian filter weights.
//	float filter[25] =
//	{
//		0.003765, 0.015019, 0.023792, 0.015019, 0.003765,
//		0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
//		0.023792, 0.094907, 0.150342, 0.094907, 0.023792,
//		0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
//		0.003765, 0.015019, 0.023792, 0.015019, 0.003765
//	};
//
//	const int half = 2;
//	const int width = cols - 1;
//	const int height = rows - 1;
//
//	// Apply blur
//
//	for (int r = 0; r < rows; ++r)
//	{
//		for (int c = 0; c < cols; ++c)
//		{
//			float blur = 0.f;
//
//			// Average pixel color summing up adjacent pixels.
//			for (int i = -half; i <= half; ++i)
//			{
//				for (int j = -half; j <= half; ++j)
//				{
//					// Clamp filter to the image border
//					int h = std::min(std::max(r + i, 0), height);
//					int w = std::min(std::max(c + j, 0), width);
//
//					// Blur is a product of current pixel value and weight of that pixel.
//					// Remember that sum of all weights equals to 1, so we are averaging sum of all pixels by their weight.
//					
//					int	index = w + cols * h;				// current pixel index									
//
//					float value = heightMap[index];			// Height at this pixel.
//
//					index = (i + half) * 5 + j + half;		// Position in the gaussian kernel.
//					float weight = filter[index];			// Weight at this position.
//
//					blur += value * weight;					// Add the weighted height to the sum.
//				}
//			}
//			int index = c + r * cols;
//			gaussMap[index] = blur;
//		}
//	}
//}


void cpuHazardClass::gaussFilter()
{
	// Hardcoded 5x5 Gaussian filter weights.
	float filter[25] =
	{
		0.003765, 0.015019, 0.023792, 0.015019, 0.003765,
		0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
		0.023792, 0.094907, 0.150342, 0.094907, 0.023792,
		0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
		0.003765, 0.015019, 0.023792, 0.015019, 0.003765
	};

	const int half = 2;
	const int width = cols - 1;
	const int height = rows - 1;

	// Apply blur
	const int size = cols * rows;
#pragma loop(hint_parallel(0))
#pragma loop(ivdep)

	for (int n = 0; n < size; n++)
	{
		float blur = 0.f;

		int r = n / cols;
		int c = n % cols;

		// Average pixel color summing up adjacent pixels.
		for (int i = -half; i <= half; ++i)
		{
			for (int j = -half; j <= half; ++j)
			{
				// Clamp filter to the image border
				int h = std::min(std::max(r + i, 0), height);
				int w = std::min(std::max(c + j, 0), width);

				// Blur is a product of current pixel value and weight of that pixel.
				// Remember that sum of all weights equals to 1, so we are averaging sum of all pixels by their weight.

				int	index = w + cols * h;				// current pixel index									

				float value = heightMap[index];			// Height at this pixel.

				index = (i + half) * 5 + j + half;		// Position in the gaussian kernel.
				float weight = filter[index];			// Weight at this position.

				blur += value * weight;					// Add the weighted height to the sum.
			}
		}
		int index = c + r * cols;
		gaussMap[index] = blur;

	}
}
