#pragma once

#include <iostream>
#include "imgMap.h"		// To import the openCV includes.
#include <math.h>

// Class for computing hazard maps.
class cpuHazardClass
{
private:
	int rows, cols;		// Dimensions of the region.
	int *score, *preGrassScore;			// Pointers to the hazard scores.
	float *heightMap, *gaussMap, *slope, *rough;		// Pointers to the input DEM heightmap, Gauss filtered DEM, slope map and roughness map.
	unsigned char *lum;						// Pixel brightness values from an image of the site map.

	float sWeight, rWeight, shWeight;			// Weightings for each hazard type.

public:
	cpuHazardClass();		// Avoid using this - in practice the linking function will check the class has been initialized properly.
	cpuHazardClass(float *DEM, const int ydim, const int xdim, unsigned char *image_lum, const float slopeWeight, const float roughWeight, const float shadowWeight);

	~cpuHazardClass();		// Destructor deletes all the pointers and the opencv Mat.

	void mapSlope();		// Compute a slope map, first stage.
	void mapRough();		// Compute a roughness map, second stage.
	void mapHazards();		// Compute a hazard map, third stage.

	void copyScores();		// Copies the contents of preGrassScores to scores!

	void grassRowRaster();	// Apply grassfire algorithm in four passes.
	void grassColRaster();
	void grassRowAntiRaster();
	void grassColAntiRaster();

	int createHazardMap();	// Linking function, asserts everything is prepared and calls each stage in turn.

	// New
	void gaussFilter();

	// Get methods for each map.
	// Each copies the array contents to a new pointer and returns it.
	int* getPreGrassMap();  // Hazard map before grassfire.
	int* getHazardMap();    // Hazard map after grassfire.
	float* getSlopeMap();	// Slope map.
	float* getRoughMap();	// Roughness map.

};
