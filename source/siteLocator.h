#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <ctime>

#include "DEM.h"
#include "imgMap.h"
#include "grassFire.h"
#include "landingSite.h"
#include "demo.h"
#include "PAN_interface.h"
#include "siteFilter.h"

#include "cpuHazardClass.h"
#include "gpuHazardClass.cuh"

#include "paramSelect.h"

class siteLocator
{
private:
	float slopeWeight, roughWeight, shadowWeight;
	bool use_gpu, verbose;
	int numSites;

	int *score;

	paramSelect UI;
	DEM heightMap;

	PAN_interface viewer;

	cv::Mat landing_region;

public:
	siteLocator();
	~siteLocator();

	void findSites(string labelFile, string demFile);

	void getDEM(string labelFile, string demFile);

	void cpuHazardMap();
	void gpuHazardMap();

	void getSiteImages(vector<landingSite> sites, int maxDim);
};
