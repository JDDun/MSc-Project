#include "siteLocator.h"

//////////////////////////////////////
// --- Constructor / Destructor --- //
//////////////////////////////////////

// Constructor sets all attributes to default values - equal weightings, using gpu, no intermediate output maps. Establishes PANGU connection.
siteLocator::siteLocator() : slopeWeight(1), roughWeight(1), shadowWeight(1), use_gpu(1), verbose(0), numSites(5), score(nullptr) {
	cout << "\nConnecting to PANGU viewer.\n";
	viewer.PAN_connect();
	cout << "PANGU viewer connection established.\n\n";
}

// Destructor de-allocates pointers and Mats.
siteLocator::~siteLocator()
{
	delete[]score;
	landing_region.release();
}


/////////////////////////////////
// --- Core Linking Method --- //
/////////////////////////////////

void siteLocator::findSites(string labelFile, string demFile)
{
	// Read the DEM and DEM label files.
	getDEM(labelFile, demFile);

	// Display the parameter selection dialog.
	UI.show();

	// Start an event loop. Exit if the UI fails to launch.
	if (Fl::run() != 0)
	{
		std::cout << "Error launching user interface, exiting program.\n";
		exit(1);
	}

	UI.getParams(use_gpu, verbose, slopeWeight, roughWeight, shadowWeight, numSites);
	
	// Find the largest dimension of the landing region in order to set the viewpoint for taking images of it.
	int maxDim = max(heightMap.getCols(), heightMap.getRows());

	// Save an image of the full landing region to file.
	int error = viewer.get_landing_region(maxDim, heightMap.getRes());
	if (error)
	{
		std::cout << "Error retrieving landing region image, exiting program.\n";
		exit(2);
	}
	landing_region = cv::imread("landing_region.ppm", CV_8UC1);

	if (use_gpu)
		gpuHazardMap();
	else
		cpuHazardMap();

	// Filter the hazard map to find landing sites.
	siteFilter filter(score, heightMap.getCols(), heightMap.getRows(), numSites, landing_region.data);
	filter.findSites();			// Find the local minimums.
	filter.filterSites();		// Filter by shadow/distance.
	// Get the selected landing sites.
	vector<landingSite> chosen_sites = filter.getSites();

	// Get images of each site from PANGU.
	cout << "Saving images of the landing sites.\n";
	getSiteImages(chosen_sites, maxDim);

	// Disconnect from PANGU viewer.
	cout << "\n\nDisconnecting from PANGU viewer.\n";
	viewer.PAN_disconnect();
	cout << "PANGU viewer connection terminated.\n";

	// Display results.
	drawSites(landing_region, chosen_sites);
}



///////////////////////////
// --- Other Methods --- //
///////////////////////////

// Parse the DEM label file and data file.
void siteLocator::getDEM(string labelFile, string demFile)
{
	bool error = heightMap.readLabel(labelFile);

	if (error)
	{
		std::cout << "Failed to read label file, exiting program.\n";
		exit(-1);
	}

	error = heightMap.readDEM(demFile);

	if (error)
	{
		std::cout << "Failed to read DEM file, exiting program.\n";
		exit(-1);
	}
}

void siteLocator::getSiteImages(vector<landingSite> sites, int maxDim)
{
	for(int i = 0; i < numSites; i++)
	{
		int error = viewer.get_site(sites[i].getX(), sites[i].getY(), maxDim, heightMap.getRes(), i);
		if (error)
		{
			std::cout << "Error retrieving the landing site images, exiting program.\n";
			exit(3);
		}
	}
}



//////////////////////////////////////
// --- Hazard Mapping Functions --- //
//////////////////////////////////////

// Using GPU.
void siteLocator::gpuHazardMap()
{
	int rows = heightMap.getRows();
	int cols = heightMap.getCols();

	cout << "Creating gpuHazard class.\n";

	// Cuda timing events.
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Start timing.
	cudaEventRecord(start);

	gpuHazardClass gpuMap(heightMap.getData(), rows, cols, landing_region.data, slopeWeight, roughWeight, shadowWeight);

	cout << "Creating hazard map.\n";
	int error = gpuMap.createHazardMap();

	if (error != 0)
	{
		cout << "\n\nExiting with code: " << error << endl;
		exit(error);
	}

	// Stop timing.
	cudaEventRecord(stop);

	// Record time taken and display in console.
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout << "Time taken for GPU computation: " << milliseconds << "ms\n";

	// Retrieve the hazard scores.
	score = gpuMap.getHazardMap();

	// if intermediate maps have been requested with verbose mode.
	if (verbose)
	{
		// slope map.
		float *slope = gpuMap.getSlopeMap();
		imageMap<float> slopeMap(slope, rows, cols);
		slopeMap.draw("Slope Map");

		// roughness map.
		float *rough = gpuMap.getRoughMap();
		imageMap<float> roughMap(rough, rows, cols);
		roughMap.draw("Roughness Map");

		// hazard map.
		int *preGrass = gpuMap.getPreGrassMap();
		imageMap<int> preGrassScoreMap(preGrass, rows, cols);
		preGrassScoreMap.draw("Pre-Grassfire Hazard Map");

		// final map post-grassfire.
		int *hazardMap = gpuMap.getHazardMap();
		imageMap<int> scoreMap(score, rows, cols);
		scoreMap.draw("Post-Grassfire Hazard Map");

		// De-allocating temporary pointers.
		delete[]slope;
		delete[]rough;
		delete[]preGrass;
		delete[]hazardMap;
	}
}

// Using CPU
void siteLocator::cpuHazardMap()
{
	int rows = heightMap.getRows();
	int cols = heightMap.getCols();

	cout << "Calling CPU hazard functions.\n";

	// Start CPU timer.
	clock_t start, stop;
	start = clock();

	cpuHazardClass cpuMap(heightMap.getData(), rows, cols, landing_region.data, slopeWeight, roughWeight, shadowWeight);

	int error = cpuMap.createHazardMap();

	if (error != 0)
	{
		cout << "\n\nExiting with code: " << error << endl;
		exit(error);
	}

	// Stop timer, display time taken.
	stop = clock();
	clock_t milliseconds = float(stop - start) / CLOCKS_PER_SEC * 1000;

	cout << "Time taken for CPU computation: " << milliseconds << "ms\n";

	// Retrieve the hazard scores.
	score = cpuMap.getHazardMap();

	// if intermediate maps have been requested with verbose mode.
	if (verbose)
	{
		// slope map.
		float *slope = cpuMap.getSlopeMap();
		imageMap<float> slopeMap(slope, rows, cols);
		slopeMap.draw("Slope Map");

		// roughness map.
		float *rough = cpuMap.getRoughMap();
		imageMap<float> roughMap(rough, rows, cols);
		roughMap.draw("Roughness Map");

		// hazard map.
		int *preGrass = cpuMap.getPreGrassMap();
		imageMap<int> preGrassScoreMap(preGrass, rows, cols);
		preGrassScoreMap.draw("Pre-Grassfire Hazard Map");

		// final map post-grassfire.
		int *hazardMap = cpuMap.getHazardMap();
		imageMap<int> scoreMap(score, rows, cols);
		scoreMap.draw("Post-Grassfire Hazard Map");


		delete[]slope;
		delete[]rough;
		delete[]preGrass;
		delete[]hazardMap;
	}
}
