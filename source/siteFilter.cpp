#include "siteFilter.h"

/////////////////////////
// --- Constructor --- //
/////////////////////////

siteFilter::siteFilter(int* hazardScores, int xdim, int ydim, int nSites, unsigned char* lum) : hazardPtr(hazardScores), rows(ydim), cols(xdim), numSites(nSites),
																								luminance(lum), minDistance(50), minLuminance(15) {}


/////////////////////
// --- Methods --- //
/////////////////////

// Locates the local minimum hazard scores - these are the best landing sites. Adds them to the vector.
void siteFilter::findSites()
{
	// Loop through the scores, compare each score to its (up to) 8 neighbouring cells.
	for (int i = 0; i < rows*cols; i++)
	{
		bool is_min = true;
		// Calculate x,y coordinates of i.
		int x = i % cols;
		int y = i / cols;

		for (int u = -1; u < 2; u++)
		{
			int xIdx = u + x;
			if ((xIdx) >= 0 && (xIdx) < cols)
			{
				for (int v = -1; v < 2; v++)
				{
					int yIdx = v + y;
					if ((yIdx) >= 0 && (yIdx) < rows)
					{
						if (hazardPtr[(x + u) + (y + v)*cols] < hazardPtr[i])
							is_min = false;
					}
				}
			}
		}
		// If i is a local minimum add to the vector.
		if (is_min)
		{
			landingSite newSite(hazardPtr[i], i, cols);
			sites.push_back(newSite);
		}
	}
}

// Sort and filter the landing site vector.
void siteFilter::filterSites()
{
	// Sort the vector of local minimums by hazard score.
	sort(sites.begin(), sites.end());

	int i = 0;
	// Keep searching for valid sites till numSites are found or the sites vector is exhausted.
	while ((filteredSites.size() < numSites) && (i < sites.size()))
	{
		int x = sites[i].getX();
		int y = sites[i].getY();


		bool valid = true;

		// Check that the landing site is not too close to the edge of the landing region.
		if ((x > cols * 0.9) || (x < cols * 0.1))
			valid = false;
		if ((y > rows * 0.9) || (y < rows * 0.1))
			valid = false;

		if (valid)
		{
			// Verify the area is not too dark.
			if (luminance[i] > minLuminance)
			{
				bool valid = true;

				// Run through the existing sites and calculate distance to them, if the new sites far enough away from all add to the vector.
				for (int j = 0; j < filteredSites.size(); j++)
				{
					float dist = sites[i].dist(filteredSites[j]);

					if (dist < minDistance)
						valid = false;
				}
				if (valid)
					filteredSites.push_back(sites[i]);
			}
		}
		i++;
	}
	// Inform user how many sites were located.
	printf("Located %d viable landing sites.\n", filteredSites.size());
}


// Return filtered sites vector.
std::vector<landingSite> siteFilter::getSites()
{
	std::vector<landingSite> chosen_sites = filteredSites;
	return chosen_sites;
}
