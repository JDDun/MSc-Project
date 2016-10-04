#pragma once

#include "landingSite.h"

// Class to locate potential landing sites and apply filtering criteria.
class siteFilter
{
private:
	int *hazardPtr;							// Just a copy of the pointer - does not need de-allocated when siteFilter leaves scope.
	int rows, cols, numSites;
	unsigned char* luminance;
	float minDistance;
	unsigned char minLuminance;

	std::vector<landingSite> sites;			// Vector of all the local minimum hazard scores.
	std::vector<landingSite> filteredSites;	// Vector of the filtered local minimums, representing the best sites.
public:
	siteFilter(int*, int, int, int, unsigned char*);

	void findSites();		// Locate the local minimum hazard scores.
	void filterSites();		// Sort the hazard scores and find the best 5 which satisfy the filtering criteria - minimum distance, minimum luminance.

	std::vector<landingSite> getSites();		// Returns filteredSites vector.
};
