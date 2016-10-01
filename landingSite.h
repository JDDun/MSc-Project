#pragma once

#include <vector>
#include <algorithm>
#include "imgMap.h"		// openCV includes.
#include <math.h>

#define minDist 50
#define minLum 50

class landingSite
{
private:
	int score;	// Hazard score and 1-D index of a local minimum.
	//int index;

	int x, y;
public:
	landingSite() = default;
	landingSite(int, int, int);

	bool operator<(landingSite);
	int getX();
	int getY();
	int getScore();

	float dist(landingSite);
};

