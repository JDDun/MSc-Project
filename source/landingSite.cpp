#include "landingSite.h"
//////////////////////
//// CLASS METHODS ///
//////////////////////

// Constructor
landingSite::landingSite(int val, int i, int xdim)
{
	score = val;
	x = i % xdim;
	y = i / xdim;
	//coordConvert(i, x, y, xdim);
}

// Less than overload, used to sort.
bool landingSite::operator<(landingSite x)
{
	if (score < x.score)
		return true;
	return false;
}
// Simple return methods.
int landingSite::getScore(){ return score; }
int landingSite::getX(){ return x; }
int landingSite::getY(){ return y; }

// Calculate the distance between two landingSites.
float landingSite::dist(landingSite other)
{
	return sqrt(pow(x - other.x, 2) + pow(y - other.y, 2));
}
