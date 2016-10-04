#pragma once

#include <string>
#include <iostream>
#include <fstream>

class DEM
{
private:
	float* data;
	int rows, cols, horRes;

public:
	DEM();
	DEM(std::string, std::string);

	bool readDEM(std::string);
	bool readLabel(std::string);

	float* getData();
	int getRows();
	int getCols();
	int getRes();
};

float ReverseFloat(const float);
