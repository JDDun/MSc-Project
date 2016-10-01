#include "DEM.h"


////////////////////////////////////////////////////////
////////////// CONSTRUCTORS //////////////////////
////////////////////////////////////////////////////////

// Default constructor
DEM::DEM() : rows(0), cols(0), horRes(0), data(nullptr) {}

// Constructor, takes two strings denoting filenames of the label and DEM, calls relevant functions.
DEM::DEM(std::string fileLabel, std::string fileDEM) : rows(0), cols(0), horRes(0), data(nullptr)
{
	bool error = 0;
	error = readLabel(fileLabel);

	if (!error)
	{
		error = error && readDEM(fileDEM);

		if (!error)
			std::cout << "xdim: " << cols << "\nydim: " << rows << "\nscale: " << horRes << "\n";
	}

	else
		std::cout << "failed to read correctly please check the input DEM and label file.\n";
}



////////////////////////////////////////////////////////
////////////// READING FILES ///////////////////////////
////////////////////////////////////////////////////////

// Parses a label (text) file to retrieve the number of rows, columns and resolution.
// Returns 0 on successful read of label file.
bool DEM::readLabel(std::string label)
{
	std::ifstream input(label);

	if (!input.is_open())
	{
		std::cout << "Failed to open file: " << label << "\n";
		return 1;
	}

	// Buffer to hold characters read from the file.
	std::string buffer;

	while (!input.eof())
	{
		bool found = false;
		// Variable to hold positions in the string, set equal to npos - the max possible value.
		std::string::size_type pos = std::string::npos;

		while (!input.eof() && !found)
		{
			// Read a line into the buffer.
			getline(input, buffer);
			// Search for the size attribute in the current line.
			pos = buffer.find("size");

			// If it was found.
			if (pos != std::string::npos)
			{
				found = true;
				buffer = buffer.substr(4);

				// Finds the first integer in the substring, copies the index of the next element of the string into pos.
				cols = stoi(buffer, &pos);
				// Finds the first integer in the sub-substring starting from pos.
				rows = stoi(buffer.substr(pos));
			}
		}

		// If the size was successfully read.
		if (cols != 0 && rows != 0)
		{
			found = false;
			while (!input.eof() && !found)
			{
				// Read a line into the buffer.
				getline(input, buffer);
				// Search for the size attribute in the current line.
				pos = buffer.find("horizontal_resolution");

				// If it was found.
				if (pos != std::string::npos)
				{
					found = true;
					buffer = buffer.substr(21);
					// Finds the integer value in the buffer.
					horRes = stoi(buffer);
				}
			}
		}
	}
	input.close();

	if (cols != 0 && rows != 0 && horRes != 0)
		return 0;

	else
	{
		std::cout << "Failed to parse label file correctly, please check the label file.\n";
		return 1;
	}
}

// Returns 0 on successful read of DEM file.
bool DEM::readDEM(std::string fname)
{
	std::cout << "Opening DEM file.\n";
	std::ifstream input(fname, std::ios::binary);
	std::cout << "File open.\n";
	// Check file opened correctly.
	if (!input)
	{
		std::cout << "Failed to open file!\n";
		return 1;
	}

	
	// First check the file is of the expected length.
	int arraySize = rows * cols;

	input.seekg(0, input.end);
	int fileSize = input.tellg();
	// Return to the start.
	input.seekg(0, input.beg);

	if (fileSize / sizeof(float) != arraySize)
	{
		std::cout << "Failure. Expected size of : " << arraySize << "\nRead size was : " << fileSize << "\n";
		return 1;
	}

	std::cout << "Read successful.\nExpected size: " << arraySize << "\nActual size: " << fileSize / sizeof(float) << "\n\n";
	// Allocate space for an array to hold the data.
	data = new float[arraySize];

	for (int i = 0; i < arraySize; i++)
	{
		input.read(reinterpret_cast <char*> (&data[i]), sizeof(float));
		// Convert endianness of the float.
		data[i] = ReverseFloat(data[i]);
	}
	input.close();
	return 0;
}


/////////////////////////
// --- GET METHODS --- //
/////////////////////////

float* DEM::getData(){ return data; }

int DEM::getRows(){ return rows; }

int DEM::getCols(){ return cols; }

int DEM::getRes(){ return horRes; }


////////////////////////////////////////////////////////
///////////////////	UTILITY FUNCTIONS //////////////////
////////////////////////////////////////////////////////

// Swaps the endianness of the input float.
float ReverseFloat(const float inFloat)
{
	float outFloat;
	char *inChar = (char*) &inFloat;
	char *outChar = (char*) &outFloat;

	// swap the bytes into a temporary buffer
	outChar[0] = inChar[3];
	outChar[1] = inChar[2];
	outChar[2] = inChar[1];
	outChar[3] = inChar[0];

	return outFloat;
}