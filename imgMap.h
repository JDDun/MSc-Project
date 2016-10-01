#pragma once

// openCV includes.
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

// Template class for creating images from hazard map components.
template <class Type>
class imageMap 
{
private:
	Type *data;     // pointer to the data - could be floats or ints.
	int rows;		// number of rows, cols in the data.
	int cols;

	unsigned char *map;
public:
	imageMap() = default;
	~imageMap();
	imageMap(Type *input, int ydim, int xdim);
	void draw(std::string label);
};

// Constructor.
template <class Type>
imageMap<Type>::imageMap(Type *input, int ydim, int xdim)
{
	data = input;
	rows = ydim;
	cols = xdim;

	map = new unsigned char[rows*cols];

	// Determine the max value (for scaling purposes).
	Type maxVal = data[0];
	for (int i = 1; i < rows*cols; i++)
		if (data[i] > maxVal)
			maxVal = data[i];

	// Create map, invert the values such that white regions are low hazard score.	
	for (int i = 0; i < rows*cols; i++)
		map[i] = unsigned char(255 - (float(data[i]) / float(maxVal) * 255.0f));
}

// Destructor
template <class Type>
imageMap<Type>::~imageMap()
{
	delete[] map;
}

template <class Type>
void imageMap<Type>::draw(std::string label)
{
	cv::Size size(cols, rows);
	cv::Mat image(size, CV_8UC1, map);

	cv::resize(image,image,cv::Size(512, 512));

	cv::imshow(label, image);
	cv::waitKey(0);

	image.release();
}
