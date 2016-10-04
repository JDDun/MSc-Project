#include "grassFire.h"

void grassFire(int*& matrix, int rows, int cols)
{
	cout << "Origional Matrix.\n";
	print5x5(matrix);

	cout << "Calling first pass.\n";
	firstPass(matrix, rows, cols);
	cout << "Complete.\nCalling second pass.\n";
	secondPass(matrix, rows, cols);
	cout << "Complete.\n";

	cout << "\nAfter first/second.\n";
	print5x5(matrix);
}

// Raster
void firstPass(int*& matrix, int rows, int cols)
{
	// Go row by row.
	for (int j = 0; j < rows; j++)
	{
		// y-coord fixed
		for (int i = 1; i < cols; i++)
		{
			// max(current,left-1)
			matrix[i + j*cols] = max(matrix[i + j*cols], matrix[(i - 1) + j*cols] - 1);
		}
	}

	// Go column by column.
	for (int i = 0; i < cols; i++)
	{
		// x-coord fixed
		for (int j = 1; j < rows; j++)
		{
			// max(current,above-1)
			matrix[i + j*cols] = max(matrix[i + j*cols], matrix[i + (j - 1)*cols] - 1);
		}
	}
}


// Anti-Raster
void secondPass(int*& matrix, int rows, int cols)
{
	// Go row by row.
	for (int j = rows-1; j >= 0; j--)
	{
		// y-coord fixed
		for (int i = cols - 2; i >= 0; i--)
		{
			// max(current,right-1)
			matrix[i + j*cols] = max(matrix[i + j*cols], matrix[(i + 1) + j*cols] - 1);
		}
	}

	// Go column by column.
	for (int i = cols - 1; i >= 0; i--)
	{
		// x-coord fixed
		for (int j = rows - 2; j >= 0; j--)
		{
			// max(current,below-1)
			matrix[i + j*cols] = max(matrix[i + j*cols], (matrix[i + (j + 1)*cols] - 1));
		}
	}
}


// Prints a 5x5 sub section of the matrix.
void print5x5(int* matrix)
{
	for (int i = 0; i < 25; i++)
	{
		cout << matrix[(i % 5) + (i / 5) * 1024] << "    ";
		if ((i + 1) % 5 == 0)
			cout << endl;
	}
}

