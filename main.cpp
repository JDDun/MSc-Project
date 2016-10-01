#include "siteLocator.h"

int main(int argc, char *argv[])
{
	std::string fileDEM = argv[1];
	std::string fileLabel = fileDEM + ".txt";

	siteLocator test;

	test.findSites(fileLabel, fileDEM);

	return 0;
}