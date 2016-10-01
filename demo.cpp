#include "demo.h"

// Given a vector of candidate landing sites, circle them on the provided image and display the output sequentially.
void drawSites(cv::Mat surface, std::vector<landingSite> sites)
{
	std::string label;

	
	int centreX = surface.cols / 2 - 1;
	int centreY = surface.rows / 2 - 1;

	int centreIndex = (centreY * surface.cols) + centreX;
	landingSite centre(0, centreIndex, surface.cols);			// Centre of the image, used purely for calculating distances. Hazard score is irrelevant.
 
	cv::Mat withSites;

	surface.copyTo(withSites);

	cv::cvtColor(withSites, withSites, CV_GRAY2RGB);		// Set the output image to colour so I can draw coloured circles.

	// Draw a 10x10 box in the centre of the image.
	cv::rectangle(withSites, cv::Rect(cv::Point(centreX - 5, centreY - 5),cv::Point(centreX + 5,centreY + 5)), cv::Scalar(0, 150, 0), 2);

	// Draw circles at each of the numSites locations in the points array, in colour.
	for (int i = 0; i < sites.size(); i++)
	{
		// openCV point object of the x,y coordinates.
		cv::Point coord(sites[i].getX(), sites[i].getY());

		// Text label to identify the site.
		label = "Site " + std::to_string(i + 1);

		// Draw a circle around the site.
		cv::circle(withSites, coord, 15, cv::Scalar(150, 0, 0), 3);

		// Label the site.
		cv::putText(withSites, label, cv::Point(coord.x - 75, coord.y - 20), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.0);
	}
	// Resize the image to 768x768 before displaying it.
	cv::resize(withSites, withSites, cv::Size(768, 768));
	
	// Display circled sites.
	cv::imshow("Landing Sites (Circled)", withSites);
	cv::waitKey(0);

	cv::imwrite("landing_circled.png", withSites);



	for (int i = 0; i < sites.size(); i++)
	{
		// Buffer for file name.
		char file[50];
		sprintf_s(file, 50, "test_site_%03d.ppm", i);
		// Read the image into an opencv Mat object, resize it.
		cv::Mat siteImage = cv::imread(file);
		cv::resize(siteImage, siteImage, cv::Size(512, 512));

		// Label the site.
		label = "Site " + std::to_string(i + 1);
		cv::putText(siteImage, label, cv::Point(siteImage.cols / 2 - 220, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(0, 255, 0), 1.0);
		
		// Label the hazard score.
		label = "Hazard Score: " + std::to_string(sites[i].getScore());
		cv::putText(siteImage, label, cv::Point(siteImage.cols / 2 - 220, 75), cv::FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(0, 255, 0), 1.0);
		
		// Label the distance to the centre (region of interest).
		float distance = sites[i].dist(centre);
		label = "Distance from ROI: " + (std::to_string(distance).substr(0,6));		// Distance is cut down to 5 digits.
	
		//sprintf_s(buff, 50, "Distance from ROI: %03g", distance);
		cv::putText(siteImage, label, cv::Point(siteImage.cols / 2 - 220, 100), cv::FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(0, 255, 0), 1.0);
		
		// Display the landing site with labelled information.
		cv::imshow("Landing Site " + std::to_string(i + 1), siteImage);

		cv::waitKey(0);
	}
}