#include "PAN_interface.h"

PAN_interface::PAN_interface()
{
	int error = 0;

#ifdef _WIN32
	WSAData wsaData;
	if (WSAStartup(MAKEWORD(1, 1), &wsaData))
	{
		(void)fprintf(stderr, "Failed to initialise winsock 1.1\n");
		error = 1;
	}
#endif
	// Kill the program if theres been an error.
	if (error)
		exit(error);

	/* First get the numeric IP address of the server */
	addr = hostid_to_address((char *)SERVER_NAME);

	/* Create a communications TCP/IP socket */
	sock = socket(AF_INET, SOCK_STREAM, 0);
	if (sock == -1)
	{
		(void)fprintf(stderr, "Error: failed to create socket.\n");
		error = 2;
	}
	// Kill the program if theres been an error.
	if (error)
		exit(error);

	/* Connect the socket to the remote server */
	saddr.sin_family = AF_INET;
	saddr.sin_addr.s_addr = addr;
	saddr.sin_port = htons(SERVER_PORT);
	saddr_len = sizeof(struct sockaddr_in);
	if (connect(sock, (struct sockaddr *)&saddr, saddr_len) == -1)
	{
		const char *fmt = "Error: failed to connect to %s:%d\n";
		(void)fprintf(stderr, fmt, SERVER_NAME, SERVER_PORT);
		error = 2;
	}
	// Kill the program if theres been an error.
	if (error)
		exit(error);
}

////////////////////////////////////
///// Connect / Disconnect /////////
////////////////////////////////////

void PAN_interface::PAN_connect()
{
	/* Start the PANGU network communications protocol */
	pan_protocol_start(sock);
}

void PAN_interface::PAN_disconnect()
{
	/* Terminate the PANGU network communications protocol */
	pan_protocol_finish(sock);
	SOCKET_CLOSE(sock);

#ifdef _WIN32
	WSACleanup();
#endif
}

///////////////////////////////////
///// Getting Images //////////////
///////////////////////////////////

int PAN_interface::get_landing_region(int maxDim, float scale)
{

	int status;
	float x, y, z, yaw, pitch, roll;
	//char fname[1024];

	/* Initialise the camera position */
	// z-position is set such that each image pixel corresponds to a single DEM cell.
	x = 0.0f, y = 0.0f, z = maxDim * scale / 2 / tan(PI / 12);

	yaw = 0.0f, pitch = -90.0f, roll = 0.0f;

	/* Define the field of view we want to use */
	pan_protocol_set_field_of_view(sock, 30.0);

	/* Instruct the viewer to use this position */
	pan_protocol_set_viewpoint_by_angle(sock, x, y, z, yaw, pitch, roll);

	status = get_and_save_image(sock, "landing_region.ppm");
	if (status) return status;

	/* Success */
	return 0;
}

// Save an image of a target landing site to file.
int PAN_interface::get_site(int x_coord, int y_coord, int maxDim, float scale, int siteNum)
{
	int status;
	float x, y, z, yaw, pitch, roll;
	char fname[1024];

	z = 512.0f;										// 'low' height.
	yaw = 0.0f, pitch = -90.0f, roll = 0.0f;		// Looking straight down vertically.

	/* Define the field of view */
	pan_protocol_set_field_of_view(sock, 30.0);

	// Determine the relative x,y coordinates for the viewer position.
	x = x_coord - (maxDim * scale / 2);
	y = (maxDim * scale / 2) - y_coord;

	/* Initialise the camera position */
	pan_protocol_set_viewpoint_by_angle(sock, x, y, z, yaw, pitch, roll);

	/* Get this image */
	(void)sprintf(fname, "test_site_%03d.ppm", siteNum);
	status = get_and_save_image(sock, fname);
	if (status) return status;

	/* Success */
	return 0;
}



////////////////////////////////////////
////		Utility Functions      ////
///////////////////////////////////////


// Converts given host ID to usable address.
static unsigned long
hostid_to_address(char *s)
{
	struct hostent *host;

	/* Assume we have a dotted IP address ... */
	long result = inet_addr(s);
	if (result != (long)INADDR_NONE) return result;

	/* That failed so assume DNS will resolve it. */
	host = gethostbyname(s);
	return host ? *((long *)host->h_addr_list[0]) : INADDR_NONE;
}


// Takes and saves images from viewer.
static int
get_and_save_image(SOCKET sock, char *fname)
{
	unsigned long todo;
	unsigned char *ptr, *img;
	FILE *handle;

	/* Retrieve an image */
	(void)fprintf(stderr, "Getting image '%s'\n", fname);
	img = pan_protocol_get_image(sock, &todo);

	/* Open the output file for writing */
	handle = fopen(fname, "wb");
	if (!handle)
	{
		const char *fmt = "Error: failed to open '%s' for writing\n";
		(void)fprintf(stderr, fmt, fname);
		return 4;
	}

	/* Write the image data to the file */
	ptr = img;
	while (todo > 1024)
	{
		long wrote;
		wrote = fwrite(ptr, 1, 1024, handle);
		if (wrote < 1)
		{
			const char *fmt = "Error writing to '%s'\n";
			(void)fprintf(stderr, fmt, fname);
			(void)fclose(handle);
			return 5;
		}
		else
		{
			todo -= wrote;
			ptr += wrote;
		}
	}
	if (todo)
	{
		long wrote;
		wrote = fwrite(ptr, 1, todo, handle);
		if (wrote < 1)
		{
			const char *fmt = "Error writing to '%s'\n";
			(void)fprintf(stderr, fmt, fname);
			(void)fclose(handle);
			return 5;
		}
	}

	/* Close the file */
	(void)fclose(handle);

	/* Release the image data */
	(void)free(img);

	/* Return success */
	return 0;
}
