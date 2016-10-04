#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#include "socket_stuff.h"
#include "pan_protocol_lib.h"

#define SERVER_NAME	"localhost"				// Local Version
//#define SERVER_NAME	"192.168.0.14"		// Network version.
#define SERVER_PORT	10363					// Default PANGU port.

#define PI 3.14159265						// Numerical constant used for trig calculation.

// Class to connect, disconnect to PANGU and request images.
class PAN_interface
{
private:
	long addr;
	int result = 0;
	SOCKET sock;
	unsigned long saddr_len;
	struct sockaddr_in saddr;

public:
	PAN_interface();

	void PAN_connect();
	void PAN_disconnect();

	int get_landing_region(int maxDim, float scale);

	int get_site(int x_coord, int y_coord, int maxDim, float scale, int siteNum);
};

static int get_and_save_image(SOCKET, char *);
static unsigned long hostid_to_address(char *);
