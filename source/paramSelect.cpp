#include "paramSelect.h"

///////////////////////////////////////////
// --- Fluid Designer generated code --- //
///////////////////////////////////////////

void paramSelect::cb_bt_default_i(Fl_Button*, void*) {
	radio_CPU->value(false);
	radio_GPU->value(true);

	check_verbose->value(0);
	in_shadow->value(1);
	in_rough->value(1);
	in_slope->value(1);

	in_sites->value("5");

	std::cout << "Reset to default.\n";
}
void paramSelect::cb_bt_default(Fl_Button* o, void* v) {
	((paramSelect*)(o->parent()->user_data()))->cb_bt_default_i(o, v);
}

void paramSelect::cb_bt_launch_i(Fl_Button*, void*) {
	use_GPU = radio_GPU->value();
	use_verbose = check_verbose->value();

	shadowWeight = in_shadow->value();
	slopeWeight = in_slope->value();
	roughWeight = in_rough->value();

	numSites = std::stoi(in_sites->value());

	param_window->hide();
	
	// Error check the weights.
	validate();
}
void paramSelect::cb_bt_launch(Fl_Button* o, void* v) {
	((paramSelect*)(o->parent()->user_data()))->cb_bt_launch_i(o, v);
}

paramSelect::paramSelect() {
	use_GPU = true;
	use_verbose = false;
	slopeWeight = 1;
	roughWeight = 1;
	shadowWeight = 1;
	numSites = 5;

	{ param_window = new Fl_Double_Window(415, 380, "Parameter Selection Dialog");
	param_window->user_data((void*)(this));
	{ bt_default = new Fl_Button(102, 330, 70, 20, "DEFAULT");
	bt_default->callback((Fl_Callback*)cb_bt_default);
	} // Fl_Button* bt_default
	{ gpu_radio_group = new Fl_Group(102, 55, 25, 55, "Computation Mode");
	{ radio_CPU = new Fl_Round_Button(102, 55, 25, 25, "CPU");
	radio_CPU->type(102);
	radio_CPU->down_box(FL_ROUND_DOWN_BOX);
	radio_CPU->when(FL_WHEN_CHANGED);
	} // Fl_Round_Button* radio_CPU
	  { radio_GPU = new Fl_Round_Button(102, 85, 25, 25, "GPU");
	  radio_GPU->type(102);
	  radio_GPU->down_box(FL_ROUND_DOWN_BOX);
	  radio_GPU->when(FL_WHEN_CHANGED);
	  radio_GPU->value(1);
	  } // Fl_Round_Button* radio_GPU
	  gpu_radio_group->end();
	} // Fl_Group* gpu_radio_group
	{ bt_launch = new Fl_Button(242, 330, 70, 20, "LAUNCH");
	bt_launch->color((Fl_Color)80);
	bt_launch->labelfont(1);
	bt_launch->callback((Fl_Callback*)cb_bt_launch);
	} // Fl_Button* bt_launch
	{ check_verbose = new Fl_Check_Button(242, 55, 30, 20, "Verbose Output");
	check_verbose->down_box(FL_DOWN_BOX);
	check_verbose->align(Fl_Align(FL_ALIGN_TOP));
	} // Fl_Check_Button* check_verbose
	{ in_slope = new Fl_Value_Input(195, 146, 75, 24, "Slope Weight");
	in_slope->value(1);
	} // Fl_Value_Input* in_slope
	{ in_rough = new Fl_Value_Input(195, 176, 75, 24, "Roughness Weight");
	in_rough->value(1);
	} // Fl_Value_Input* in_rough
	{ in_shadow = new Fl_Value_Input(195, 206, 75, 24, "Shadow Weight");
	in_shadow->value(1);
	} // Fl_Value_Input* in_shadow
	{ in_sites = new Fl_Int_Input(195, 271, 60, 24, "Number of Landing Sites");
	//in_sites->align(Fl_Align(FL_ALIGN_TOP));
	in_sites->value("5");
	} // Fl_Int_Input* in_sites
	param_window->end();
	} // Fl_Double_Window* param_window
}

///////////////////////////
// --- Other Methods --- //
///////////////////////////

// Print all the parameter values to console.
void paramSelect::print_params()
{
	std::cout << "Slope: " << slopeWeight << "  Rough: " << roughWeight << "  Shadow: " << shadowWeight << std::endl;

	std::cout << "Verbose: " << use_verbose << "\n";

	std::cout << "GPU: " << use_GPU << "\n";
}

// Display the dialog box.
void paramSelect::show()
{
	param_window->show();
}

// Get the parameter values.
void paramSelect::getParams(bool& gpu, bool& verb, float& s, float& r, float& sh, int& ns)
{
	gpu = use_GPU;
	verb = use_verbose;
	s = slopeWeight;
	r = roughWeight;
	sh = shadowWeight;
	ns = numSites;
}

// Error checking for the weighting parameters.
void paramSelect::validate()
{
	// If any of the weights are negative.
	if (slopeWeight < 0 || roughWeight < 0 || shadowWeight < 0)
	{
		std::cout << "Invalid weightings, resetting weights to default values.\n";

		slopeWeight = 1;
		roughWeight = 1;
		shadowWeight = 1;
	}

	else if (slopeWeight == 0 && roughWeight == 0 && shadowWeight == 0)
	{
		std::cout << "Invalid weightings, resetting weights to default values.\n";

		slopeWeight = 1;
		roughWeight = 1;
		shadowWeight = 1;
	}

	if (numSites <= 0)
	{
		std::cout << "Invalid number of sites, resetting number of sites to default value of 5.\n";
		numSites = 5;
	}
}
