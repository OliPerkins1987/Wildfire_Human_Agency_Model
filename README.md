Wildfire_Human_Agency_Model

This repository contains code to run the Wildfire Human Agency Model, a global geospatial model of human fire use and management. 

It has two principle branches. These are main (here) and v2_025. Main is setup as in the GMD model description paper (https://doi.org/10.5194/gmd-17-3993-2024). V2_025 is setup as in this subsequent paper (https://iopscience.iop.org/article/10.1088/1748-9326/adeff5). 

Please see the note on differences (v1 vs vs_025). The essence is that main [v1] is setup for use in Earth System Model resolution (~1.5 degrees), whilst V2_025 is designed for use in ISIMIP & other impacts modelling protocols.

Installation instructions follow below. see /docs for user guidelines. All questions and bugs to oliver.perkins@kcl.ac.uk

*** Installation guidelines ***

1.1 Install code and data	

Download or clone code from main branch.

Data for historical model runs can be downloaded from: 
https://zenodo.org/records/8363979


1.2	Set up virtual environment

Instructions for setting up a virtual environment (using conda) with required python packages are now provided for each of Windows, MacOS and Linux; these instructions have been tested for Windows 10 & 11, MaxOS 14.1 and Linux (Mint 20).
Package requirements are provided as a wham_312.yml file for Windows users, a requirements_312.txt for Linux and requirements_312_mac.txt for MacOS users. These files are stored in the base directory of where you downloaded & unzipped the wham code.

1.2.1	Windows:
Windows users can either use the .yml file for convenience, or follow Linux instructions below using requirements.txt. 
Open an anaconda command prompt and navigate to the where wham_312.yml file is saved then run the following commands:
conda env create -f wham_312.yml
conda activate wham_312

1.2.2 MacOS:
Open terminal, navigate to where requirements.txt is saved, and run the following commands: 
conda create --name wham_312 python=3.12
conda activate wham_312
pip install -r requirements_312_mac.txt
conda install -c conda-forge netcdf4
NB: there are known issues with installation of netcdf4 for Mac users using pip. Using conda forge should solve this.

1.2.3 Linux:
Open terminal, navigate to where requirements.txt is saved, and run the following commands: 
conda create --name wham_312 python=3.12
conda activate wham_312
pip install -r requirements_312.txt


1.3 Install wham code

Installation of wham code is done using python install from command line or terminal. Install the code by navigating to the directory where you unzipped the code, and running the following command: 
pip install -e .


1.4 Data download and set up

Check where you have saved (and unzipped) the model data (from step 1). We now need to modify a line of code to specify to the model where data are saved: 

i.	Open the ‘local_load_up_func.py’ script (in the src/data_import directory) in a text editor
ii.	Go to lines 41 & 43, edit the paths to point to where the data files are stored (i.e. the unzipped model data folder), and the sub directory where the map data is stored. by default these are ~/data/ and ~/drive/ respectively.
	If copying and pasting the path on Windows, please replace any backslash (\) characters in the path either with Unix-style forward slash (/), or with a double backslash (\\), since a single backslash is interpreted as an escape character in Python.
	Ensure the file paths both end in a trailing slash - e.g. -  ‘…/mypath/’ or ‘…\\mypath\\’
iii.	save your edits to ‘local_load_up_func.py’ 

1.5 Testing

To check that has all worked, at the command line or in terminal navigate to /tests in the code files and follow instructions below. 

1.5.1	Windows:
Type & run the following command:
pytest
Running code tests may trigger a Window’s Defender Firewall; this relates to the dask.distributed library making requests to establish a slave & master set of parallel CPU cores. Please select yes, when asked for permission. 

1.5.2	MacOS:
Running tests may breach the number of open files allowed by shell. To alter this, enter: 
ulimit -n 2048
Type & run the following command:
pytest

1.5.3	Linux:
Type & run the following command:
pytest

Tests should take around 1-2 mins to run on a medium performance desktop. Tests should return 76 warnings, these are composed of the following warning messages & can be ignored:
•	dividing by zero – this is where the land fraction equals 0
•	np.bool being deprecated – this arises from the netCDF4 package 
•	np.log being applied to a negative number – this relates to the netCDF4 package using -3.3e38 for missing values

However, any test failures need to be explored. Please report them to oliver.perkins@kcl.ac.uk

