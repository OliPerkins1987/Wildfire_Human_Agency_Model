# Wildfire_Human_Agency_Model

This repository contains code to run the Wildfire Human Agency Model, a global geospatial model of human fire use and management. 

Installation instructions follow below. see docs/ for user guidelines. All questions and bugs to oliver.perkins@kcl.ac.uk

## Installation guidelines

### 1.1 Install code and data	

Download or clone code from main branch.

Data for historical model runs can be downloaded from: 
https://zenodo.org/records/8363979

### 1.2	Set up virtual environment

Instructions for setting up a virtual environment (using conda) with required python packages are now provided for each of Windows, MacOS and Linux; these instructions have been tested for Windows 10 & 11, MaxOS 14.1 and Linux (Mint 20).
Package requirements are provided as a wham38.yml file for Windows users, a requirements.txt for Linux and requirements_mac.txt for MacOS users. These files are stored in the base directory of where you downloaded & unzipped the wham code.

#### 1.2.1	Windows:
Windows users can either use the .yml file for convenience, or follow Linux instructions below using requirements.txt. 
Open an anaconda command prompt and navigate to the where wham38.yml file is saved then run the following commands:  
```shell
conda env create -f wham38.yml  
conda activate wham38  
```

#### 1.2.2 MacOS:
Open terminal, navigate to where requirements.txt is saved, and run the following commands:  
```shell
conda create --name wham38 python=3.8  
conda activate wham38  
pip install -r requirements_mac.txt  
conda install -c conda-forge netcdf4  
```
NB: there are known issues with installation of netcdf4 for Mac users using pip. Using conda forge should solve this.

#### 1.2.3 Linux:
Open terminal, navigate to where requirements.txt is saved, and run the following commands:  
```shell
conda create --name wham38 python=3.8  
conda activate wham38  
pip install -r requirements.txt  
```

### 1.3 Install wham code

Installation of wham code is done using python install from command line or terminal. Install the code by navigating to the directory where you unzipped the code, and running the following command: 
```shell
python setup.py install  
```

### 1.4 Data download and set up

Check where you have saved (and unzipped) the model data (from step 1). We now need to modify a line of code to specify to the model where data are saved: 

1. Open the ‘local_load_up.py’ script (in the src/data_import/ directory) in a text editor  
2. Go to lines 25-26 & edit the paths to point to where the data files are stored (i.e. the unzipped model data folder), and the sub directory where the map data is stored (by default …/wham_dynamic/)  
- If copying and pasting the path on Windows, please replace any backslash (\) characters in the path either with Unix-style forward slash (/), or with a double backslash (\\), since a single backslash is interpreted as an escape character in Python.  
- Ensure the file paths both end in a trailing slash - e.g. -  ‘…/mypath/’ or ‘…\\mypath\\’  
3. save your edits to ‘local_load_up.py’ 

### 1.5 Testing

To check that has all worked, at the command line or in terminal navigate to tests/ in the code files and follow instructions below. 

#### 1.5.1	Windows:
Type & run the following command:
```shell
pytest  
```
Running code tests may trigger a Window’s Defender Firewall; this relates to the dask.distributed library making requests to establish a slave & master set of parallel CPU cores. Please select yes, when asked for permission. 

#### 1.5.2	MacOS:
Running tests may breach the number of open files allowed by shell. To alter this, enter:  
```shell
ulimit -n 2048  
```
Type & run the following command:  
```shell
pytest
```

#### 1.5.3	Linux:
Type & run the following command:
```shell
pytest  
```

Tests should take around 15-30 mins to run on a medium performance desktop. Tests should return 76 warnings, these are primarily composed of the following warning messages & can be ignored:  
- dividing by zero – this is where the land fraction equals 0  
- np.bool being deprecated – this arises from the netCDF4 package  
- np.log being applied to a negative number – this relates to the netCDF4 package using -3.3e38 for missing values  

However, any test failures need to be explored. Please report them to oliver.perkins@kcl.ac.uk

