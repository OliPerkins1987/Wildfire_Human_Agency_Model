# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 10:18:46 2021

@author: Oli
"""
import setuptools

#with open("README.md", "r") as f:
#    readme = f.read()

setuptools.setup(
    name="wham",
    author="OP",
    #author_email="ahf.kuhnregnier@gmail.com",
    #description="Utilities for the analysis of wildfire data.",
    #long_description=readme,
    #long_description_content_type="text/markdown",
    #url="https://github.com/akuhnregnier/wildfires",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    #classifiers=[
    #    "Programming Language :: Python :: 3",
    #    "License :: OSI Approved :: MIT License",
    #    "Operating System :: OS Independent",
    #],
    #entry_points="""
    #    [console_scripts]
    #    valid-ports=wildfires.ports:main
    #    general-lab=wildfires.dask_cx1.dask_utils:main
    #    mosaic-modis-tiles=wildfires.data.mosaic_modis_tiles:main
     # """,
    python_requires=">=3.7",
    #setup_requires=["setuptools-scm"],
    #use_scm_version=dict(write_to="src/wildfires/_version.py"),
    #include_package_data=True,
)