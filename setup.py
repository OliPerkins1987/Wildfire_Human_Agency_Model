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
    author_email="oliver.perkins@kcl.ac.uk",
    description="WHAM!: Wildfire_Human_Agency_Model.",
    #long_description=readme,
    #long_description_content_type="text/markdown",
    url="https://github.com/OliPerkins1987/Wildfire_Human_Agency_Model",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},

    python_requires=">=3.8",
    #setup_requires=["setuptools-scm"],
    #use_scm_version=dict(write_to="src/wildfires/_version.py"),
    #include_package_data=True,
)