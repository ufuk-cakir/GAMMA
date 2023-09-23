

**Content**
1. Installation
2. 
# Galactic Attributes of Mass, Metallicity, and Age

This project, developed by Ufuk Çakir, introduces the code  to generate the GAMMA (Galactic Attributes of Mass, Metallicity, and Age) dataset, a comprehensive collection of galaxy data tailored for Machine Learning applications. 
The dataset offers detailed 2D maps and 3D cubes of 11 960 galaxies,capturing essential attributes: stellar age, metallicity, and mass. Ideal for feature extraction, clustering, and regression tasks, GAMMA offers a unique lens for exploring galac-tic structures through computational methods and is a bridge between astrophysical simulations and the field of scientific machine learning (ML)

*Interdisciplinary Center for Scientific Computing (IWR), Heidelberg University, 09/2023*


## Installation

The dataset loads galaxies from the [IllustrisTNG](https://www.tng-project.org/) suite. For thet, the respective python package should be installed:
```
$ cd ~
$ git clone https://github.com/illustristng/illustris_python.git
$ pip install illustris_python/
```

For installation, run  
`source setup.sh`

Check the [Starting Guide](https://www.tng-project.org/data/docs/scripts/) on the TNG webpage for more information.

## Configuration

The [config.json](config.json) file contains all the settings nedded to run the data generation. All the configuration should be made there.
The required fields are:
- simulation: The simluation from which the data should be generated. Currently only "IllustrisTNG" is supported.
- "particle_types": The particle type to calculate the images
- "galaxy_parameters": Additional parameters to be saved for each galaxy.
- "img_res": Image resolution in each dimension
- "path": Output Path of the Created HDF5 File
- "halo_ids": If none, it will do automatic selection of galaxies.
- "dim": dimension of image , either (2 and/or 3) dimensional
- "log_M_min": lower Mass cut in log10(M_sun/h)
- "log_M_max": upper Mass cut in log10(M_sun/h)
- "fields": Fields to calculate the images. For each field the attributes "mass_weighted" and "normed" define wheter or not to calculate a mass weighted image and to norm or not.
- "GalaxyArgs": Arguments specified to load galaxy defined in the [Galaxy Class](src/gamma/galaxy.py)



## Generation
To generate the dataset run
`source generate_data.sh`


## Data Structure

The data will be stored in a HDF5 File in the following way:






