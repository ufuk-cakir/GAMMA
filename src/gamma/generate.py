"""This file contains the functions to generate the galaxy data and save it to the HDF5 file.
"""
import json
import argparse
import h5py
import os
import numpy as np
from tqdm import tqdm

from megs.data import Galaxy

def _create_data_structure(
    n_galaxies,
    image_res,
    galaxy_parameters,
    particle_types,
    fields,
    path="./",
    dim=None,
):
    """Creates the HDF5 data structure for the galaxy data

    Parameters
    ----------
    n_galaxies: int
        Number of galaxies to be saved
    image_res: int
        Image resolution. The images are either 2D or 3D arrays with shape (image_res, image_res) or (image_res, image_res, image_res), depending on the dim argument.
    galaxy_parameters: list
        List of galaxy parameters to be saved. These need to be attributes of the Galaxy class and are saved in the "Galaxies/Attributes" group.
    particle_types: list
        List of particle types to be saved e.g. ["stars", "gas"].
    fields: dict
        Dictionary of fields to be saved, where the key is the field name and the value is a boolean indicating if the field is mass weighted or not.
        The fields are the same for all particle types and are saved in the "Images" group. The value is used to calculate the image later. #TODO maybe change this
    path: str
        Path to the HDF5 file
    dim: int
        Dimension of the images. If None, both 2D and 3D images are saved. If 2, only 2D images and if 3, only 3D images are saved.

    Example:
    --------
    >>> create_data_structure(1000, (64,64), galaxy_paramters=["mass","halo_id"], particle_types =["stars"], {"Masses":False, "GFM_Metallicity":False, "GFM_StellarFormationTime":True})
    
    This will create a "galaxy_data.hdf5" HDF5 file with the following structure and save it to the current directory:
    
    >>> Galaxies
        Attributes
            mass: (1000,)
            halo_id: (1000,)
        Particles
            stars
                Images
                    dim2
                        Masses: (1000,64,64)
                        GFM_Metallicity: (1000,64,64)
                        GFM_StellarFormationTime: (1000,64,64)
                    dim3
                        Masses: (1000,64,64,64)
                        GFM_Metallicity: (1000,64,64,64)
                        GFM_StellarFormationTime: (1000,64,64,64)
    """
    # Check if dim is valid
    if dim not in [None, 2, 3]:
        raise ValueError("dim should be either None, 2 or 3.")
    if dim is None:
        # If dim is None, save both 2D and 3D images
        dim = [2, 3]
    # Open the HDF5 file in "w" mode to create a new file
    with h5py.File(os.path.join(path, "galaxy_data.hdf5"), "w") as f:
        # Create the Galaxies group
        galaxies_group = f.create_group("Galaxies")
        galaxy_attributes = galaxies_group.create_group("Attributes")
        # Create the datasets for the galaxy parameters
        for parameter in galaxy_parameters:
            galaxy_attributes.create_dataset(
                parameter, shape=(n_galaxies,), maxshape=(None,)
            )

        particles_group = galaxies_group.create_group("Particles")
        # Create the Particle Types group
        for particle_type in particle_types:
            particle_type_group = particles_group.create_group(particle_type)

            # Create the Images group
            images_group = particle_type_group.create_group("Images")

            # Create the dim2 and dim3 groups
            for d in dim:
                dim_group = images_group.create_group(f"dim{d}")

                # Create the datasets for the fields
                for field in fields:
                    # Determine the shape of the dataset
                    if d == 2:
                        shape = (n_galaxies, image_res, image_res)
                        maxshape = (None, None, None)
                    if d == 3:
                        shape = (n_galaxies, image_res, image_res, image_res)
                        maxshape = (None, None, None, None)
                    dim_group.create_dataset(field, shape=shape, maxshape=maxshape)


def _calculate_images(
    simulation,
    halo_ids,
    fields,
    plot_factor,
    image_res,
    path="./",
    resume=None,
    **kwargs,
):
    """Calculates the images for the galaxies and saves them to the HDF5 file
    
    Needs to be run after _create_data_structure() method.

    Parameters
    ----------
    simulation: str
        Simulation name (e.g. IllustrisTNG). Used to initialise the Galaxy class. The Galaxy class for a specific simulation should be defined in the simulations.py file. #TODO: Change name
    halo_ids: list
        List of halo IDs to calculate the images for. The halo_ids are used to load the galaxy data from the simulation.
    fields: dict
        Dictionary of fields to be saved, where the key is the field name. The values of the dictionary are passed to the get_image() method of the Galaxy class.
    plot_factor: float
        Factor for the image range. The image range is calculated as halfmass_radius*plot_factor and the image is centred on the galaxy centre.
        For the halfmass_radius only the particle type specified in the particle_type argument are used.
    path: str
        Path to the HDF5 file to save the data to. This file should be created using create_data_structure() method.
    resume: int, default=None
        Flag to resume the calculation from the last halo ID. If None, the calculation starts from the first halo ID in the list.
    **kwargs: dict
        Keyword arguments passed to the Galaxy class. Halo ID and particle type are overwritten in the loop.
    """
    # Check if the HDF5 file exists, which should be created using create_data_structure() method.
    if not os.path.exists(os.path.join(path, "galaxy_data.hdf5")):
        raise FileNotFoundError(
            f"{os.path.join(path, 'galaxy_data.hdf5')} does not exist. This should have been created using the _create_data_structure() method."
        )

    n_galaxies = len(halo_ids)
    # Open the HDF5 file in "append" mode
    with h5py.File(os.path.join(path, "galaxy_data.hdf5"), "a") as f:
        # Check if the "index_position" attribute exists
        if "index_position" in f.attrs:
            index_position = f.attrs["index_position"]
            # Check if the index position is valid
            if index_position > n_galaxies:
                raise ValueError(
                    f"Index position {index_position} is greater than the number of galaxies {n_galaxies}"
                )

            # Check if resume flag is set
            if resume is None:
                # Ask the user if they want to continue from the last index position
                if (
                    input(f"Continue from index position {index_position}? (y/n): ")
                    == "y"
                ):
                    resume = True
                else:
                    resume = False

            # Set the index position to the last index position if the user wants to continue
            if resume:
                index_position = f.attrs["index_position"]
            else:
                # Reset the index position since the user does not want to continue from the last index position
                index_position = 0

        else:
            # Attribute does not exist so set the index position to 0 to start the loop from the beginning
            index_position = 0

        # Save the index position to the HDF5 file
        f.attrs["index_position"] = index_position
        print(
            f"Starting to calculate data from index position {index_position} out of {n_galaxies} galaxies."
        )
        # Loop through the galaxies
        for index, haloid in enumerate(tqdm(halo_ids[index_position:])):
            # Create the galaxy object
            kwargs["halo_id"] = haloid

            # TODO: This loads the particle type specified in the kwargs. Need to change this to load all particle types
            g = Galaxy(simulation=simulation, **kwargs)

            # Get the galaxy parameters
            for parameter in f["Galaxies/Attributes"].keys():
                if hasattr(g, parameter):
                    f["Galaxies/Attributes"][parameter][index] = getattr(g, parameter)
                else:
                    raise ValueError(
                        f"Galaxy class does not have the attribute {parameter}"
                    )

            # Get the particle data
            for particle_type in f["Galaxies"]["Particles"].keys():
                # loop thorugh the dimensions
                for d in f["Galaxies"]["Particles"][particle_type]["Images"].keys():
                    # loop through the fields
                    for field in f["Galaxies"]["Particles"][particle_type]["Images"][
                        d
                    ].keys():
                        # Get the image
                        dim = int(d[-1])  # TODO: This is a bit hacky. Maybe change this
                        image = g.get_image(
                            field=field,
                            plotfactor=plot_factor,
                            res=image_res,
                            dim=dim,
                            **fields[field],
                        )
                        f["Galaxies"]["Particles"][particle_type]["Images"][d][field][
                            index
                        ] = image

            # Update the index position
            f.attrs["index_position"] += 1

        # Show the user that the images have been calculated
        print(
            "Images calculated and saved to HDF5 file: ",
            os.path.join(path, "galaxy_data.hdf5"),
        )


# TODO: Maybe specify all the parameters in a seperate JSON file and load them in the function. Maybe more convenient for the user.
def generate_data(
    simulation,
    halo_ids,
    fields,
    plot_factor,
    image_res,
    galaxy_parameters,
    particle_types,
    overwrite=None,
    resume=None,
    path="./",
    dim=None,
    **kwargs,
):
    """
    Generates the data for the galaxies and saves it to an HDF5 file.

    This method creates the HDF5 file data structure and saves the galaxy parameters and images to the file. The images are calculated using the get_image() method of the Galaxy class.
    This is later used to build the morphology model.

    Parameters
    ----------
    simulation: str
        Simulation name (e.g. IllustrisTNG). Used to initialise the Galaxy class. The Galaxy class for a specific simulation should be defined in the simulations.py file. #TODO: Change name
    halo_ids: list
        List of halo IDs to calculate the images for. The halo_ids are used to load the galaxy data from the simulation.
    fields: dict
        Dictionary of fields to be saved, where the key is the field name. The values of the dictionary are passed to the get_image() method of the Galaxy class.
        For more information on the fields see the get_image() method of the Galaxy class.
    plot_factor: float
        Factor for the image range. The image range is calculated as halfmass_radius*plot_factor and the image is centred on the galaxy centre.
        For the halfmass_radius only the particle type specified in the particle_type argument are used.
    path: str
        Path to the HDF5 file to save the data to. This file should be created using create_data_structure() method.
    dim: int, default = None
        Dimension of the images to be calculated. If None, both 2D and 3D images are calculated. Set to 2 or 3 to calculate only 2D or 3D images.
    **kwargs: dict
        Keyword arguments passed to the Galaxy class. Halo ID and particle type are overwritten in the loop.
        e.g. {"base_path":basePath,"halo_id":0,"particle_type": "stars", "snapshot":99} for IllustrisTNG

    Example
    -------
    Set up the parameters to call the generate_data() method
    
    >>> simulation = "IllustrisTNG" # Simulation name, used to initialise the Galaxy class
    >>> halo_ids = [0,1,2,3,4,5,6,7,8,9] # List of halo IDs to calculate the images for
    >>> particle_types = ["stars", "gas"]
    >>> img_res = [64,64]
    >>> plot_factor = 10
    >>> path = "./" # Path where the HDF5 file will be saved
    
    Define the fields to be saved
    
    >>> fields = {"Masses":{"mass_weighted":False,
                            "normed":True},
                "GFM_Metallicity":{"mass_weighted":True,
                                    "normed":True},
                "GFM_StellarFormationTime":{"mass_weighted":True,
                                            "normed":False}
                }
    >>> galaxy_parameters = ["mass", "halo_id"] # List of galaxy parameters to be saved
    >>> generate_data(simulation, halo_ids, fields, plot_factor, img_res, galaxy_parameters, particle_types, path, dim=None)
    
    This will create the HDF5 file and save the galaxy parameters and images to the file in the following structure:
    
    >>> ./galaxy_data.hdf5
        ├── Galaxies: Group
        │   ├── Attributes: Group
        │   │   ├── mass: (10,)
        │   │   └── halo_id: (10,)
        │   └── Particles: Group
        │       ├── gas: Group
        │       │   └── Images: Group
        │       │       ├── 2D:     # 2D images
        │       │       │   ├── Masses: (10, 64, 64)
        │       │       │   ├── GFM_Metallicity: (10, 64, 64)
        │       │       │   └── GFM_StellarFormationTime: (10, 64, 64)
        │       │       └── 3D    # 3D images
        │       │           ├── Masses: (10, 64, 64, 64)
        │       │           ├── GFM_Metallicity: (10, 64, 64, 64)
        │       │           └── GFM_StellarFormationTime: (10, 64, 64, 64)
        │       └── stars
        │           └── Images
        │               ├── dim2
        │               │   ├── Masses: (10, 64, 64)
        │               │   ├── GFM_Metallicity: (10, 64, 64)
        │               │   └── GFM_StellarFormationTime: (10, 64, 64)
        │               └── dim3
        │                   ├── Masses: (10, 64, 64, 64)
        │                   ├── GFM_Metallicity: (10, 64, 64, 64)
        │                   └── GFM_StellarFormationTime: (10, 64, 64, 64)
        
    
    You can access the datasets by hand using h5py:
    
    >>> import h5py
    >>> f = h5py.File("./galaxy_data.hdf5", "r")
    >>> f["Galaxies/Attributes/mass"][0]
    1.0e+12
    >>> img = f["Galaxies/Particles/gas/Images/dim2/Masses"][0] # First galaxy, gas particles, Masses field image
    >>> img.shape
    (64, 64)
    >>> f.close()
    
    This data file is later used to build the morphology model.
    """
    n_galaxies = len(halo_ids)

    # If File does not exist, create it
    if not os.path.exists(os.path.join(path, "galaxy_data.hdf5")):
        # Create the HDF5 file and the data structure
        _create_data_structure(
            n_galaxies=n_galaxies,
            image_res=image_res,
            galaxy_parameters=galaxy_parameters,
            particle_types=particle_types,
            fields=fields,
            path=path,
            dim=dim,
        )
    else:
        # Check if overwrite Flag is set
        if overwrite is None:
            # Ask if the file should be overwritten
            if (
                input(
                    f"The file {os.path.join(path,'galaxy_data.hdf5')} already exists. Do you want to overwrite it? (y/n)"
                )
                == "y"
            ):
                overwrite = True
        if overwrite is True:
            print(
                "Overwriting the existing file: ",
                os.path.join(path, "galaxy_data.hdf5"),
            )
            _create_data_structure(
                n_galaxies=n_galaxies,
                image_res=image_res,
                galaxy_parameters=galaxy_parameters,
                particle_types=particle_types,
                fields=fields,
                path=path,
                dim=dim,
            )
        else:
            print("Loading the existing file: ", os.path.join(path, "galaxy_data.hdf5"))
    # Calculate the images
    _calculate_images(
        simulation,
        halo_ids,
        fields,
        plot_factor,
        image_res,
        path,
        resume=resume,
        **kwargs,
    )


def main():
    # Parse the command line arguments
    parser = argparse.ArgumentParser(
        description="Generate the data for the morphology model."
    )
    parser.add_argument(
        "--config," "-c",
        help="Path to the configuration file",
        type=str,
        default="./config.json",
        required=True,
        dest="config",
    )
    parser.add_argument(
        "--overwrite," "-ow",
        help="Flag wheter to overwrite existing galaxy data.",
        type=bool,
        dest="overwrite",
    )
    parser.add_argument(
        "--resume," "-r",
        help="Flag wheter to resume at the last index position in the loop.",
        type=bool,
        dest="resume",
    )
    args = parser.parse_args()
    # Load the configuration file
    try:
        with open(args.config) as f:
            config = json.load(f)
    except:
        raise ImportError("Could not load the configuration file: ", args.config)

    # Check if all of the required parameters are in the configuration file
    for key in [
        "simulation",
        "halo_ids",
        "fields",
        "plot_factor",
        "path",
        "img_res",
        "galaxy_parameters",
        "particle_types",
        "GalaxyArgs",
    ]:
        if key not in config:
            raise ImportError(
                "The configuration file does not contain the required parameter: ", key
            )

    # Check if halo_ids in the configuration file is a list
    if (
        isinstance(config["halo_ids"], list)
        or isinstance(config["halo_ids"], np.ndarray)
        or isinstance(config["halo_ids"], int)
    ):
        halo_ids = config["halo_ids"]  # List of halo IDs to calculate the images for

    # If halo_ids is a string, try loading the halo IDs from the file
    elif isinstance(config["halo_ids"], str):
        # Check if the file exists
        if not os.path.isfile(config["halo_ids"]):
            raise ImportError("The file does not exist: ", config["halo_ids"])
        # Check if the file is a text file
        if config["halo_ids"].endswith(".txt"):
            try:
                halo_ids = np.loadtxt(config["halo_ids"], dtype=int)
            except:
                raise ImportError(
                    "Could not load the halo IDs from the file: ", config["halo_ids"]
                )
        elif config["halo_ids"].endswith(".hdf5"):
            try:
                f = h5py.File(config["halo_ids"], "r")
                halo_ids = f["halo_ids"][:]
                f.close()
            except:
                raise ImportError(
                    "Could not load the halo IDs from the file: ", config["halo_ids"]
                )
        elif config["halo_ids"].endswith(".npy"):
            try:
                halo_ids = np.load(config["halo_ids"])
            except:
                raise ImportError(
                    "Could not load the halo IDs from the file: ", config["halo_ids"]
                )
        print("Loaded the halo IDs from the file: ", config["halo_ids"])
    else:
        raise ImportError(
            "The halo_ids parameter in the configuration file should be a list/np.ndarray/int or a string as a path to a file. Suppoorted file formats are: .txt, .hdf5, .npy"
        )

    # Set up the parameters to call the generate_data() method
    try:
        simulation = config[
            "simulation"
        ]  # Simulation name, used to initialise the Galaxy class
        particle_types = config["particle_types"]
        img_res = config["img_res"]
        plot_factor = config["plot_factor"]
        path = config["path"]  # Path where the HDF5 file will be saved
        galaxy_parameters = config[
            "galaxy_parameters"
        ]  # List of galaxy parameters to be saved
        fields = config["fields"]  # Dictionary of fields to be saved
        dim = config["dim"]
        kwargs = config["GalaxyArgs"]  # Keyword arguments passed to the Galaxy class

    except:
        raise ImportError(
            "Could not load the parameters from the configuration file. Please check the documentation for the correct format."
        )

    print("Config loaded successfully.")
    # Call the generate_data() method
    generate_data(
        simulation=simulation,
        halo_ids=halo_ids,
        fields=fields,
        plot_factor=plot_factor,
        image_res=img_res,
        galaxy_parameters=galaxy_parameters,
        particle_types=particle_types,
        path=path,
        overwrite=args.overwrite,
        resume=args.resume,
        dim=dim,
        **kwargs,
    )


if __name__ == "__main__":
    main()
