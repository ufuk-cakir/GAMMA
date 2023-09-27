import numpy as np 
import argparse
import json


# At this point only illustris galaxies are supported
from .simulations import select_illustris_galaxies


galaxy_selection = select_illustris_galaxies





def main():
    parser = argparse.ArgumentParser(
        description="Select Galaxies from TNG100-1"
    )
    parser.add_argument(
        "--config," "-c",
        help="Path to the configuration file",
        type=str,
        default="./config.json",
        required=True,
        dest="config",
    ) 
    
    args = parser.parse_args()
    
    
    # Load the configuration file
    with open(args.config) as f:
        config = json.load(f)
        
    particle_type = config["GalaxyArgs"]["particle_type"]
    basepath = config["GalaxyArgs"]["base_path"]
    snapshot = config["GalaxyArgs"]["snapshot"]
    halo_ids_path = config["halo_ids"]
    log_m_min = config["log_M_min"]
    log_m_max = config["log_M_max"]
    
    if halo_ids_path is not None:
        raise Exception("Halo IDs already exist. Please delete the file and try again.")
    else:
        halo_ids_path = "./halo_ids.npy"
    # Write the halo ids to the config file
    config["halo_ids"] = halo_ids_path
    with open(args.config, "w") as f:
        json.dump(config, f)
        
    # Select galaxies
    galaxy_ids = select_illustris_galaxies(
        basepath=basepath,
        snapshot=snapshot,
        particle_type=particle_type,
        M_min = 10**log_m_min,
        M_max = 10**log_m_max,
    )
    # Save the galaxy ids  
    np.save(halo_ids_path, galaxy_ids)





if __name__ == "__main__":
    # execute only if run as a script
    main()