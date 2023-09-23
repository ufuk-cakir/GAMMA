'''This module contains the class for specific simulations. Currently only IllustrisTNG is implemented.
You can add your own simulation by creating a new class based on the following template:
class YourSimulation():
    def __init__(self, halo_id, particle_type, data_path):
        self.data_path = data_path #Path to the data.
        self.halo_id= halo_id #Halo ID of a subhalo
        self.particle_type = particle_type #Particle type of the subhalo
        
        # Add any other attributes you need. 
        # These can later be accessed by the galaxy object using the getattr function, and can be saved in the HDF5 file.
        
        self._load_data()
    def _load_data(self):
        #Load the data from the snapshot and subhalo catalog.
        #The data needs to be stored in the following attributes:
        
        self.particle_coordinates #Coordinates of the particles,used for face-on rotation and image rendering.
        self.particle_masses #Masses of the particles, used for face-on rotation and image rendering.
        self.center #Center of the subhalo. Used for centering the galaxy
        self.halfmassrad #Half mass radius of the subhalo. Used for the image rendering
        self.hsml #Smoothing length of the subhalo. Used for the image rendering.
        
        
    def get_field(self, field):
        #Get the field from the snapshot.
        #The field should be returned as a numpy array and converted to physical units.
        #This field is then later used to create the weighted image.

        return field

Note that the class name should be the same as the simulation name, since this is used to create the galaxy object.
'''



import numpy as np
from astropy.cosmology import Planck15 as cosmo
import requests
import os
import h5py


class illustrisAPI():
    
    DATAPATH = "./tempdata"
    URL = "http://www.tng-project.org/api/"
    
    def __init__(self,api_key,particle_type = "stars",simulation = "TNG100-1",snapshot = 99,):
        ''' Illustris API class.
        
        Class to load data from the Illustris API.
        
        Parameters
        ----------
        api_key : str
            API key for the Illustris API.
        particle_type : str
            Particle type to load. Default is "stars".
        simulation : str
            Simulation to load from. Default is "TNG100-1".
        snapshot : int
            Snapshot to load from. Default is 99.
        '''
        
        self.headers = {"api-key":api_key}
        self.particle_type = particle_type
        self.snapshot = snapshot
        self.simulation = simulation
        self.baseURL = f"{self.URL}{self.simulation}/snapshots/{self.snapshot}"
     
    def get(self, path, params = None, name = None):
        ''' Get data from the Illustris API.
        
        Parameters
        ----------
        path : str
            Path to load from.
        params : dict
            Parameters to pass to the API.
        name : str
            Name to save the file as. If None, the name will be taken from the content-disposition header.

        Returns
        -------
        r : requests object
            The requests object.
        
        '''
        
        os.makedirs(self.DATAPATH,exist_ok=True)
        r = requests.get(path, params=params, headers=self.headers)
        # raise exception if response code is not HTTP SUCCESS (200)
        r.raise_for_status()
        if r.headers['content-type'] == 'application/json':
            return r.json() # parse json responses automatically
        if 'content-disposition' in r.headers:
            filename = r.headers['content-disposition'].split("filename=")[1] if name is None else name
            with open(f"{self.DATAPATH}/{filename}.hdf5", 'wb') as f:
                f.write(r.content)
            return filename # return the filename string
        return r
        
    def get_subhalo(self, id):
        ''' Get subhalo data from the Illustris API.
        
        Returns the subhalo data for the given subhalo ID.
        
        Parameters
        ----------
        id : int
            Subhalo ID to load.

        Returns
        -------
        r : dict
            The subhalo data.
            
        '''
        
        return self.get(f'{self.baseURL}/subhalos/{id}')
    
    def load_hdf5(self, filename):
        ''' Load HDF5 file.
        
        Loads the HDF5 file with the given filename.
        
        Parameters
        ----------
        filename : str
            Filename to load.

        Returns
        -------
        returndict : dict
            Dictionary containing the data from the HDF5 file.
        '''
        # Check if filename ends with .hdf5
        if filename.endswith(".hdf5"):
            filename = filename[:-5]
            
        returndict = dict()
        
        with h5py.File(f"{self.DATAPATH}/{filename}.hdf5", 'r') as f:
            for type in f.keys():
                if type == "Header":
                    continue
                if type.startswith("PartType"):
                    for fields in f[type].keys():
                        returndict[fields] = f[type][fields][()]
                        
        return returndict
    
    def get_particle_data(self,id, fields):
        ''' Get particle data from the Illustris API.
        
        Returns the particle data for the given subhalo ID.

        Parameters
        ----------
        id : int
            Subhalo ID to load.
        fields : str or list
            Fields to load. If a string, the fields should be comma-separated.
        
        Returns
        -------
        data : dict
            Dictionary containing the particle data in the given fields.
        ''' 
        # Get fields in the right format
        if isinstance(fields, str):
            fields = [fields]
        fields = ','.join(fields)
        
        url = f'{self.baseURL}/subhalos/{id}/cutout.hdf5?{self.particle_type}={fields}'
        self.get(url, name = "cutout")
        data = self.load_hdf5("cutout")
        
        return data
        
        





try:
    import illustris_python as il 
except ImportError:
    print("IllustrisTNG not installed. Please install it or define other simulations in simulations.py.")



_h = cosmo.H(0).value/100 #Hubble constant
_age = cosmo.age(0).value #Age of the universe in Gyr


def select_illustris_galaxies(basepath,snapshot,M_min,M_max, particle_type = "stars"):
    '''Galaxy selection for IllustrisTNG.
    
    Selects all galaxies with stellar mass between M_min and M_max and with SubhaloFlag == 1 (i.e proper galaxies).
    
    Parameters:
    -----------
    basepath: str
        Path to the IllustrisTNG data.
    snapshot: int
        Snapshot number.
    M_min: float
        Minimum stellar mass in Msun/h.
    M_max: float
        Maximum stellar mass in Msun/h.
    particle_type: str
        Particle type of the galaxy. Default: "stars"

    Returns:
    --------
    halo_ids: numpy array
        Array of halo IDs of the selected galaxies.
    '''
    subhalos = il.groupcat.loadSubhalos(basepath, snapshot, fields=["SubhaloMassType", "SubhaloFlag"])
    
    stellar_mass = subhalos['SubhaloMassType'][:,il.util.partTypeNum(particle_type)] * 10**10 / 0.704
    
    # Check if M_Min and M_max are set
    if M_min is None:
        M_min = 0
    if M_max is None:
        M_max = np.inf

    # Get halo IDs of all subhalos with stellar   10^9.5 Msun/h < Mstar < 10^13 Msun/h
    mass_cut = np.where((stellar_mass> M_min) & (stellar_mass < M_max))[0]

    # Get halo IDs of all subhalos with SubhaloFlag == 0 (i.e. no Galaxy and should be ignored)
    flag_cut = np.where(subhalos['SubhaloFlag'] == 1)[0]

    # Get the common halo IDs that satisfy both conditions
    halo_ids = np.intersect1d(mass_cut, flag_cut)
    
    print("Found {} galaxies with stellar mass between {} and {} Msun/h.".format(len(halo_ids), M_min, M_max))
    return halo_ids

def scale_to_physical_units(x, field):
    '''get rid of the Illustris units.'''

    # If the field string contains the word "Mass"
    if 'Mass' in field:
        return x * 1e10 / _h
    if field == 'Masses':
        return x * 1e10 / _h

    elif field == 'Coordinates':
        return x / _h

    elif field == 'SubfindHsml':
        return x / _h

    elif field == 'SubfindDensity':
        return x * 1e10 * _h * _h

    elif field == 'GFM_StellarFormationTime':
        #Calculates Age of Stars
        return (_age-cosmo.age(1 / x - 1).value)*1e9 #Gyr
    elif field =="GFM_Metallicity":
        return(x/0.0127) #Solar Metallicity
    else:
        print("No unit conversion for Field {}. Return without changes.".format(field))
        return x


class IllustrisTNG():
    '''Class for the IllustrisTNG simulation.'''
    
    def __init__(self, halo_id, particle_type,base_path, snapshot):
        self.base_path = base_path
        self.halo_id= halo_id
        self.particle_type = particle_type
        self.snapshot = snapshot
       
        self._load_data() 
    
    def _load_data(self):
        '''Load the data from the snapshot and subhalo catalog.'''
        self.subhalo = il.groupcat.loadSingle(self.base_path, self.snapshot, subhaloID=self.halo_id)
        self.center = self.subhalo['SubhaloPos']
        self.mass = scale_to_physical_units(self.subhalo['SubhaloMassType'][il.util.partTypeNum(self.particle_type)], 'Masses')
        self.halfmassrad = self.subhalo['SubhaloHalfmassRadType'][il.util.partTypeNum(self.particle_type)]
        self.halfmassrad_DM= self.subhalo['SubhaloHalfmassRadType'][il.util.partTypeNum('DM')]
        self.particles = il.snapshot.loadSubhalo(self.base_path, self.snapshot, self.halo_id, self.particle_type)
    
        if self.particle_type == "stars": 
            # Get only real stars, not wind particles
            self.real_star_mask = np.where(self.particles["GFM_StellarFormationTime"]>0)[0]
            self.hsml = self.particles["StellarHsml"][self.real_star_mask]
        else:
            self.real_star_mask = np.ones(len(self.particles["Coordinates"]), dtype=bool)
            #Is this correct? Is this the smoothing length used for visualization?
            self.hsml = self.particles["SubfindHsml"]
        
        self.particle_coordinates = self.particles["Coordinates"][self.real_star_mask]
        self.particle_masses = scale_to_physical_units(self.particles["Masses"][self.real_star_mask], 'Masses')
    def get_field(self, field, particle_type=None):
        '''Load a field from the particle data. Used for the image generation.
        The field is returned as a numpy array and converted to physical units.

        Parameters
        ----------
        field : str
            Name of the field to load. The field should be stored in the snapshot.  
        particle_type : str, optional
            If the field is stored for multiple particle types, this specifies which particle type to return. 
            If None, the field is returned for all particle types. The default is None.

        Returns
        -------
        numpy.array
            The field converted to physical units. 
        

        Examples
        --------
        >>> galaxy = Galaxy("IllustrisTNG", halo_id=0, particle_type="stars")
        >>> galaxy.get_field("GFM_StellarFormationTime")
        '''
        
        if field in self.particles.keys():
            return_field= scale_to_physical_units(self.particles[field][self.real_star_mask], field)
        elif field in self.subhalo.keys():
            return_field= scale_to_physical_units(self.subhalo[field], field)
        else:
            raise ValueError("Field {} not in snapshot.".format(field))

       # If "Type" is in the field name, check which particle type is requested
        if "Type" in field:
            if particle_type is None:
                return return_field
            else:
                return return_field[il.util.partTypeNum(particle_type)]
        else:
            return return_field