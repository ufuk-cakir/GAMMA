from .simulations import *  # Import all Galaxy Classes of the simulations
import numpy as np
import sys
from .image_utils import image2D, image3D, norm, face_on_rotation, horizontal_rotation

def _str_to_class(classname):
    """Converts a string to a class."""
    # Check if the class is defined in the simulations.py file
    if hasattr(sys.modules[__name__], classname):
        # Return the class
        return getattr(sys.modules[__name__], classname)
    else:
        raise ValueError("Simulation {} not supported.".format(classname))


class Galaxy:
    """Base class for all galaxies.
    This module contains the Galaxy class as the main class of the package. It is used to load a galaxy from a simulation and to render images of the galaxy.
    One can specify the simulation the galaxy is from and the halo id of the galaxy. The Galaxy class will then load the galaxy data from the simulation based 
    on the Class defined in the simulatoins module and rotate the galaxy to face-on and horizontal orientation.
    The get_image method can then be used to render images of different fields of the galaxy, and chose if they should be mass weighted or not.


    Parameters
    ----------
    simulation : str, optional
        The simulation the galaxy is from. Curenntly only "IllustrisTNG" is supported. You can add your own simulation by adding a new class to the simulations.py file.

    **kwargs : dict
        Additional arguments for the galaxy class. This is used to initialize the galaxy class of a specific simulation. See the documentation of the galaxy class defined in simulations.py for more information.
    
    Attributes
    ----------
    galaxy_object : object
        The galaxy object of the simulation. This is the class defined in the simulations.py file.
    plot_factor : int
        Factor used in the horizontal_rotation method defined in the rotation.py module to scale the image. The default is 10. For more information see the documentation of the horizontal_rotation method.
    res : int
        Resolution of the image. The default is 64. For more information see the documentation of the horizontal_rotation method.
    smoothing_length : numpy.array
        Smoothing length of the galaxy. This is used for the image rendering.
    coordinates : numpy.array
        Coordinates of the particles. This is used for the image rendering.
    rotated_flag : bool
        Flag to check if the galaxy is already rotated. This is used for the image rendering.
        
    Examples
    --------
    >>> galaxy = Galaxy("IllustrisTNG", halo_id=0, particle_type="stars") # Load a galaxy from the IllustrisTNG simulation. Note that the Class IllustrisTNG needs to be defined in the simulations.py file
    >>> galaxy.get_image("GFM_StellarFormationTime", mass_weighted=True) # Render an image of the age of the stars in the galaxy
    >>> galaxy.get_rotation_matrix() # Get the rotation matrix used to rotate the galaxy to face-on and horizontal orientation
    """

    def __init__(self, simulation="IllustrisTNG", **kwargs):
        self.simulation = simulation
        # Gemeral Galaxy Properties??
        self.rotated_flag = False

        # Load the defined Galaxy Class
        self.galaxy_object = _str_to_class(self.simulation)(**kwargs)
        # Set default Atributes for the image rendering
        self.plot_factor = 10
        self.res = 64
        if hasattr(self.galaxy_object, "hsml"):
            self.smoothing_length = self.galaxy_object.hsml
        else:
            # Calculate smoothing length maybe later (TODO)
            raise AttributeError("Galaxy object does not have a smoothing length.")

        self.coordinates = self._rotate_galaxy()
        

    def get_coordinates(self):
        """
        Get the coordinates of the particles.

        The coordinates are the coordinates of the particles in the galaxy. They are rotated to face-on and horizontal orientation.
        
        Returns
        -------
        numpy.array
            The coordinates of the particles.
        """
        return self.coordinates

    def _rotate_galaxy(self, _plotfactor=10):
        """Rotate the galaxy to face-on and horizontal orientation.

        This function is called when the galaxy object is initialized. It is not necessary to call it again. First the galaxy is rotated face-on and then horizontal.
        The rotation is done using the rotation.py module.

        Parameters
        ----------
        _plotfactor : int, optional
            Factor used in the horizontal_rotation method defined in the rotation.py module to scale the image. The default is 10. For more information see the documentation of the horizontal_rotation method.

        Returns
        -------
        numpy.array
            The rotated coordinates of the galaxy.
        """
        if self.rotated_flag:
            return self.coordinates

        face_on_rotated_coords, rotation_matrix_face_on = face_on_rotation(
            coordinates=self.particle_coordinates,
            particle_masses=self.particle_masses,
            rHalf=self.halfmassrad,
            subhalo_pos=self.center,
            return_rotation_matrix=True,
        )
        # maybe horizontal rotataion is not working properly
        # Create temporary image to get the rotation angle: Maybe there is a better way to do this. Only calculating a hist does not work properly.
        img = self._render_image_2D(
            field=self.particle_masses, coordinates=face_on_rotated_coords
        )
        horizontal_rotated_coords, rotation_matrix_horizontal = horizontal_rotation(
            img=img,
            coordinates=face_on_rotated_coords,
            halfmassrad=self.halfmassrad,
            plotfactor=_plotfactor,
            return_rotation_matrix=True,
        )
        self._total_rotation_matrix = np.dot(
            rotation_matrix_horizontal, rotation_matrix_face_on
        )
        self.rotated_flag = True
        return horizontal_rotated_coords

    def __getattr__(self, name):
        """Delegate all other attributes to the galaxy object. This is used to access the attributes of the simulation galaxy class defined in the simulations.py file,
        if they are not defined in the general Galaxy class.
        """
        return getattr(self.galaxy_object, name)

    # -----------------Image Rendering-----------------#

    def _render_image_2D(
        self,
        field,
        coordinates=None,
    ):
        """Image Render Module for 2D images.
        This function is called by the get_image function. It renders the image using the image2D function from the image_modules.py file.
        You can change the image rendering by implementing your own image rendering function here.

        For more information of the default render method see the documentation of the image2D function.
        """

        # Check if all parameters have the same length
        if coordinates is not None:
            if (
                len(coordinates) != len(self.smoothing_length)
                or len(coordinates) != len(field)
                or len(self.smoothing_length) != len(field)
            ):
                raise ValueError(
                    "Coordinates, smoothing length and field must have the same length."
                )
            img = image2D(
                coordinates=coordinates,
                R_half=self.halfmassrad,
                weights=field,
                smoothing_length=self.smoothing_length,
                plot_factor=self.plot_factor,
                res=self.res,
            )
        else:
            if (
                len(self.coordinates) != len(self.smoothing_length)
                or len(self.coordinates) != len(field)
                or len(self.smoothing_length) != len(field)
            ):
                raise ValueError(
                    "Coordinates, smoothing length and field must have the same length."
                )

            img = image2D(
                coordinates=self.coordinates,
                R_half=self.halfmassrad,
                weights=field,
                smoothing_length=self.smoothing_length,
                plot_factor=self.plot_factor,
                res=self.res,
            )
        return img

    def _render_image_3D(
        self,
        field,
        coordinates=None,
    ):
        """Image Render Module for 3D images.
        This function is called by the get_image function. It renders the image using the image3D function from the image_modules.py file.
        You can change the image rendering by implementing your own image rendering function here.

        For more information of the default render method see the documentation of the image3D function.
        """

        # Check if all parameters have the same length
        if coordinates is not None:
            if (
                len(coordinates) != len(self.smoothing_length)
                or len(coordinates) != len(field)
                or len(self.smoothing_length) != len(field)
            ):
                raise ValueError(
                    "Coordinates, smoothing length and field must have the same length."
                )
            img = image3D(
                coordinates=coordinates,
                R_half=self.halfmassrad,
                weights=field,
                smoothing_length=self.smoothing_length,
                plot_factor=self.plot_factor,
                res=self.res,
            )
        else:
            if (
                len(self.coordinates) != len(self.smoothing_length)
                or len(self.coordinates) != len(field)
                or len(self.smoothing_length) != len(field)
            ):
                raise ValueError(
                    "Coordinates, smoothing length and field must have the same length."
                )

            img = image3D(
                coordinates=self.coordinates,
                R_half=self.halfmassrad,
                weights=field,
                smoothing_length=self.smoothing_length,
                plot_factor=self.plot_factor,
                res=self.res,
            )
        return img

    def render_image(self, field, dim, coordinates=None):
        """
        Render the image of a given field.
        
        This function is called by the get_image function. It renders the image using the _render_image_2D or _render_image_3D function based on the dimension of the image.
        You can change the image rendering by implementing your own image rendering function in the _render_image_2D or _render_image_3D function.

        Parameters
        ----------
        field : str
            The field to be rendered. Can be any field that is available in the snapshot. Used to call the get_field function of the galaxy object.
        dim : int
            The dimension of the image. Can be either 2 or 3.
        coordinates : numpy.array, optional
            The coordinates of the particles. If not given, the coordinates of the galaxy object are used. The default is None.

        Raises
        ------
        ValueError
            If the dimension is not 2 or 3.

        Returns
        -------
        numpy.array
            The rendered image.
        """
        if dim == 2:
            return self._render_image_2D(field, coordinates)
        elif dim == 3:
            return self._render_image_3D(field, coordinates)
        else:
            raise ValueError("The dimension must be either 2 or 3.")

    def get_image(
        self,
        field,
        mass_weighted=True,
        normed=False,
        res=None,
        plotfactor=None,
        dim=2,
        **kwargs
    ):
        """
        Get the image of a given field.

        This function renders the image of a given field, which can be any field that is available in the snapshot and can be accessed by the get_field function of the galaxy object.
        The images can be mass weighted and normalized.

        It uses the _render_image_2D function to render the image. You can change the image rendering by implementing your own image rendering function there.

        Parameters
        ----------
        field : str
            The field to be rendered. Can be any field that is available in the snapshot. Used to call the get_field function of the galaxy object.
        mass_weighted : bool, optional
            If True, the image is mass weighted. The default is True.
        normed : bool, optional
            If True, the image is normalized. The default is False.
        res : int, optional
            The resolution of the image. The default is None. If None, the resolution is set to the default value of the Galaxy class, which is 64.
        plotfactor : int, optional
            The plotfactor used to scale the image. The default is None. If None, the plotfactor is set to the default value of the Galaxy class, which is 10.
        dim : int, optional
            The dimension of the image. The default is 2. If 2, the image is rendered as a 2D image. If 3, the image is rendered as a 3D image.
        **kwargs : dict
            Additional arguments for the normalization function. This is only used if normed is True. For more information see the documentation of the norm function defined in the image_modules.py file.

        Returns
        -------
        numpy.array
            The image of the given field.

        Examples
        --------
        >>> galaxy = Galaxy(halo_id=0, particle_type="stars", base_path="data", snapshot=99)
        >>> image = galaxy.get_image("Masses", mass_weighted=False, normed=True)
        >>> plt.imshow(image)
        """
        # Set the resolution and plotfactor
        if res is not None:
            self.res = res
        if plotfactor is not None:
            self.plot_factor = plotfactor

        # Check if dim is 2 or 3
        if dim not in [2, 3]:
            raise ValueError("dim must be 2 or 3. This is the dimension of the image.")

        if mass_weighted:
            # First create the mass image
            masses = self.get_field("Masses")
            mass_img = self.render_image(masses, dim)
            if field == "Masses":
                # If the field is mass, return the mass image
                image = mass_img
            else:
                # Create the mass weighted field image.
                weights = self.get_field(field) * masses  # mass weighted weights
                weights_img = self.render_image(weights, dim)

                # Avoid division by zero: If the mass image value is zero, return the weights image value
                mask = np.where(mass_img != 0)
                # image = np.zeros_like(mass_img)
                image = weights_img.copy()
                image[mask] = weights_img[mask] / mass_img[mask]
                # image = weights_img/mass_img

        else:
            # Not mass weighted. Return the field image
            weights = self.get_field(field)
            image = self.render_image(weights, dim)

        if normed:
            image = norm(image, **kwargs)

        return image

    def get_rotation_matrix(self):
        """Get the total rotation matrix of the galaxy.

        The total rotation matrix is the rotation matrix that combines the rotation face on rotation and horizontal rotation.
        
        Returns
        -------
        numpy.array
            The total rotation matrix of the galaxy.
        """
        
        return self._total_rotation_matrix
    
    def get_coordinates(self):
        """Get the coordinates of the particles of the galaxy.

        The coordinates are rotated by the total rotation matrix of the galaxy.

        Returns
        -------
        numpy.array
            The correctly rotated coordinates of the particles of the galaxy.
        """
        return self.coordinates
    
    def get_rotated_velocities(self):
        """Get the correctly rotated velocities of the particles of the galaxy.

        The velocities are rotated by the total rotation matrix of the galaxy.

        Returns
        -------
        numpy.array
            The correctly rotated velocities of the particles of the galaxy.
        """
        # Rotate the velocities
        velocities = self.get_field("Velocities")
        self.velocities = np.dot(self._total_rotation_matrix, velocities.T).T
        return self.velocities
    
 