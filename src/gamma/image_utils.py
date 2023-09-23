''' 
Coordinate Rotation and Image Render Modules for the Galaxy Class
    
This module contains the coordinate rotation functions and the image render functions for the Galaxy class defined in the galaxy.py file.
The image render functions are called by the get_image function of the Galaxy class defined in the galaxy.py file.

'''
import numpy as np

from swiftsimio.visualisation.projection import scatter as scatter2D
from swiftsimio.visualisation.volume_render import scatter as scatter3D

import matplotlib.pyplot as plt
import os 
from tqdm import trange, tqdm

from sklearn.decomposition import PCA



def image3D(coordinates, R_half, weights, smoothing_length, plot_factor = 10, res = 64):
    ''' Image Render Module for 3D images.
    
    This function renders a 3D image of the given field. The image is rendered using the scatter function from the swiftsimio.visualisation.volume_render module.
    The image is calculated in plot_factor*R_half. The image is res x res x res pixels.
    
    Parameters
    ----------
    coordinates : numpy.array
        The coordinates of the particles. The Galaxy should be centered at the origin and already rotated to the xy-plane
    R_half : float
        The half mass radius of the galaxy used to set the plot range.
    weights : numpy.array
        The weights of the particles. This is the field that is rendered.
    smoothing_length : numpy.array
        The smoothing length of the particles used for the SPH kernel.    
    plot_factor : float
        The factor by which the image is zoomed in. The image is calculated for -plot_factor*R_half < x,y,z < plot_factor*R_half
    res : int
        The resolution of the image. The image is res x res pixels. The default is 64. 

    Returns
    -------
    numpy.array
        The rendered image.
    '''
    
    plot_range = plot_factor*R_half
    
    x = coordinates[:,0].copy()
    y = coordinates[:,1].copy()
    z = coordinates[:,2].copy()
    
    m =  weights
    
    h = smoothing_length.copy()
    
    #Transform Particles s.t -factor*r_halfmassrad < x <factor*r_halfmassrad -> 0 < x <1
    x = x/(2*plot_range) +1/2  
    y = y/(2*plot_range) +1/2
    z = z/(2*plot_range) +1/2

    h = h/(2*plot_range)
    
    SPH_hist = scatter3D(x=x, y = y,z = z,h = h, m = m ,res= res)
        
    return(SPH_hist)
def image2D(coordinates, R_half, weights, smoothing_length, plot_factor = 10, res = 64):
    ''' Image Render Module for 2D images.
    
    This function renders a 2D image of the given field. The image is rendered using the scatter2D function from the swiftsimio.visualisation.projection module.
    The image is rendered in the xy-plane. The image is calculated in plot_factor*R_half. The image is res x res pixels.
    
    Parameters
    ----------
    coordinates : numpy.array
        The coordinates of the particles. The Galaxy should be centered at the origin and already rotated to the xy-plane
    R_half : float
        The half mass radius of the galaxy used to set the plot range.
    weights : numpy.array
        The weights of the particles. This is the field that is rendered.
    smoothing_length : numpy.array
        The smoothing length of the particles used for the SPH kernel.    
    plot_factor : float
        The factor by which the image is zoomed in. The image is calculated for -plot_factor*R_half < x < plot_factor*R_half
    res : int
        The resolution of the image. The image is res x res pixels. The default is 64. 

    Returns
    -------
    numpy.array
        The rendered image.
    
    '''
    
    plot_range = plot_factor*R_half
    
    x = coordinates[:,0].copy()
    y = coordinates[:,1].copy()
    
    m =  weights
    
    h = smoothing_length.copy()
    
    #Transform Particles s.t -factor*r_halfmassrad < x <factor*r_halfmassrad -> 0 < x <1
    x = x/(2*plot_range) +1/2  
    y = y/(2*plot_range) +1/2

    h = h/(2*plot_range)
    
    SPH_hist = scatter2D(x=x, y = y,h = h, m = m ,res= res)
        
    return(SPH_hist)






def clip_image(data, lower = 0.1, upper = 1.):
    """Clip image to [lower,upper] quantile.
    
    This function is called by the get_image function of the Galaxy class defined in the galaxy.py file. It clips the image to the [lower,upper] quantile.

    Parameters
    ----------
    data : numpy.array
        The image to be clipped.
    lower : float, optional
        The lower quantile. The default is 0.1.
    upper : float, optional
        The upper quantile. The default is 1.

    Returns
    -------
    numpy.array
        The clipped image.
    """    
    hist = data.copy()
    L,U = np.quantile(hist,[lower,upper])
    hist = np.clip(hist, L, U)
    return(hist)




def norm(x, takelog = True, plusone = True, clip = True, lower = 0.1, upper = 1.):
    '''Normalize image.

    This function is called by the get_image function of the Galaxy class defined in the galaxy.py file. It normalizes the image by taking the log10 of the image and clipping it to the [lower,upper] quantile.
    For that we use a mask to ignore the zero values. The image is normalized to the range [0,1].
    
    Parameters
    ----------
    x : numpy.array
        The image to be normalized.
    takelog : bool, optional
        If True the log10 of the image is taken. The default is True.
    plusone : bool, optional
        If True 1 is added to the image. The default is True.
    clip : bool, optional
        If True the image is clipped to the [lower,upper] quantile. The default is True.
    lower : float, optional
        The lower quantile. The default is 0.1. Only used if clip = True.
    upper : float, optional
        The upper quantile. The default is 1.. Only used if clip = True.
        
    Returns
    -------
    numpy.array
        The normalized image.
    '''
    x = np.nan_to_num(x)
    x = x+1 if plusone else x
    mask = np.where(x!=0)
    
    x[mask] = np.log10(x[mask]) if takelog else x[mask]
    
    x[mask] = clip_image(x[mask], lower = lower, upper = upper) if clip else x[mask]
    
    
    x[mask] -= x[mask].min()
    x[mask]/=x[mask].max()
    return(x)










#------------- Coordinate Rotation -------------
def radial_distance(coords,center):
    '''Calculate the radial distance of a set of coordinates to a center. Used to calculate the moment of inertia tensor.

    Parameters
    ----------
    coords : numpy.array
        The coordinates of the particles.
    center : numpy.array
        The center of the galaxy.

    Returns
    -------
    numpy.array
        The radial distance of the particles to the center.
    '''
    d = coords-center
    r_i = d**2
    r= np.sqrt(np.sum(r_i, axis = 1))
    return(r)

def moment_of_intertia_tensor(coordinates, particle_masses, rHalf, subhalo_pos): 
    '''Calculate the moment of inertia tensor of a galaxy.
    
    The moment of inertia tensor is calculated to determine the orientation of the galaxy. 
    The tensor is calculated in the center of mass frame of the galaxy using
    
    I_ij = sum(m_i*(x_i-x_cm)*(x_j-x_cm)).
    
    for the particles within 2*rHalf of the center of the galaxy. 

    Parameters
    ----------
    coordinates : numpy.array
        The coordinates of the particles.
    particle_masses : numpy.array
        The masses of the particles.
    rHalf : float
        The half mass radius of the galaxy.
    subhalo_pos : numpy.array
        The center of the galaxy.

    Returns
    -------
    numpy.array
        The moment of inertia tensor of the galaxy.
    '''

    rad = radial_distance(coords = coordinates, center = subhalo_pos) 
    wGas = np.where((rad <= 2.0*rHalf))[0] 
    masses = particle_masses[wGas] 
    xyz = coordinates[wGas,:]   

    xyz = np.squeeze(xyz)


    if xyz.ndim == 1:
        xyz = np.reshape( xyz, (1,3) )

    for i in range(3):
        xyz[:,i] -= subhalo_pos[i]

    I = np.zeros( (3,3), dtype='float32' )

    I[0,0] = np.sum( masses * (xyz[:,1]*xyz[:,1] + xyz[:,2]*xyz[:,2]) )
    I[1,1] = np.sum( masses * (xyz[:,0]*xyz[:,0] + xyz[:,2]*xyz[:,2]) )
    I[2,2] = np.sum( masses * (xyz[:,0]*xyz[:,0] + xyz[:,1]*xyz[:,1]) )
    I[0,1] = -1 * np.sum( masses * (xyz[:,0]*xyz[:,1]) )
    I[0,2] = -1 * np.sum( masses * (xyz[:,0]*xyz[:,2]) )
    I[1,2] = -1 * np.sum( masses * (xyz[:,1]*xyz[:,2]) )
    I[1,0] = I[0,1]
    I[2,0] = I[0,2]
    I[2,1] = I[1,2]

    return I



def rotation_matrix(inertiaTensor, return_value = "face-on"):
    '''Calculate the rotation matrix to orient the galaxy.
    
    The rotation matrix is calculated from the moment of inertia tensor using the eigenvalues and eigenvectors of the tensor. The rotation matrix is used to rotate the galaxy to the x,y,z axes.
    
    Parameters
    ----------
    inertiaTensor : numpy.array
        The moment of inertia tensor of the galaxy. This can be calculated using the moment_of_intertia_tensor function.
    return_value : str, optional
        The orientation of the galaxy. The default is "face-on". Can be "face-on" or "edge-on".

    Returns
    -------
    numpy.array
        The rotation matrix to orient the galaxy.
    
    '''

    # get eigen values and normalized right eigenvectors
    eigen_values, rotation_matrix = np.linalg.eig(inertiaTensor)

    # sort ascending the eigen values
    sort_inds = np.argsort(eigen_values)
    eigen_values = eigen_values[sort_inds]

    # permute the eigenvectors into this order, which is the rotation matrix which orients the
    # principal axes to the cartesian x,y,z axes, such that if axes=[0,1] we have face-on
    new_matrix = np.matrix( (rotation_matrix[:,sort_inds[0]],
                             rotation_matrix[:,sort_inds[1]],
                             rotation_matrix[:,sort_inds[2]]) )

    phi = np.random.uniform(0, 2*np.pi)
    theta = np.pi / 2
    psi = 0

    A_00 =  np.cos(psi)*np.cos(phi) - np.cos(theta)*np.sin(phi)*np.sin(psi)
    A_01 =  np.cos(psi)*np.sin(phi) + np.cos(theta)*np.cos(phi)*np.sin(psi)
    A_02 =  np.sin(psi)*np.sin(theta)
    A_10 = -np.sin(psi)*np.cos(phi) - np.cos(theta)*np.sin(phi)*np.cos(psi)
    A_11 = -np.sin(psi)*np.sin(phi) + np.cos(theta)*np.cos(phi)*np.cos(psi)
    A_12 =  np.cos(psi)*np.sin(theta)
    A_20 =  np.sin(theta)*np.sin(phi)
    A_21 = -np.sin(theta)*np.cos(phi)
    A_22 =  np.cos(theta)

    random_edgeon_matrix = np.matrix( ((A_00, A_01, A_02), (A_10, A_11, A_12), (A_20, A_21, A_22)) )

    # prepare return with a few other useful versions of this rotation matrix
    r = {}
    r['face-on'] = new_matrix
    r['edge-on'] = np.matrix( ((1,0,0),(0,0,1),(0,-1,0)) ) * r['face-on'] # disk along x-hat
    r['edge-on-smallest'] = np.matrix( ((0,1,0),(0,0,1),(1,0,0)) ) * r['face-on']
    r['edge-on-y'] = np.matrix( ((0,0,1),(1,0,0),(0,-1,0)) ) * r['face-on'] # disk along y-hat
    r['edge-on-random'] = random_edgeon_matrix * r['face-on']
    r['phi'] = phi
    
    return r[return_value]


def face_on_rotation(rHalf, subhalo_pos,coordinates, particle_masses, return_rotation_matrix = False):
    '''Rotate the galaxy to face-on orientation.

    The galaxy is rotated to face-on orientation using the rotation matrix calculated from the moment of inertia tensor.

    Parameters
    ----------
    rHalf : float
        The half mass radius of the galaxy. 
    subhalo_pos : numpy.array
        The position of the subhalo.
    coordinates : numpy.array
        The coordinates of the particles.
    particle_masses : numpy.array
        The masses of the particles.

    Returns
    -------
    numpy.array
        The rotated coordinates of the particles.
    '''
    I = moment_of_intertia_tensor(coordinates=coordinates, rHalf=rHalf,particle_masses=particle_masses, subhalo_pos=subhalo_pos)
    rot_matrix = rotation_matrix(inertiaTensor=I, return_value = "face-on")
    
    
    #Rotate Particles to face-on with the calculated Rotation Matrix
    pos = coordinates- subhalo_pos
    rot_pos= np.dot(rot_matrix, pos.T).T
    rotated_particles = np.asarray(rot_pos)
    if return_rotation_matrix == True:
        return rotated_particles, rot_matrix
    return rotated_particles    






#------------- Vertical Rotation -------------

def calc_rotation_matrix(angle):
    '''Calculate the rotation matrix around the z-axis for a given angle.

    Calculate the rotation matrix around the z-axis for a given angle. The rotation matrix is used to rotate the galaxy around the z-axis.
    
    Parameters
    ----------
    angle : float
        The angle to rotate the galaxy around the z-axis. Needs to be in radians.

    Returns
    -------
    numpy.array
        The rotation matrix.
    '''
    rot_mat = np.array([[np.cos(angle),-np.sin(angle),0],
                           [np.sin(angle), np.cos(angle),0],
                           [0 , 0, 1]])
    return(rot_mat)

def get_horizontal_angle(img):
    '''Calculate the angle of the galaxy bar in the image.

    Uses the PCA to calculate the angle of the galaxy in the image. The PCA is calculated on the pixels with a value above the 75% quantile in order to only take the brightest pixels into account. 
    The angle is calculated from the first principal component, which represents the direction of maximum variance in the data i.e. the direction of the galaxy bar.

    Parameters
    ----------
    img : numpy.array
        The image of the galaxy.

    Returns
    -------
    float
        The angle of the galaxy in the image in radians.
    '''
    fit=PCA(n_components=2).fit(np.argwhere(img>=np.quantile(img,.75)))
    return np.arctan2(*fit.components_[0])


def horizontal_rotation(img, coordinates, halfmassrad,plotfactor=10, return_rotation_matrix = False):
    '''Rotate the galaxy to be horizontal.
    
    The galaxy is rotated to be horizontal using the angle of the galaxy bar in the image calculated with the PCA.
    
    Parameters
    ----------
    img : numpy.array
        The image of the galaxy.
    coordinates : numpy.array
        The coordinates of the particles.
    halfmassrad : float
        The half mass radius of the galaxy.
    plotfactor : float, optional
        The factor to multiply the half mass radius with to get the size of the image. The default is 10.

    Returns
    -------
    numpy.array
        The rotated coordinates of the particles.
    '''
    #First Create Dummy hist
    hist = img.copy()
    hist = clip_image(hist, lower = 0.9, upper = 1.0)
    angle = get_horizontal_angle(hist)
    
    #Rotate
    horizontal_rotation_matrix = calc_rotation_matrix(-angle)
    rotated_coordinates = np.dot(horizontal_rotation_matrix, coordinates.T).T
    if return_rotation_matrix == True:
        return rotated_coordinates, horizontal_rotation_matrix
    return rotated_coordinates



#------------- Visualisation -------------
import plotly.graph_objects as go
def volume(hist ,opacity = .1, isomin = None, isomax = None, surface_count = 30, add_small_number = True, norm_hist = True, **kwargs):
    '''Visualise a 3D histogram as a volume.
    
    Uses plotly to visualise a 3D histogram as a volume. The volume can be normalised and a small number can be added to the histogram to avoid visualising empty space.
    
    Parameters
    ----------
    hist : numpy.array
        The 3D histogram to visualise.
    opacity : float, optional
        The opacity of the volume. The default is .1.
    isomin : float, optional
        The minimum value of the volume. The default is 0.
    isomax : float, optional
        The maximum value of the volume. The default is None.
    surface_count : int, optional
        The number of surfaces to use for the volume. The default is 30.
    add_small_number : bool, optional
        Whether to add a small number to the histogram to avoid visualizing empty space. The default is True.
    norm_hist : bool, optional
        Whether to normalise the histogram. The default is True.
    **kwargs :
        Additional arguments to pass to the normalisation function.
    '''
    if isomin is None: isomin = hist.min()
    if isomax is None: isomax = hist.max()
    data_hist =hist.copy()
    
    if norm_hist == True:
        data_hist = norm(data_hist, **kwargs)
    if add_small_number == True:
        data_hist += 1e-10
    xx, yy, zz = np.where(data_hist != 0)
    s = data_hist[xx,yy,zz]
    fig = go.Figure(data=go.Volume(
        x=xx,
        y=yy,
        z=zz,
        value=s,
        isomin=isomin,
        isomax=isomax,
        opacity=opacity, # needs to be small to see through all surfaces
        surface_count=surface_count,# needs to be a large number for good volume rendering
        ))
    fig.update_layout(scene_xaxis_showticklabels=False,
                  scene_yaxis_showticklabels=False,
                  scene_zaxis_showticklabels=False)
    
    fig.show()