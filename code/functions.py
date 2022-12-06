import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
from skimage.feature import blob_log
from apsg import *


microm2m = 1.0E-6
m2microm = 1.0E6
cm = 10**(-7) # H/m  == T·m/A


def random_normal_seeded(mean_value, std_dev, size, seed_value=0):

    """
    Generates pseudo-random normal gaussian distribution, but the seed
    gives more controllable results.

    Parameters:
        * mean_value: float number, corresponds to the mean of the normal distribution
        * std_dev: float number, corresponds to the standart deviation of the normal distribution
        * size: scalar number, corresponds to the size of the distribution array
        * seed_value: float number, it is the seed for the distribution
    """
    
    np.random.seed(seed_value)
    normal_Gaussian_distribution = np.random.normal(mean_value, std_dev, size)
    
    return(normal_Gaussian_distribution)

def random_randint_seeded(min_value, max_value, size, seed_value=0):

    """
    Generates pseudo-random normal gaussian distribution, but the seed
    gives more controllable results.

    Parameters:
        * min_value: float number, corresponds to the minimum value pickable
        * min_value: float number, corresponds to the maximum value pickable
        * size: scalar number, corresponds to the size of the distribution array
        * seed_value: float number, it is the seed for the distribution
    """
    
    np.random.seed(seed_value)
    distribution_array = np.random.randint(min_value, max_value, size)
    
    return(distribution_array)



def magnetic_moment_vector(declination, inclination, magnetization, radius):    
    """
    Calculates the cartesian components of natural remanent magnetization (NRM)
    for uniformily magnetized spheres (mx, my, mz) required in the forward model.

    Parameters
    ----------
    declination : array
        Array containing magnetization declination (0 to 360 degrees)
    inclination : array
        Array containing magnetization inclination (-90 to 90  degress)
    magnetization : array
        Array containing magnetization intensity (A/m)
    magnetization : array
        Array containing sphere's radius (m)

    Returns
    -------
    magnetic_moment : array
        1D array with the magnetic moments concatenated.

    """
    inclination = np.radians(inclination)
    declination = np.radians(declination)
    volume = 4 / 3 * np.pi * radius**3
    moment_amplitude = magnetization * volume
    mx = moment_amplitude * np.cos(inclination) * np.cos(declination)
    my = moment_amplitude * np.cos(inclination) * np.sin(declination)
    mz = moment_amplitude * np.sin(inclination)
    magnetic_moment = np.stack([mx, my, mz], axis=1).ravel()
    
    return magnetic_moment


def sensitivity_matrix(X, Y, Z, Xc, Yc, Zc):     
    """
    Generates the sensitivity matrix (A) for vertical component (z) of magnetization

    Parameters:
        *  X,  Y,  Z: 1D-arrays with x, y and z data positions
        * Xc, Yc, Zc: 1D-arrays with x, y and z data  of the sphere's center positions

    Constant:
        * cm: is a constant given by 4*np.pi/µo = 10**(-7) 
            where µo is the vaccum magnetic permeability
    """
    nsources = Xc.size
    nparameters = nsources * 3
    ndata = X.size
    A = np.empty((ndata, nparameters)) 
    for j in range(nsources):
        dx = X.ravel() - Xc[j]
        dy = Y.ravel() - Yc[j]
        dz = Z.ravel() - Zc[j]
        distance_5 = np.sqrt(dx**2 + dy**2 + dz**2)**5
        A[:, j * 3] = 3 * dx * dz / distance_5  # M_xz
        A[:, j * 3 + 1] = 3 * dy * dz / distance_5  # M_yz
        A[:, j * 3 + 2] = -1 * (dx**2 + dy**2 - 2 * dz**2) / distance_5  # M_zz
    A *= cm
    
    return A
    


def regular(area, shape, z=None):

    """
    Generates the regular grid --> source https://legacy.fatiando.org/cookbook.html

    Parameters:
        * area: array with x axis size (nx) and y axis size (ny)
        * shape: array with min(x), max(x), min(y) and max(y)
        * z: float with the distance between the sample surface and the sensor
    """
    
    nx, ny = shape
    x1, x2, y1, y2 = area
    xs = np.linspace(x1, x2, nx)
    ys = np.linspace(y1, y2, ny)
    
    arrays = np.meshgrid(ys, xs)[::-1]
    
    if z is not None:
        arrays.append(z*np.ones(nx*ny, dtype=np.float64))
    return [i.ravel() for i in arrays]



def add_noise(data, error=0.05, method='max amplitude', seed=None):    
    """
    Generates a Gaussian noise (normal distribution, mean equals zero, and desv. pad. equals the
    error percent input)

    Parameters:
        * data: simulated 1D data vector.
        * error: float number with the error value (vary with the method)
        * method: string with the method to be used ('fixed' or 'max amplitude')
                    - if the chosen method is 'max amplitude' --> error percentage (default = 5%)
                        based on the maximum amplitude of the anomaly. 
                    - if the chosen method is 'fixed' --> the noise will be a zero mean gaussian 
                        distribution with desv. pad. equals to the input error.
    """    
    valid_methods = ["max amplitude", "fixed"]
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}.")
    if method == 'max amplitude':
        std = np.ptp(data) * error
    elif method == 'fixed':
        std = error    
    random = np.random.default_rng(seed=seed)
    noise = random.normal(0, std, data.shape)
    noisy_data = data + noise
 
    return noisy_data


def derivative_fd(data_2D, X, Y, order=1):

    """
    Generates both x and y derivatives using finite difference (fd) method.

    Parameters:
        * data_2D: 2D data matrix.
        * X: 2D X coordinates points of data matrix.
        * Y: 2D Y coordinates points of data matrix.
        * order: the order of the derivative, default is order=1 for the first derivatives.
    """

    x_derivative=np.zeros(np.shape(data_2D))
    y_derivative=np.zeros(np.shape(data_2D))
    

    x_derivative[1:-1,1:-1]= (data_2D[2:,1:-1]-data_2D[:-2,1:-1])/(X[2:,1:-1]-X[:-2,1:-1])
    y_derivative[1:-1,1:-1]= (data_2D[1:-1,2:]-data_2D[1:-1,:-2])/(Y[1:-1,2:]-Y[1:-1,:-2])
               
    if order>1:
        x_derivative, _  = derivative_fd(x_derivative, X, Y, order=order-1)
        _,  y_derivative = derivative_fd(y_derivative, X, Y, order=order-1)
            

    # boundary conditions
    x_derivative[ 0, :] = x_derivative[1 , :]
    x_derivative[-1, :] = x_derivative[-2, :]
    y_derivative[: , 0] = y_derivative[: , 1]
    y_derivative[: ,-1] = y_derivative[: ,-2]
    
    
    return(x_derivative, y_derivative)



def wave_numbers(data_2D, X, Y): 

    """
    Generates the x, y and z wave number for FFT filters (e.g, derivatives and upward cont.).

    Parameters:
        * data_2D: 2D data matrix.
        * X: 2D X coordinates matrix.
        * Y: 2D Y coordinates matrix.
    """

    wx = np.zeros(np.shape(data_2D))
    wy = np.zeros(np.shape(data_2D))
    
    nx, ny = np.shape(data_2D)
    y_step = (np.max(Y) - np.min(Y)) / (ny-1)
    x_step = (np.max(X) - np.min(X)) / (nx-1)
    
    # x frequency
    for j in range(np.shape(data_2D)[1]):
        kx = np.fft.fftfreq(nx, x_step)
        wx[:,j] = 2*np.pi*kx  # wave number in the x direction

    # y frequency
    for i in range(np.shape(data_2D)[0]):
        ky = np.fft.fftfreq(ny, y_step)
        wy[i,:] = 2*np.pi*ky   # wave number in the y direction
    
    # radial wave number
    wz = np.sqrt(wx**2+wy**2) 

    return(wx, wy, wz)



def y_derivative_fft(data_2D, wy, order=1):
    
    """
    Generates y derivatives using fast Fourier transform (fft) method.

    Parameters:
        * data_2D: 2D data matrix.
        * wy: 2D Y wavenumber matrix.
        * order: the order of the derivative, default is order=1 for the first x and y derivatives.
    """
    
    f_hat = np.fft.fft2(data_2D)
    derivative_factor = (1j*wy)**order
    
    y_derivative = np.real(np.fft.ifft2(derivative_factor*f_hat))
    
    # boundary conditions
    y_derivative[0 , :] = y_derivative[1 , :]
    y_derivative[-1, :] = y_derivative[-2, :]
    y_derivative[: , 0] = y_derivative[: , 1]
    y_derivative[: ,-1] = y_derivative[: ,-2]
    
    return(y_derivative)



def x_derivative_fft(data_2D, wx, order=1):

    """
    Generates x derivatives using fast Fourier transform (fft) method.

    Parameters:
        * data_2D: 2D data matrix.
        * wx: 2D X wavenumber matrix.
        * order: the order of the derivative, default is order=1 for the first x and y derivatives.
    """ 
    
    f_hat = np.fft.fft2(data_2D)
    derivative_factor = (1j*wx)**order
    
    x_derivative = np.real(np.fft.ifft2(derivative_factor*f_hat))
    
    # boundary conditions
    x_derivative[0 , :] = x_derivative[1 , :]
    x_derivative[-1, :] = x_derivative[-2, :]
    x_derivative[: , 0] = x_derivative[: , 1]
    x_derivative[: ,-1] = x_derivative[: ,-2]

    return(x_derivative)



def z_derivative_fft(data_2D, wz, order=1):

    """
    Generates z derivatives using fast Fourier transform (fft) method.

    Parameters:
        * data_2D: 2D data matrix.
        * wz: 2D Z wavenumber matrix.
        * order: the order of the derivative, default is order=1 for the first x and y derivatives.
    """ 
    
    f_hat = np.fft.fft2(data_2D)
    derivative_factor = wz**order
    
    z_derivative = np.real(np.fft.ifft2(derivative_factor*f_hat))
    
    # boundary conditions
    z_derivative[0 , :] = z_derivative[1 , :]
    z_derivative[-1, :] = z_derivative[-2, :]
    z_derivative[: , 0] = z_derivative[: , 1]
    z_derivative[: ,-1] = z_derivative[: ,-2]
    
    return(z_derivative)



def upward_continuation(data_2D, delta_z, wz):

    """
    Apply the fft based upward continuation filtering.

    Parameters:
        * data_2D: 2D data matrix.
        * wz: 2D Z wavenumber matrix.
        * delta_z: difference (in meters) between the data level and the new level (Z - Z0)
    """ 
    
    f_hat = np.fft.fft2(data_2D)                 
    up_cont_factor = np.exp((delta_z)*(wz))
    
    up_cont = np.real(np.fft.ifft2(f_hat*up_cont_factor))
    
    # boundary conditions
    up_cont[ 0, :] = up_cont[1 , :]
    up_cont[-1, :] = up_cont[-2, :]
    up_cont[: , 0] = up_cont[: , 1]
    up_cont[: ,-1] = up_cont[: ,-2]
    
    return(up_cont)



def z_derivative_fd(upward_1, upward_2, delta_Z1, delta_Z2):

    """
    Generates z derivative using finite difference (fd) method.

    Parameters:
        * upward_1: upwarded 2D data matrix.
        * upward_2: upwarded 2D data matrix.
        * delta_Z1: difference (in meters) between the data level and the new level (Z - Z0)
        * delta_Z2: difference (in meters) between the data level and the new level (Z - Z0)
    
    Warning: Keep in mind that this approximation calculates the z derivative
    for the z level = (delta_Z2 - delta_Z1)/2. So, If you want the derivative
    for the z = 5 m, you will need the upward for the levels 4 and 6 meters as
    input.

    """

    z_derivative = np.zeros(np.shape(upward_1))
    z_derivative=(upward_1-upward_2)/(np.absolute(delta_Z2 - delta_Z1))
    
    return(z_derivative)



def Horiz_Grad(x_derivative, y_derivative):

    """
    Calculates the horizontal gradients of the potential field.

    Parameters:
        * x_derivative: X derivative 2D data matrix.
        * y_derivative: Y derivative 2D data matrix.
    """ 
    
    horizontal_gradient = np.sqrt(np.absolute(y_derivative**2+x_derivative**2))
    
    return(horizontal_gradient)



def Total_Grad(x_derivative, y_derivative, z_derivative): 

    """
    Calculates the horizontal gradients of the potential field.

    Parameters:
        * x_derivative: X derivative 2D data matrix.
        * y_derivative: Y derivative 2D data matrix.
        * z_derivative: Z derivative 2D data matrix.
    """ 
    
    total_gradient = np.sqrt(np.absolute(y_derivative**2+x_derivative**2+z_derivative**2))
    
    return(total_gradient)


def sources_finder(Grad_Data, data = [], threshold=0.05, min_sigma=1, max_sigma=100, num_sigma=50, overlap=1.0, radius_increment=0.1):
    
    """
    Applies the modified scikit-image blob detection algotithm to find the window range
    of each source detected.
    The blobs will be assumed as bright areas surrounded by a darker neighborhood.

    original code: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.blob_log
    
    Parameters:
        * Grad_Data: 2D gradient data matrix.
        * data (optional): 2D data matrix to replace the background Grad_Data in the plot.
        * threshold: float number representing the lower boundary for scale maxima detection. 
                     The threshold must be reduced in order to detect low intensities blobs.
        * min_sigma: scalar number representing the minimum standard deviation for the applied
                     Gaussian Kernel.
        * max_sigma: scalar number representing the maxium standard deviation for the applied
                     Gaussian Kernel.                    
        * mum_sigma: scalar number representing the intermediate values between the minimum and
                     maxium standard deviation.
        * overlap: float number (between 0 and 1) representing the maximum value that two blobs
                     can overlap their areas, otherwise the smaller one is excluded.
        * radius_increment: float number that gives the option of expanding or reducing the size 
                     of the window range of each source (e.g, 0.1 gives a window 10% bigger).
    """ 

    input_data = (Grad_Data / Grad_Data.max())  # normalized data (0<= data <=1) turning into grayscale
    
    blobs_log = blob_log(input_data,  min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold = threshold, overlap=overlap)

    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)

    blobs_list = [blobs_log]
    colors = ['yellow']
    titles = ['Blob Detection (Laplacian of Gaussian)']
    sequence = zip(blobs_list, colors, titles)

    circles = []

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)
    for idx, (blobs, color, title) in enumerate(sequence):
        plt.title(title, fontsize=18)
        
        if np.size(data)==0: # plot contour of the gradient map
            plt.imshow(input_data)
        else:                # plot contour of the optional map
            plt.imshow(data)
        plt.gca().invert_yaxis()
        for blob in blobs:
            y, x, r = blob
            if (r>=2):       # excludes really smalls blobs caused by high frequency noise
                circles = np.append(circles, [np.round(y),np.round(x),np.round(r + radius_increment*r)])
                c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
                ax.add_patch(c)

    plt.tight_layout()
    plt.show()
    
    
    circles_T = np.reshape(circles, (int(np.size(circles)/3),3))

                
    ######################################################################
    # This section works better than the parameter overlap for blob clusters
    # remove larger circles wrappind smaller ones
    circles_teste = []
    for i in range(np.shape(circles_T)[0]):
        for j in range(np.shape(circles_T)[0]):
            center_distance = np.sqrt( (circles_T[i, 0]-circles_T[j, 0])**2 + (circles_T[i, 1]-circles_T[j, 1])**2 )
            radius_sum = np.absolute(circles_T[i, 2] + circles_T[j, 2])
            radius_dif = np.absolute(circles_T[i, 2] - circles_T[j, 2])
             
            if i != j:
                if (center_distance <= radius_dif) or (center_distance <= 0.5*radius_sum) or (int(center_distance)==0):
                    if circles_T[i, 2] < circles_T[j, 2]:
                        circles_teste.append(circles_T[j, 0])
                        circles_teste.append(circles_T[j, 1])
                        circles_teste.append(circles_T[j, 2])

                    

    circles_teste = np.reshape(circles_teste, (int(np.size(circles_teste)/3),3))

    for i in circles_teste:  
        bubble = []
        delete = (circles_T==i)
        for j in range(np.shape(delete)[0]):
            if delete[j, 0]==True & delete[j, 1]==True & delete[j, 2]==True:
                bubble.append(True)
            else:
                bubble.append(False)
        circles_T = np.delete(circles_T, bubble, 0)
    
    ########################################################################

    # Generates a 2D array with the vertices coordinates of each source window
    euler_windows = []

    for i in range(np.shape(circles_T)[0]):
            x1 = int(circles_T[i,0]-circles_T[i,2])
            x2 = int(circles_T[i,0]+circles_T[i,2])
            y1 = int(circles_T[i,1]-circles_T[i,2])
            y2 = int(circles_T[i,1]+circles_T[i,2])
            
            # Removes windows that extend beyond the edges of the data (causes Euler to fail)
            if (x1>=0) & (x2<=np.shape(input_data)[0]) & (y1>=0) & (y2<=np.shape(input_data)[1]):         
                euler_windows = np.append(euler_windows, ([x1, x2, y1, y2]) )
                
    euler_windows_T = np.reshape(euler_windows, (int(np.size(euler_windows)/4),4))

    return(circles, euler_windows_T)
    

    
def euler_windows_view(X, Y, Z, data_2D, euler_windows_T, show_windows=False, color='y', figsize=(10,10)):
    
    """
    This function shows the selected Euler windows (where the Euler equation has a
    positive z value). It is also possible to observe each window individually. 

    Parameters:
        * X: 2D X coordinates matrix.
        * Y: 2D Y coordinates matrix.
        * Z: 2D Z coordinates matrix.
        * data_2D: 2D data matrix.
        * euler_windows_T: matrix (L x 4) with the x_min, x_max, y_min and y_max of 
                           the j-th source window, j = 1, 2, ..., L.
        * show_windows: boolean, if True the function shows each window individually.
        * color: string that sets the edge color of the windows in the plot.
        * figsize: list that sets the size of the figure to be plotted.

    """ 

    nx, ny = np.shape(data_2D)
    y_step = (np.max(Y)*m2microm - np.min(Y)*m2microm) / (ny-1)
    x_step = (np.max(X)*m2microm - np.min(X)*m2microm) / (nx-1)
    
    plt.figure(figsize=figsize)
    plt.contourf(Y*m2microm, X*m2microm, data_2D, cmap='viridis')
    
    for i in range(np.shape(euler_windows_T)[0]):
        x1 = int(euler_windows_T[i,0]) * x_step
        x2 = int(euler_windows_T[i,1]) * x_step
        y1 = int(euler_windows_T[i,2]) * y_step
        y2 = int(euler_windows_T[i,3]) * y_step
        
        plt.hlines(x1, y1, y2, color=color)
        plt.hlines(x2, y1, y2, color=color)
        plt.vlines(y1, x1, x2, color=color)
        plt.vlines(y2, x1, x2, color=color)
    
    if show_windows:
    
        for i in range(np.shape(euler_windows_T)[0]):
            x1 = int(euler_windows_T[i,0])
            x2 = int(euler_windows_T[i,1])
            y1 = int(euler_windows_T[i,2])
            y2 = int(euler_windows_T[i,3])

            plt.figure(figsize=figsize)
            plt.contourf(Y[x1:x2, y1:y2]*m2microm, X[x1:x2, y1:y2]*m2microm, data_2D[x1:x2, y1:y2])
    
    return()
   

def solve_euler(X, Y, Z, data_2D, x_derivative, y_derivative, z_derivative, delta_z, structural_index):
    
    """
    Solves the Euler equation returning the center positioning of the source.

    Parameters:
        * X: 2D X coordinates matrix.
        * Y: 2D Y coordinates matrix.
        * Z: 2D Z coordinates matrix.
        * data_2D: 2D data matrix.
        * x_derivative: X derivative 2D data matrix.
        * y_derivative: Y derivative 2D data matrix.
        * z_derivative: Z derivative 2D data matrix.
        * delta_z: float number, if the input data_2D is upward continuation data
                   then the Z coordinates must be corrected adding delta_z.
        * structural_index: scalar number with the index of the source (equals to 3 for sphere) .
    """ 


    # first member --> components of G matrix
    x_deriv_1D = np.array(np.reshape(x_derivative, (np.size(x_derivative),1)))
    y_deriv_1D = np.array(np.reshape(y_derivative, (np.size(y_derivative),1)))
    z_deriv_1D = np.array(np.reshape(z_derivative, (np.size(z_derivative),1)))
    ni   = np.array(np.ones(np.shape(x_deriv_1D))*structural_index)

    G = np.zeros((np.size(x_deriv_1D), 4))
    G[:,0] = x_deriv_1D[:,0]
    G[:,1] = y_deriv_1D[:,0]
    G[:,2] = z_deriv_1D[:,0]
    G[:,3] =   ni[:,0]

    # second member --> data vector d
    X_1D  = np.array(np.reshape(X, (np.size(X),1)))
    Y_1D  = np.array(np.reshape(Y, (np.size(Y),1)))
    Z_1D  = np.array(np.reshape(Z, (np.size(Z),1))) + delta_z
    ni_d = structural_index*data_2D            # struct index times data
    ni_d_ = np.array(np.reshape(ni_d, (np.size(ni_d),1)))

    d = (X_1D[:,:]*x_deriv_1D[:,:]) + (Y_1D[:,:]*y_deriv_1D[:,:]) + ((Z_1D[:,:])*z_deriv_1D[:,:])  + (ni_d_)

    # solving linear system using least square to find the parameters vector p
    p = np.linalg.solve(G.T@G, G.T@d)
    
    return(p[0],p[1],p[2]) # Return source center positions(Xc, Yc, Zc)




def solve_euler_windows(euler_windows, X, Y, Z, data_2D, x_derivative, y_derivative, z_derivative, delta_z=0, structural_index = 3.0):
    
    """
    Solves the Euler equation for each data window source.

    Parameters:
        * X: 2D X coordinates matrix.
        * Y: 2D Y coordinates matrix.
        * Z: 2D Z coordinates matrix.
        * data_2D: 2D data matrix.
        * x_derivative: X derivative 2D data matrix.
        * y_derivative: Y derivative 2D data matrix.
        * z_derivative: Z derivative 2D data matrix.
        * delta_z: float number, if the input data_2D is upward continuation data
                   then the Z coordinates must be corrected adding delta_z.
        * structural_index: scalar number with the index of the source (equals to 3 for sphere) .
    """ 


    euler_position = []
    for i in range(np.shape(euler_windows)[0]):
        x1 = int(euler_windows[i, 0])
        x2 = int(euler_windows[i, 1])
        y1 = int(euler_windows[i, 2])
        y2 = int(euler_windows[i, 3])
        source_position = solve_euler( X[x1:x2, y1:y2], Y[x1:x2, y1:y2], Z[x1:x2, y1:y2], 
                                       data_2D[x1:x2, y1:y2], x_derivative[x1:x2, y1:y2], 
                                       y_derivative[x1:x2, y1:y2], z_derivative[x1:x2, y1:y2], 
                                       delta_z = delta_z, structural_index = structural_index)
        euler_position = np.append(euler_position, source_position)

    euler_position = np.reshape(euler_position, (int(np.size(euler_position)/3),3))
    
    Xc = []
    Yc = []
    Zc = []

    filtered_euler_windows = []
    for row in range (np.shape(euler_position)[0]):
        if euler_position[row, 2] >= 0: # filter only the Zc positive values (i.e., only the successful ones)
            Xc = np.append(Xc, euler_position[row, 0] )
            Yc = np.append(Yc, euler_position[row, 1] )
            Zc = np.append(Zc, euler_position[row, 2] )

            x1 = int(euler_windows[row, 0])
            x2 = int(euler_windows[row, 1])
            y1 = int(euler_windows[row, 2])
            y2 = int(euler_windows[row, 3])

            window = np.array([x1, x2, y1, y2])
            
            filtered_euler_windows = np.append(filtered_euler_windows, window)
    
    filtered_euler_windows = np.reshape(filtered_euler_windows, (int(np.size(filtered_euler_windows)/4),4))

    
    return(Xc, Yc, Zc, filtered_euler_windows)



def least_square_solver(X, Y, Z, Xc, Yc, Zc, Bz):

    """
    Calculates the magnetic moment vector (mx_j, my_j, mx_j) for each source (j) using
    the least square estimator. 
    It generates the sensibility matrix (A) using the center positioning (Xc, Yc, Zc) of
    each source.

    Parameters:
        *  X,  Y,  Z: 1D-arrays with x, y and z data positions
        * Xc, Yc, Zc: 1D-arrays with x, y and z data  of the sphere's center positions
        * Bz: 1D-arrays with observed potential field data

    """

    Bz = np.squeeze(np.reshape(Bz, (np.size(Bz),1))) # to avoid matrix dimension miss match

    A = sensibility_matrix(X, Y, Z, Xc, Yc, Zc)
    m = np.linalg.solve(A.T@A, A.T@Bz)
    
    
    w = int(np.size(m))
    m_T = np.reshape(m, (int(w/3),3) )
    
    mx = []
    my = []
    mz = []
    

    for row in range (np.shape(m_T)[0]):
        mx = np.append(mx, m_T[row, 0] )
        my = np.append(my, m_T[row, 1] )
        mz = np.append(mz, m_T[row, 2] )
        
    
    forward_model = (A@m)

    return(mx, my, mz, A, forward_model)



def directions(mx, my, mz, plot = False, show_mean = False, show_alpha95 = False):

    """
    Calculates the magnetization directions for all the sources.

    Parameters:
        * mx, my, mz: 1D-arrays with x, y and z magnetic moment of the sources
        * plot: boolean, if plot = True the function plots the directions in an
                equal area stereogram plot
        * show_mean: boolean, if plot = True the mean direction will also be plotted
        * show_alpha95: boolean, if plot = True the alpha95 cone will also be plotted
    """

    D = []
    I = []
    
    I_mean = ((math.atan2( np.mean(mz), (np.sqrt(np.mean(my)**2 + np.mean(mx)**2)) ) )* (180/np.pi))
    D_mean = ((math.atan2( np.mean(my),  np.mean(mx)) * (180/np.pi) ) )
    
    for i in range (np.size(mx)):

        I = np.append( I, (math.atan2( mz[i], (np.sqrt(my[i]**2+mx[i]**2)) ) )* (180/np.pi))
        D = np.append( D, (math.atan2( my[i],  mx[i] ) * (180/np.pi) ) )
            
        
    if plot == True:
        from apsg.feature import LineationSet
    
        fig = plt.figure(figsize=(5, 5))
        s = StereoNet(grid=True, legend=True)
        
        group = []
        
        for w in range (np.size(D)):
            
            if I[w] >= 0:
                symbol = '.'
                color = 'r'
            else:
                symbol = '.'
                color = 'k'
            
            if w == 0:
                s.line((lin(float(D[w]), np.round(np.absolute(I[w])))), color=color, marker=symbol)
                group.append((lin(float(D[w]), np.round(np.absolute(I[w])))))

            else:
                s.line((lin(float(D[w]), np.round(np.absolute(I[w])))), color=color, marker=symbol)
                group.append((lin(float(D[w]), np.round(np.absolute(I[w])))))
         
        group = LineationSet(group) # create a group variable with all lines
        
        if show_mean == True:
            s.line((lin(float(D_mean), np.round(np.absolute(I_mean)))), color='g', marker='o')
            print('Mean direction: '+str(D_mean)+' / '+str(I_mean))
        if show_alpha95 == True:
            s.cone(group.fisher_cone_a95(), color='b') # gives the a95 cone for the group


        s.render2fig(fig)                       
    
    return(D, I)

    
def uncertainties(sigma_zero, A, mx, my, mz):

    """
    Calculates the uncertainties of the magnetict directions and magnetic moment
    for all the sources.

    Parameters:
        * sigma_zero: standart deviation of the pontential field data acquisition
        * A: sensitive matrix (see "sensibility_matrix" function) 
        * mx, my, mz: 1D-arrays with x, y and z magnetic moment of the sources

    """

    Cov_matrix = (sigma_zero**2)*np.linalg.inv(A.T@A)
    diag_cov_matrix = np.diag(Cov_matrix)
    
    sigma_mx = []
    sigma_my = []
    sigma_mz = []
    
    i = -1
    
    while i <= (np.size(diag_cov_matrix)-2):
        i += 1
        sigma_mx = np.append(sigma_mx, diag_cov_matrix[i])
        
        i += 1
        sigma_my = np.append(sigma_my, diag_cov_matrix[i])
        
        i += 1
        sigma_mz = np.append(sigma_mz, diag_cov_matrix[i])
        
    # Mag Dec uncertainties 
    dD_dmx = -my/(mx**2+my**2)
    dD_dmy = mx/(mx**2+my**2)

    sigma_D = np.sqrt( ((dD_dmx)**2*(sigma_mx)) + ((dD_dmy)**2*(sigma_my)) ) * (180/np.pi) 

    # Mag Inc uncertainties    
    dI_dmx = (-mx*mz) / ( np.sqrt(mx**2+my**2) * (mx**2 + my**2 + mz**2)  )
    dI_dmy = (-my*mz) / ( np.sqrt(mx**2+my**2) * (mx**2 + my**2 + mz**2)  )
    dI_dmz = (np.sqrt(mx**2+my**2) ) / ((mx**2 + my**2 + mz**2))

    sigma_I = np.sqrt( (dI_dmx)**2*(sigma_mx) + (dI_dmy)**2*(sigma_my) + (dI_dmz)**2*(sigma_mz) ) * (180/np.pi) 
    
    # Mag Moment uncertainties 
    dm_dmx = mx/np.sqrt(mx**2+my**2+mz**2)
    dm_dmy = my/np.sqrt(mx**2+my**2+mz**2)
    dm_dmz = mz/np.sqrt(mx**2+my**2+mz**2)
    
    sigma_m = np.sqrt( (dm_dmx)**2*(sigma_mx) + (dm_dmy)**2*(sigma_my) + (dm_dmz)**2*(sigma_mz) )
    
    return(sigma_D, sigma_I, sigma_m)
    

    
def robust_solver(X, Y, Z, Xc, Yc, Zc, Bz, tolerance=1.0E-5):

    """
    Calculates the magnetic moment vector (mx_j, my_j, mx_j) for each source (j) using
    the robust estimator, which is less sensible to outliers in the data. 
    It generates the sensibility matrix (A) using the center positioning (Xc, Yc, Zc) of
    each source.

    Parameters:
        *  X,  Y,  Z: 1D-arrays with x, y and z data positions
        * Xc, Yc, Zc: 1D-arrays with x, y and z data  of the sphere's center positions
        * Bz: 1D-arrays with observed potential field data
        * tolerance (optional): float number used to set a break condition for the loop.
    """

    #rearrange Bz data into a column array and squeeze to avoid errors with scipy.sparse.diags()
    Bz = np.squeeze(np.reshape(Bz, (np.size(Bz),1)))

    background = Bz.mean()
    Bz = Bz - background # remove signal caused by deeper sources    


    A = sensibility_matrix(X, Y, Z, Xc, Yc, Zc)
    m = np.linalg.solve(A.T@A, A.T@Bz) # least square (Rk, k=0)
    e = 1.0E-20 # small positive value to avoid singularities
    
       
    # setting the diagonal matrix N x N
    r_k = 1/(np.absolute(np.matmul(A,m) - Bz + e))
    R_k = sp.sparse.diags(r_k , offsets=0, format='csc')


 
    break_condition = np.inf
    for i in range(1,11):

        print('iteration: ', i)
        
        bubble_m = np.copy(m) # for the breaking condition
        
        ### first iteration
        A_T = A.T
        A_T_R = (A_T@R_k)
        

        A_A_T_R = sp.sparse.csc_matrix(np.matmul(A_T_R, A)) # convert to csc to avoid format error
        d_A_T_R = np.matmul(A_T_R, Bz)


        m = sp.sparse.linalg.spsolve(A_A_T_R, d_A_T_R)
        
        norm_m_plus_one = np.linalg.norm(m) # for the breaking condition
        
       
        if break_condition<=tolerance:
            break
        

        #### recalculating R_k for the next iteration
        r_k = 1/(np.absolute(np.matmul(A,m) - Bz + e))
        R_k = sp.sparse.diags(r_k , offsets=0, format='csc')
        
        break_condition = np.linalg.norm(m-bubble_m)/(1+(norm_m_plus_one)) # following Aster et al. (2005)
    
    w = int(np.size(m))
    m_T = np.reshape(m, (int(w/3),3) )
    
    mx = []
    my = []
    mz = []
    
    for row in range (np.shape(m_T)[0]):
        mx = np.append(mx, m_T[row, 0] )
        my = np.append(my, m_T[row, 1] )
        mz = np.append(mz, m_T[row, 2] )
        
    forward_model = (A@m)+background

    return(mx, my, mz, A, forward_model)