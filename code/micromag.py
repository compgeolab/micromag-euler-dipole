# Copyright (c) 2023 Gelson Ferreira de Souza-Junior, Leonardo Uieda.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
"""
Functions for performing the processing and inversion of the microscopy data.
"""
import numpy as np
import xarray as xr
import skimage.feature
import numba
import harmonica as hm
import verde as vd
import choclo


TESLA_TO_NANOTESLA = 1e9
MICROMETER_TO_METER = 1e-6


def gaussian_noise(error, shape, seed=None):
    """
    Generate a pseudo-random noise array of the given shape.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0, scale=error, size=shape)
    return noise


def angles_to_vector(inclination, declination, amplitude):
    """
    Generate a 3-component vector from inclination, declination, and amplitude

    Inclination is positive downwards and declination is the angle with the y
    component. The vector has x, y, and z (upward) Cartesian components.

    Parameters
    ----------
    inclination : float or array
        The inclination values in degrees.
    declination : float or array
        The declination values in degrees.
    amplitude : float or array
        The vector amplitude values.

    Returns
    -------
    vector : 1D or 2D array
        The calculated x, y, z vector components. 1D if it's a single vector.
        If N vectors are calculated, the "vector" will have shape (N, 3) with
        each vector in a row of the array.
    """
    inclination = np.radians(inclination)
    declination = np.radians(declination)
    amplitude = np.asarray(amplitude)
    sin_inc = np.sin(-inclination)
    cos_inc = np.cos(-inclination)
    sin_dec = np.sin(declination)
    cos_dec = np.cos(declination)
    x = cos_inc * sin_dec * amplitude
    y = cos_inc * cos_dec * amplitude
    z = sin_inc * amplitude
    return np.transpose([x, y, z])


def vector_to_angles(vector):
    """
    Generate inclination, declination, and amplitude from a 3-component vector

    Inclination is positive downwards and declination is the angle with the y
    component. The vector has x, y, and z (upward) Cartesian components.

    Parameters
    ----------
    vector : 1D or 2D array
        The x, y, z vector components. Can be a 1D array for a single vector
        or 2D for multiple. If 2D, then each vector should be a row of the
        array.

    Returns
    -------
    inclination : float or array
        The inclination values in degrees.
    declination : float or array
        The declination values in degrees.
    amplitude : float or array
        The vector amplitude values.
    """
    vector = np.asarray(vector)
    # if vector.ndim == 1:
        # vector = np.array([vector])
    x, y, z = vector.T
    amplitude = np.sqrt(x**2 + y**2 + z**2)
    inclination = -np.degrees(np.arctan2(z, np.hypot(x, y)))
    declination = np.degrees(np.arctan2(x, y))
    return inclination, declination, amplitude


def dipole_bz(coordinates, dipole_coordinates, dipole_moments):
    """
    Calculate the z component (upward) of the magnetic field of dipoles

    Units:

    * Coordinates: µm
    * Dipole moment: A.m²
    * Magnetic field: nT

    Parameters
    ----------
    coordinates : tuple of arrays
        The coordinates of the computation/data points. Should be a tuple with
        arrays representing the x, y, and z coordinates of each point.
    dipole_coordinates : tuple of arrays
        The coordinates of the dipoles. Should be a tuple with arrays
        representing the x, y, and z coordinates of each dipole.
    dipole_moments : list or 2D-array
        A list of 3-component 1D arrays representing the dipole moment of each
        dipole. Can also be a 2D array with M rows (M = number of dipoles) and
        3 columns.

    Returns
    -------
    bz : array
        The calculated bz (positive upward) in nT. The array will have the
        same shape as the input coordinate arrays.
    """
    # Save the data coordinate shape so we can reshape the bz afterwards
    data_shape = coordinates[0].shape
    # Make sure coordinates are 1D arrays and the dipole moment is 2D also
    # convert all units to SI
    coordinates = [np.asarray(c).ravel() * MICROMETER_TO_METER for c in coordinates]
    dipole_coordinates = [
        np.asarray(c).ravel() * MICROMETER_TO_METER for c in dipole_coordinates
    ]
    dipole_moments = np.asarray(dipole_moments)
    if dipole_moments.ndim == 1:
        dipole_moments = np.array([dipole_moments])
    # Initialize and allocate the output array for bz
    bz = np.zeros(coordinates[0].shape)
    _dipole_bu_fast(
        coordinates[0],
        coordinates[1],
        coordinates[2],
        dipole_coordinates[0],
        dipole_coordinates[1],
        dipole_coordinates[2],
        dipole_moments,
        bz,
    )
    bz *= TESLA_TO_NANOTESLA
    return bz.reshape(data_shape)


@numba.jit(nopython=True, parallel=True)
def _dipole_bu_fast(e, n, u, de, dn, du, dipole_moments, bu):
    """
    This is the bit that runs the fast for-loops
    """
    for i in numba.prange(e.size):
        for j in range(de.size):
            result = choclo.dipole.magnetic_u(
                easting_p=e[i],
                northing_p=n[i],
                upward_p=u[i],
                easting_q=de[j],
                northing_q=dn[j],
                upward_q=du[j],
                magnetic_moment=dipole_moments[j, :],
            )
            bu[i] += result


def dipole_moment_inversion(data, dipole_coordinates):
    """
    Estimate the dipole moment through linear inversion.

    Parameters
    ----------
    data : xarray.DataArray
        The observed vertical magnetic field grid. Must have x, y, and z as
        coordinates.
    dipole_coordinates : 1D array
        The (x, y, z) coordinates of the dipole.

    Returns
    -------
    dipole_moment : 1D array
        The estimated dipole moment vector.
    covariance : 2D array
        The estimated covariance matrix of the dipole moment vector.
    r2 : float
        The R² coefficient of determination of the inversion.
    """
    # Make it a dataset so we can be sure the variable name is "bz"
    table = vd.grid_to_table(data.to_dataset(name="bz"))
    # Verde drops non-dimension coordinates so we have to add z back.
    # This is a bug in Verde.
    table["z"] = data.z.values.ravel()

    n_data = table.shape[0]
    n_params = 3
    A = np.empty((n_data, n_params))
    d = table.bz.values / TESLA_TO_NANOTESLA
    # Fill the Jacobian using a fast calculation with numba
    _dipole_jacobian_fast(
        table.x.values * MICROMETER_TO_METER,
        table.y.values * MICROMETER_TO_METER,
        table.z.values * MICROMETER_TO_METER,
        dipole_coordinates[0] * MICROMETER_TO_METER,
        dipole_coordinates[1] * MICROMETER_TO_METER,
        dipole_coordinates[2] * MICROMETER_TO_METER,
        A,
    )
    hessian = A.T @ A
    neg_gradient = A.T @ d
    dipole_moment = np.linalg.solve(hessian, neg_gradient)

    residuals = d - A @ dipole_moment
    residuals_sum_sq = np.sum(residuals**2)
    # Estimate of the true error variance (since we'll never know it)
    chi_squared =  residuals_sum_sq / (n_data - n_params)
    covariance = chi_squared * np.linalg.inv(hessian)
    r2 = 1 - residuals_sum_sq / np.linalg.norm(d - d.mean())**2

    return dipole_moment, covariance, r2


@numba.jit(nopython=True, parallel=True)
def _dipole_jacobian_fast(e, n, u, de, dn, du, jacobian):
    """
    This is the bit that runs the fast for-loops
    """
    constant = choclo.constants.VACUUM_MAGNETIC_PERMEABILITY / (4 * np.pi)
    for i in numba.prange(e.size):
        # Calculating the distance only once saves a lot of computation time
        distance = choclo.utils.distance_cartesian(
            e[i], n[i], u[i], de, dn, du,
        )
        # These are the second derivatives of 1/r
        jacobian[i, 0] = constant * choclo.point.kernel_eu(
            easting_p=e[i],
            northing_p=n[i],
            upward_p=u[i],
            easting_q=de,
            northing_q=dn,
            upward_q=du,
            distance=distance,
        )
        jacobian[i, 1] = constant * choclo.point.kernel_nu(
            easting_p=e[i],
            northing_p=n[i],
            upward_p=u[i],
            easting_q=de,
            northing_q=dn,
            upward_q=du,
            distance=distance,
        )
        jacobian[i, 2] = constant * choclo.point.kernel_uu(
            easting_p=e[i],
            northing_p=n[i],
            upward_p=u[i],
            easting_q=de,
            northing_q=dn,
            upward_q=du,
            distance=distance,
        )


def covariance_to_angle_std(dipole_moment, covariance):
    """
    Calculate the inc, dec, amp standard deviations from the covariance matrix
    """
    mx2, my2, mz2 = dipole_moment**2
    m = np.linalg.norm(dipole_moment)
    var_x, var_y, var_z = np.diag(covariance)
    sigma_inc = np.degrees(
        np.sqrt(
            (mx2 * mz2 * var_x + my2 * mz2 * var_y + (mx2 + my2)**2 * var_z)
            / ((mx2 + my2) * m**4)
        )
    )
    sigma_dec = np.degrees(np.sqrt((my2 * var_x + mx2 * var_y) / (mx2 + my2)**2))
    sigma_amp = np.sqrt((mx2 * var_x + my2 * var_y + mz2 * var_z) / m**2)
    return sigma_inc, sigma_dec, sigma_amp


def data_gradients(data):
    """
    Add the derivatives and total gradient amplitude
    """
    dx = data.differentiate("x")
    dy = data.differentiate("y")
    spacing = np.mean([np.abs(data.x[1] - data.x[0]), np.abs(data.y[1] - data.y[0])])
    # Need to set the exact same coordinates because the xrft inverse transform
    # creates slightly different ones because of round-off errors.
    data_up = hm.upward_continuation(data, spacing).assign_coords(dict(x=data.x, y=data.y))
    data_down = hm.upward_continuation(data, -spacing).assign_coords(dict(x=data.x, y=data.y))
    # Forward difference only to avoid downward continuation.
    dz = (data_up - data_down) / (2 * spacing)
    tga = np.sqrt(dx**2 + dy**2 + dz**2)
    tga.attrs = {"long_name": "total gradient amplitude", "units": "nT/µm"}
    dx.attrs = {"long_name": "x-derivative", "units": "nT/µm"}
    dy.attrs = {"long_name": "y-derivative", "units": "nT/µm"}
    dz.attrs = {"long_name": "z-derivative", "units": "nT/µm"}
    return xr.Dataset({"tga": tga, "x_deriv": dx, "y_deriv": dy, "z_deriv": dz})


def detect_anomalies(data, size_range, size_increment=2, nsizes=10, threshold=0.5, overlap=0.5):
    """
    Run the blob detection and produce bounding boxes in data coordinates
    """
    min_sigma, max_sigma = [0.5 * i for i in size_range]
    spacing = np.mean([np.abs(data.x[1] - data.x[0]), np.abs(data.y[1] - data.y[0])])
    iy, ix, sigma_pix = skimage.feature.blob_log(
        data,
        min_sigma=min_sigma / spacing,
        max_sigma=max_sigma / spacing,
        threshold=threshold,
        num_sigma=nsizes,
        overlap=overlap,
    ).T
    blob_coords = (data.x.values[ix.astype("int")], data.y.values[iy.astype("int")])
    blob_sizes = sigma_pix * np.sqrt(2) * spacing * size_increment
    windows = [
        [x - size, x + size, y - size, y + size]
        for size, x, y in zip(blob_sizes, *blob_coords)
    ]
    return windows


def euler_deconvolution(data, x_deriv, y_deriv, z_deriv):
    """
    Estimate the (x, y, z) position and base level by Euler Deconvolution
    """
    si = 3
    grids = xr.Dataset(
        dict(field=data, x_deriv=x_deriv, y_deriv=y_deriv, z_deriv=z_deriv)
    )
    table = vd.grid_to_table(grids)
    # Verde drops non-dimension coordinates so we have to add z back.
    # This is a bug in Verde.
    table["z"] = grids.z.values.ravel()
    n_data = table.shape[0]
    G = np.empty((n_data, 4))
    G[:, 0] = table.x_deriv
    G[:, 1] = table.y_deriv
    G[:, 2] = table.z_deriv
    G[:, 3] = si
    h = (
        table.x * table.x_deriv
        + table.y * table.y_deriv
        + table.z * table.z_deriv
        + si * table.field
    )
    p = np.linalg.solve(G.T @ G, G.T @ h)
    return p[:3], p[3]
