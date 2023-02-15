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
import choclo


TESLA_TO_NANOTESLA = 1e9
MICROMETER_TO_METER = 1e-6


def cartesian_vector(inclination, declination, amplitude):
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
    x, y, z : floats or arrays
        The calculated x, y, z vector components.
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
    return x, y, z


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


def gaussian_noise(error, shape, seed=None):
    """
    Generate a pseudo-random noise array of the given shape.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0, scale=error, size=shape)
    return noise


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
