"""
Created on Thu May 21 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for geometry processing

"""

from scipy.interpolate import UnivariateSpline

import numpy as np


# --- Curve Utils ---
def fit_spline_1d(pts, k=3, s=None):
    """
    Fits a spline to 1D curve.

    Parameters
    ----------
    pts : numpy.ndarray
        Points to be smoothed.
    k : int, optional
        Degree of the spline. Default is 3.
    s : float, optional
        Parameter that controls the smoothness of the spline. Default is None.

    Returns
    -------
    UnivariateSpline
        Spline fit to the given points.
    """
    t = np.linspace(0, 1, len(pts))
    s = len(pts) / s if s else len(pts) / 15
    return UnivariateSpline(t, pts, k=k, s=s)


def fit_spline_3d(pts, k=3, s=None):
    """
    Fits a cubic spline to an array containing xyz coordinates.

    Parameters
    ----------
    pts : numpy.ndarray
        Array of xyz coordinates to be smoothed.
    k : int, optional
        Degree of the spline. Default is 3.
    s : float, optional
        Parameter that controls the smoothness of the spline. Default is None.

    Returns
    -------
    spline_x : UnivariateSpline
        Spline fit to x-coordinates of the given points.
    spline_y : UnivariateSpline
        Spline fit to the y-coordinates of the given points.
    spline_z : UnivariateSpline
        Spline fit to the z-coordinates of the given points.
    """
    spline_x = fit_spline_1d(pts[:, 0], k=k, s=s)
    spline_y = fit_spline_1d(pts[:, 1], k=k, s=s)
    spline_z = fit_spline_1d(pts[:, 2], k=k, s=s)
    return spline_x, spline_y, spline_z


def resample_curve_1d(pts, n_pts=None, s=None):
    """
    Smooths a 1D curve by fitting a spline and resampling it.

    Parameters
    ----------
    n_pts : int or None, optional
        Number of points to resample.
    s : float, optional
        Parameter that controls the smoothness of the spline. Default is None.

    Returns
    -------
    numpy.ndarray
        Resampled points.
    """
    # Fit spline
    dt = max(n_pts or len(pts), 5)
    k = min(3, len(pts) - 1)

    # Check for degenerate case
    if k == 0:
        return np.repeat(pts, n_pts, axis=0)

    # Resample points
    t = np.linspace(0, 1, dt)
    spline = fit_spline_1d(pts, k=k, s=s)
    return spline(t)


def resample_curve_3d(pts, n_pts=None, s=None):
    """
    Smooths an Nx3 array of points by fitting a spline. Points are assumed
    to form a continuous curve that does not have any branching points.

    Parameters
    ----------
    pts: numpy.ndarray
        Array of points to be smoothed.
    n_pts : int
        Number of points to be sampled from the spline. Default is None.
    s : float
        Parameter that controls the smoothness of the spline. Default is None.

    Returns
    -------
    pts : numpy.ndarray
        Resampled points.
    """
    # Compute spline parameters
    dt = max(n_pts or len(pts), 5)
    k = min(3, len(pts) - 1)

    # Check for degenerate case
    if k == 0:
        return np.repeat(pts, n_pts, axis=0)

    # Fit spline
    spline_x, spline_y, spline_z = fit_spline_3d(pts, k=k, s=s)

    # Resample points
    t = np.linspace(0, 1, dt)
    pts = np.column_stack(
        (
            spline_x(t).astype(np.float32),
            spline_y(t).astype(np.float32),
            spline_z(t).astype(np.float32),
        )
    )
    return pts


# --- Miscellaneous ---
def make_digital_line(p1, p2):
    """
    Generates integer voxel coordinates along a 3D line between p1 and p2.

    Parameters
    ----------
    p1 : Tuple[int]
        Start coordinate of line.
    p2 : Tuple[int]
        End coordinate of line.

    Returns
    -------
    line : numpy.ndarray
        Voxel coordinates representing the straight line between p1 and p2.
    """
    # Convert coordinates to arrays
    p1 = np.array(p1, dtype=int)
    p2 = np.array(p2, dtype=int)

    # Determine number of points
    diff = p2 - p1
    n = np.max(np.abs(diff))
    if n == 0:
        return p1[None, :]

    # Generate line
    t = np.linspace(0, 1, n + 1)
    line = np.round(p1 + np.outer(t, diff)).astype(int)
    return line


def make_line(p1, p2, n_steps):
    """
    Generates a series of points representing a straight line between two 3D
    coordinates.

    Parameters
    ----------
    p1 : Tuple[float]
        Start coordinate of line.
    p2 : Tuple[float]
        End coordinate of line.
    n_steps : int
        Number of steps to interpolate between the two coordinates.

    Returns
    -------
    numpy.ndarray
        Coordinates representing the straight line between p1 and p2.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    t_steps = np.linspace(0, 1, n_steps)
    return np.array([(1 - t) * p1 + t * p2 for t in t_steps], dtype=int)


def make_voxels_connected(voxels):
    """
    Makes a list of voxels that form a discrete curve 27-connected.

    Parameters
    ----------
    voxels : List[Tuple[int]]
        List of voxel coordinates that form a discrete path.

    Returns
    -------
    voxels_out : numpy.ndarray
        List of voxels that is 27-connected.
    """
    voxels = np.asarray(voxels, dtype=int)
    voxels_out = []
    for a, b in zip(voxels[:-1], voxels[1:]):
        line = make_digital_line(a, b)
        if voxels_out:
            line = line[1:]
        voxels_out.extend(line)
    return np.array(voxels_out, dtype=int)


def midpoint(xyz_1, xyz_2):
    """
    Computes the midpoint between "xyz_1" and "xyz_2".

    Parameters
    ----------
    xyz_1 : numpy.ndarray
        n-dimensional coordinate.
    xyz_2 : numpy.ndarray
        n-dimensional coordinate.

    Returns
    -------
    numpy.ndarray
        Midpoint of "xyz_1" and "xyz_2".
    """
    return np.mean([xyz_1, xyz_2], axis=0)
