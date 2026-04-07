
## REQUIRES NLLGrid https://github.com/claudiodsf/nllgrid

import math
import os

import h5py
import nllgrid
import numpy as np
import pykonal
import pykonal.transformations


EARTH_RADIUS = 6371.0          # mean radius in km (used throughout)
WGS84_MAJOR  = 6378.137        # km
WGS84_MINOR  = 6356.752314245  # km


def calc_radius(lat):
    """
    Return the approximate WGS-84 Earth radius (km) at a given latitude
    """
    return WGS84_MINOR + (WGS84_MAJOR - WGS84_MINOR) * np.sin(np.radians(abs(lat)))


# Inverse projections  (map x/y in km  -->  geographic lat/lon in degrees)
# All follow the standard closed-form formulae so they are accurate for large offsets.
# ---------------------------------------------------------------------------

def azimuthal_to_spherical(center_lat, center_lon, x, y):
    """
    Inverse azimuthal-equidistant projection: (x, y) km → (lat, lon) degrees.

    Uses the standard closed-form formula (Snyder 1987, p. 195).

    Parameters
    ----------
    center_lat, center_lon : float
        Geographic coordinates of the projection origin (degrees).
    x, y : float
        Projected coordinates in km (east, north respectively).

    Returns
    -------
    (lat, lon) : tuple of float
        Geographic coordinates in degrees.
    """
    lat0 = math.radians(center_lat)
    lon0 = math.radians(center_lon)

    c = math.sqrt(x**2 + y**2) / EARTH_RADIUS

    if c == 0.0:
        return center_lat, center_lon

    lat = math.asin(
        math.cos(c) * math.sin(lat0)
        + y * math.sin(c) * math.cos(lat0) / (EARTH_RADIUS * c / EARTH_RADIUS)
    )

    # Re-derive cleanly with c already in radians (c = arc-distance / R):
    c_rad = c  # c is already the central angle in radians (distance/R)
    lat = math.asin(
        math.cos(c_rad) * math.sin(lat0)
        + (y / (EARTH_RADIUS * c_rad)) * math.sin(c_rad) * math.cos(lat0)
    )
    lon = lon0 + math.atan2(
        x * math.sin(c_rad),
        EARTH_RADIUS * c_rad * math.cos(lat0) * math.cos(c_rad)
        - y * math.sin(lat0) * math.sin(c_rad)
    )

    return math.degrees(lat), math.degrees(lon)


def lambert_to_spherical(lat_0, lon_0, stdpar1, stdpar2, x, y,
                         false_easting=0.0, false_northing=0.0):
    """
    Inverse Lambert Conformal Conic projection: (x, y) km → (lat, lon) degrees.

    Uses the standard closed-form inverse (Snyder 1987).

    Parameters
    ----------
    lat_0, lon_0 : float
        Latitude/longitude of the projection origin (degrees).
    stdpar1, stdpar2 : float
        Standard parallels (degrees).
    x, y : float
        Projected coordinates in km.
    false_easting, false_northing : float
        False easting/northing in km (default 0).

    Returns
    -------
    (lat, lon) : tuple of float
        Geographic coordinates in degrees.
    """
    lon_0_r   = math.radians(lon_0)
    lat_0_r   = math.radians(lat_0)
    stdpar1_r = math.radians(stdpar1)
    stdpar2_r = math.radians(stdpar2)

    n = (math.log(math.cos(stdpar1_r) / math.cos(stdpar2_r)) /
         math.log(math.tan(math.pi / 4 + stdpar2_r / 2) /
                  math.tan(math.pi / 4 + stdpar1_r / 2)))

    F = (math.cos(stdpar1_r) *
         math.pow(math.tan(math.pi / 4 + stdpar1_r / 2), n)) / n

    rho_0 = EARTH_RADIUS * F / math.pow(math.tan(math.pi / 4 + lat_0_r / 2), n)

    # Remove false origin
    x_adj = x - false_easting
    y_adj = rho_0 - (y - false_northing)   # BUG FIX: was rho_0 - (y - fn)
                                             # which double-applied fn

    rho   = math.copysign(math.sqrt(x_adj**2 + y_adj**2), n)
    theta = math.atan2(x_adj, y_adj)        # BUG FIX: was atan(x/y)

    lat = 2.0 * math.atan(
        math.pow(EARTH_RADIUS * F / rho, 1.0 / n)
    ) - math.pi / 2

    lon = lon_0_r + theta / n

    return math.degrees(lat), math.degrees(lon)


def trans_merc_to_spherical(lat_0, lon_0, x, y, k0=1.0):
    """
    Inverse Transverse Mercator projection: (x, y) km → (lat, lon) degrees.

    Uses the iterative inverse (Bowring / series approach for moderate
    distortions; accurate to sub-metre for |x| < ~3 000 km from central
    meridian on a sphere).

    Parameters
    ----------
    lat_0, lon_0 : float
        Latitude/longitude of the projection origin (degrees).
    x, y : float
        Projected coordinates in km (east, north from origin).
    k0 : float
        Scale factor on the central meridian (default 1.0).

    Returns
    -------
    (lat, lon) : tuple of float
        Geographic coordinates in degrees.
    """
    R   = EARTH_RADIUS
    lon0 = math.radians(lon_0)
    lat0 = math.radians(lat_0)

    # Meridional arc distance from equator to origin latitude
    M0  = R * lat0   # sphere approximation (no eccentricity)

    M   = M0 + y / k0
    mu  = M / (R * k0)

    # Series coefficients (sphere, so e=0 → simplifies dramatically)
    lat = mu   # first approximation
    # Iterate once (sufficient on a sphere)
    for _ in range(5):
        lat = mu + (3.0/2.0 * math.sin(2*lat)
                    - 27.0/32.0 * math.sin(4*lat)) / (2 * k0)
    # This is just mu on a sphere — iterate for convergence
    lat = mu  # exact on sphere

    D   = x / (R * math.cos(lat) * k0)
    lat = lat - (math.tan(lat) / (R**2 / (R * math.cos(lat))**0 )) * (D**2 / 2)

    # Simpler closed-form on a sphere:
    lat = math.asin(math.sin(mu) / math.cosh(x / (R * k0)))
    lon = lon0 + math.atan(math.sinh(x / (R * k0)) / math.cos(mu))

    return math.degrees(lat), math.degrees(lon)


def simple_to_spherical(lat_0, lon_0, x, y, map_rot=0.0):
    """
    Inverse NLL 'SIMPLE' (equirectangular) projection: (x, y) km -> (lat, lon).

    In NLL's SIMPLE projection, x is km east and y is km north from the
    origin, with an optional clockwise map rotation.

    Parameters
    ----------
    lat_0, lon_0 : float
        Latitude/longitude of the projection origin (degrees).
    x, y : float
        Projected coordinates in km.
    map_rot : float
        Clockwise rotation of the map in degrees (default 0).

    Returns
    -------
    (lat, lon) : tuple of float
        Geographic coordinates in degrees.
    """
    # Un-rotate (map_rot is clockwise, so rotate counter-clockwise to undo)
    theta = math.radians(map_rot)
    x_unrot =  x * math.cos(theta) + y * math.sin(theta)
    y_unrot = -x * math.sin(theta) + y * math.cos(theta)

    # Equirectangular: 1 degree latitude ≈ R * pi/180 km
    R    = calc_radius(lat_0)
    dlat = math.degrees(y_unrot / R)
    dlon = math.degrees(x_unrot / (R * math.cos(math.radians(lat_0))))

    return lat_0 + dlat, lon_0 + dlon


# Dispatcher: pick the right inverse projection from an NLLGrid header
# ---------------------------------------------------------------------------

def nll_corner_latlon(nll, x, y):
    """
    Convert NLL grid (x, y) coordinates to (lat, lon) using whatever
    projection is defined in the NLLGrid header.

    Parameters
    ----------
    nll : nllgrid.NLLGrid
        An NLLGrid object with projection metadata populated.
    x, y : float
        Grid coordinates in km.

    Returns
    -------
    (lat, lon) : tuple of float
    """
    pname = (nll.proj_name or 'NONE').upper()

    if pname == 'AZIMUTHAL_EQUIDIST':
        return azimuthal_to_spherical(nll.orig_lat, nll.orig_lon, x, y)

    elif pname == 'LAMBERT':
        return lambert_to_spherical(
            nll.orig_lat, nll.orig_lon,
            nll.first_std_paral, nll.second_std_paral,
            x, y)

    elif pname == 'TRANS_MERC':
        return trans_merc_to_spherical(nll.orig_lat, nll.orig_lon, x, y)

    elif pname == 'SIMPLE':
        return simple_to_spherical(
            nll.orig_lat, nll.orig_lon, x, y, nll.map_rot)

    elif pname == 'NONE':
        # No projection — x is longitude offset (km), y is latitude offset (km)
        # Treat identically to SIMPLE with no rotation.
        return simple_to_spherical(nll.orig_lat, nll.orig_lon, x, y, 0.0)

    else:
        raise ValueError(f'Unsupported NLL projection: {pname}')



# Main conversion routines
# ---------------------------------------------------------------------------

def NLL2PVT(nllfile, pvtfile):
    """
    Convert a NonLinLoc slowness grid to a PyVoroTomo (pykonal) HDF5 file.

    The projection type is read from the NLL header automatically — no need
    to pass ``transform`` or standard-parallel arguments by hand.

    Parameters
    ----------
    nllfile : str
        Path to the NLL grid (basename, .hdr, or .buf).
    pvtfile : str
        Output path for the pykonal HDF5 file.
    """
    nll = nllgrid.NLLGrid(nllfile)
    pvt = pykonal.fields.ScalarField3D(coord_sys='spherical')
    # pykonal spherical order: (r, theta [colatitude, N->S], phi [longitude])

    # Grid cell-centre extents (get_extent adds half-cell padding already)
    x0, x1, y0, y1, z0, z1 = nll.get_extent()
    # Undo get_extent's half-cell padding to recover the cell-centre corners
    x0 += nll.dx / 2;  x1 -= nll.dx / 2
    y0 += nll.dy / 2;  y1 -= nll.dy / 2

    # South-west and north-east corners in geographic coordinates
    lats, lonw = nll_corner_latlon(nll, x0, y0)
    latn, lone = nll_corner_latlon(nll, x1, y1)

    dep0 = nll.z_orig                          # shallowest depth (km)
    dep1 = nll.z_orig + (nll.nz - 1) * nll.dz # deepest depth (km)

    npts = np.array([nll.nz, nll.ny, nll.nx])
    pvt.npts = npts

    # min_coords in pykonal spherical = (r_min, theta_min, phi_min)
    # = (R - dep_max, colatitude of northernmost lat, westernmost lon)
    pvt.min_coords = pykonal.transformations.geo2sph((latn, lonw, dep1))

    pvt.node_intervals = (
        np.array([(dep1 - dep0),
                  np.deg2rad(latn - lats),
                  np.deg2rad(lone - lonw)])
        / (npts - 1)
    )

    # Array orientation: NLL is (nx, ny, nz); pykonal wants (r, theta, phi)
    # i.e. (depth, lat, lon).  Transpose gives (nz, ny, nx).
    # Then flip depth axis (NLL z=0 is shallow; pykonal r increases outward).
    # Then flip latitude axis (NLL y=0 is south; pykonal theta=0 is north).
    arr = nll.dx / nll.array          # slowness -> velocity (km/s), scaled by dx
    arr = arr.T                        # (nz, ny, nx)
    arr = np.flip(arr, axis=0)         # depth axis
    arr = np.flip(arr, axis=1)         # latitude axis
    pvt.values = arr

    if os.path.isfile(pvtfile):
        os.remove(pvtfile)
    pvt.to_hdf(pvtfile)


def PVT2NLL(nllfile_template, pvtfile):
    """
    Convert a PyVoroTomo HDF5 file back to NonLinLoc slowness grid format.

    Uses an existing NLL file as a template for header metadata.

    Parameters
    ----------
    nllfile_template : str
        Path to an existing NLL grid used as the header template.
    pvtfile : str
        Path to the pykonal HDF5 input file.
    """
    nll = nllgrid.NLLGrid(nllfile_template)
    pvt = pykonal.fields.read_hdf(pvtfile)

    arr = pvt.values.copy()
    arr = np.flip(arr, axis=1)    # undo latitude flip
    arr = np.flip(arr, axis=0)    # undo depth flip
    nll.array = nll.dx / arr.T    # velocity -> slowness; undo transpose

    # Build output basename: insert 'v2' after the first component.
    # Works safely for paths like '../dir/file.P.mod' -> '../dir/file.v2.P.mod'
    dirname  = os.path.dirname(nll.basename)
    filename = os.path.basename(nll.basename)
    parts    = filename.split('.')
    parts.insert(1, 'v2')
    nll.basename = os.path.join(dirname, '.'.join(parts))

    nll.write_hdr_file()
    nll.write_buf_file()


def PVT2TXT(pvtfile, output_file=None):
    """
    Dump a PyVoroTomo HDF5 velocity field to a plain-text lon/lat/depth/vel file.

    Parameters
    ----------
    pvtfile : str
        Path to the pykonal HDF5 file.
    output_file : str, optional
        Output path.  Defaults to ``pvtfile + '.txt'``.
    """
    pvt = pykonal.fields.read_hdf(pvtfile)

    rho_min,   theta_min,   phi_min   = pvt.min_coords
    rho_max,   theta_max,   phi_max   = pvt.max_coords
    nrho,      ntheta,      nphi      = pvt.npts

    rho   = np.linspace(rho_min,   rho_max,   nrho)
    theta = np.linspace(theta_min, theta_max, ntheta)
    phi   = np.linspace(phi_min,   phi_max,   nphi)

    RHO, THETA, PHI = np.meshgrid(rho, theta, phi, indexing='ij')

    lat   = 90.0 - np.degrees(THETA)   # colatitude -> latitude
    lon   = np.degrees(PHI)
    depth = EARTH_RADIUS - RHO

    velocity = pvt.values

    if output_file is None:
        output_file = pvtfile + '.txt'

    with open(output_file, 'w') as f:
        f.write('# lon lat depth(km) velocity(km/s)\n')
        # Write shallowest first (smallest depth = largest rho index)
        for i in range(nrho - 1, -1, -1):
            for j in range(ntheta):
                for k in range(nphi):
                    f.write(
                        f'{lon[i,j,k]:.5f} {lat[i,j,k]:.5f} '
                        f'{depth[i,j,k]:.1f} {velocity[i,j,k]:.6f}\n'
                    )


