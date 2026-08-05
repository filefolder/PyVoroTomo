"""
Conversions between NonLinLoc (NLL) 3-D model grids and pykonal
(PyVoroTomo, "PVT") spherical velocity fields.

Requires
--------
nllgrid  https://github.com/claudiodsf/nllgrid
pykonal  https://github.com/filefolder/pykonal_0.5
numpy, scipy, pyproj (pyproj comes in via nllgrid)

Design notes
------------
*   All map projection work is delegated to ``nllgrid.NLLGrid.project`` /
    ``.iproject``.  Those use pyproj with the ellipsoid named in the grid
    header (lcc / tmerc / aeqd / eqc) and apply ``map_rot``.  Re-implementing
    the projections by hand -- especially on a sphere -- guarantees a
    disagreement with the grids NLL itself produced.

*   An NLL grid is a *rectangle in projected x/y*.  A pykonal spherical field
    is a *rectangle in lat/lon*.  These are not the same region, and node
    (i,j,k) of one is not co-located with node (i,j,k) of the other.  The
    conversion is therefore a genuine resampling: build the target grid, map
    every target node back through the projection, and interpolate.  Simply
    transposing and flipping the array (as a naive implementation does)
    shears the model by an amount that grows with distance from the
    projection origin.

*   Target nodes that fall outside the source grid are unavoidable -- the
    lat/lon bounding box of a conic projection's rectangle always spills over
    its edges.  Such nodes are filled by clamping to the nearest edge of the
    source grid rather than by extrapolation, which keeps velocities physical.
    The fraction of clamped nodes is reported.
"""

import os
import warnings

import numpy as np
import nllgrid
import pykonal
import pykonal.transformations
from scipy.interpolate import RegularGridInterpolator

# pykonal's own Earth radius -- must be used for every rho <-> depth
# conversion so that geo2sph/sph2geo round-trip exactly.
EARTH_RADIUS = pykonal.constants.EARTH_RADIUS   # 6371.0 km

__all__ = ['NLL2PVT', 'PVT2NLL', 'PVT2TXT',
           'nll_node_coords', 'nll_velocity', 'geographic_bounds']


# ---------------------------------------------------------------------------
# Grid-value <-> velocity conversion
# ---------------------------------------------------------------------------

def nll_velocity(nll):
    """
    Return the grid values of `nll` as velocity in km/s.

    Handles every NLL model grid type.  ``SLOW_LEN`` stores
    slowness x cell-length, so it requires cubic cells.
    """
    a = np.asarray(nll.array, dtype=np.float64)
    t = nll.type

    if t == 'VELOCITY':
        return a
    if t == 'VELOCITY_METERS':
        return a / 1000.0
    if t == 'SLOWNESS':                      # s/km
        return 1.0 / a
    if t == 'VEL2':                          # (km/s)^2
        return np.sqrt(a)
    if t == 'SLOW2':                         # (s/km)^2
        return 1.0 / np.sqrt(a)
    if t == 'SLOW2_METERS':                  # (s/m)^2
        return 1.0 / (1000.0 * np.sqrt(a))
    if t == 'SLOW_LEN':                      # slowness * cell length [s]
        if not (np.isclose(nll.dx, nll.dy) and np.isclose(nll.dx, nll.dz)):
            raise ValueError(
                f'SLOW_LEN grid requires cubic cells, got '
                f'dx={nll.dx}, dy={nll.dy}, dz={nll.dz}')
        return nll.dx / a

    raise ValueError(
        f'Grid type {t!r} is not a velocity model grid. '
        'Supported: VELOCITY, VELOCITY_METERS, SLOWNESS, VEL2, SLOW2, '
        'SLOW2_METERS, SLOW_LEN.')


def velocity_to_nll(nll, vel):
    """Inverse of :func:`nll_velocity`: velocity in km/s -> `nll.type` units."""
    v = np.asarray(vel, dtype=np.float64)
    t = nll.type

    if t == 'VELOCITY':
        return v
    if t == 'VELOCITY_METERS':
        return v * 1000.0
    if t == 'SLOWNESS':
        return 1.0 / v
    if t == 'VEL2':
        return v ** 2
    if t == 'SLOW2':
        return 1.0 / v ** 2
    if t == 'SLOW2_METERS':
        return 1.0 / (1000.0 * v) ** 2
    if t == 'SLOW_LEN':
        return nll.dx / v

    raise ValueError(f'Unsupported grid type for writing: {t!r}')


# ---------------------------------------------------------------------------
# NLL grid geometry
# ---------------------------------------------------------------------------

def nll_node_coords(nll):
    """
    Node-centre coordinate axes (x, y, z) of an NLL grid, in km.

    Note this is *not* ``nll.get_extent()``, which pads by half a cell on
    each side for plotting and spans ``(n+1)*d`` rather than ``(n-1)*d``.
    """
    x = nll.x_orig + np.arange(nll.nx) * nll.dx
    y = nll.y_orig + np.arange(nll.ny) * nll.dy
    z = nll.z_orig + np.arange(nll.nz) * nll.dz
    return x, y, z


def geographic_bounds(nll, n_edge=401):
    """
    Latitude/longitude bounding box of an NLL grid.

    The image of a projected rectangle is curvilinear, so the extremes do
    not generally lie at the four corners -- for a Lambert grid the extreme
    latitude sits mid-edge and the extreme longitudes at two different
    corners.  The whole boundary is therefore sampled.

    Longitudes are unwrapped relative to ``nll.orig_lon`` so that grids
    crossing the antimeridian give a continuous (possibly >180 deg) range.

    Returns
    -------
    (lat_min, lat_max, lon_min, lon_max) : tuple of float
    """
    x, y, _ = nll_node_coords(nll)
    xs = np.linspace(x[0], x[-1], n_edge)
    ys = np.linspace(y[0], y[-1], n_edge)
    ones = np.ones(n_edge)

    bx = np.concatenate([xs, xs, x[0] * ones, x[-1] * ones])
    by = np.concatenate([y[0] * ones, y[-1] * ones, ys, ys])

    lon, lat = nll.iproject(bx, by)
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)

    if not np.all(np.isfinite(lon)) or not np.all(np.isfinite(lat)):
        raise ValueError(
            'Projection returned non-finite coordinates on the grid boundary; '
            'check the TRANSFORM line in the NLL header.')

    # Unwrap about the projection origin so the range is continuous.
    lon = nll.orig_lon + (lon - nll.orig_lon + 180.0) % 360.0 - 180.0

    return float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max())


def inscribed_bounds(nll, n=600):
    """
    Largest axis-aligned lat/lon box *entirely inside* the NLL grid.

    Whereas :func:`geographic_bounds` returns the enclosing box (which for a
    continental-scale conic grid overshoots the model by 10-20% of its area),
    this returns the biggest box containing no points outside the source
    model, so no velocity has to be invented.  The trade-off is that the
    corners of the real model are discarded.

    Returns
    -------
    (lat_min, lat_max, lon_min, lon_max) : tuple of float
    """
    lat_min, lat_max, lon_min, lon_max = geographic_bounds(nll)
    x, y, _ = nll_node_coords(nll)
    lats = np.linspace(lat_max, lat_min, n)
    lons = np.linspace(lon_min, lon_max, n)
    LAT, LON = np.meshgrid(lats, lons, indexing='ij')
    X, Y = nll.project(LON.ravel(), LAT.ravel())
    inside = ~((X < x[0]) | (X > x[-1]) | (Y < y[0]) | (Y > y[-1]))
    inside = inside.reshape(n, n)

    # maximal all-true rectangle, via the histogram method
    best = (0, None)
    heights = np.zeros(n, dtype=int)
    for i in range(n):
        heights = np.where(inside[i], heights + 1, 0)
        stack = []
        for j in range(n + 1):
            cur = heights[j] if j < n else 0
            start = j
            while stack and stack[-1][1] >= cur:
                s, h = stack.pop()
                area = h * (j - s)
                if area > best[0]:
                    best = (area, (i - h + 1, i, s, j - 1))
                start = s
            stack.append((start, cur))
    if best[1] is None:
        raise ValueError('No inscribed box found; check the projection.')
    i0, i1, j0, j1 = best[1]
    return (float(lats[i1]), float(lats[i0]),
            float(lons[j0]), float(lons[j1]))


# ---------------------------------------------------------------------------
# NLL -> pykonal
# ---------------------------------------------------------------------------

def NLL2PVT(nllfile, pvtfile, npts=None, oversample=1.0,
            bounds='inner', fill='clamp', clip=None, verbose=True):
    """
    Convert a NonLinLoc model grid to a pykonal spherical HDF5 field.

    The output is a regular grid in (radius, colatitude, longitude) covering
    the geographic bounding box of the NLL grid.  Values are trilinearly
    interpolated from the NLL grid after mapping each target node through the
    header's projection.

    Parameters
    ----------
    nllfile : str
        Path to the NLL grid (basename, .hdr or .buf).
    pvtfile : str
        Output path for the pykonal HDF5 file.
    npts : sequence of 3 int, optional
        Target grid size as (n_radius, n_latitude, n_longitude).  By default
        the node count is chosen so that the angular spacing matches the NLL
        cell size at the centre latitude of the grid, which keeps the
        resolution of the source model without aliasing.
    oversample : float
        Multiplier applied to the default `npts`.  Ignored if `npts` is given.
    bounds : {'outer', 'inner'} or 4-tuple
        Geographic extent of the output.  ``'outer'`` (default) is the box
        enclosing the whole NLL grid; because a projected rectangle is
        curvilinear in lat/lon this necessarily includes nodes outside the
        model (~5% for a 1000 km grid, ~14% for a 3000 km one).  ``'inner'``
        is the largest box fully inside the model, so nothing is invented, at
        the cost of discarding the model corners.  A 4-tuple
        ``(lat_min, lat_max, lon_min, lon_max)`` sets it explicitly.
    fill : {'clamp', 'nan'} or float
        What to put at nodes outside the source grid.  ``'clamp'`` takes the
        nearest edge velocity -- reasonable for a small overhang, misleading
        for a continental grid where it extrudes the boundary hundreds of km.
        ``'nan'`` or a float (e.g. a reference-model velocity) makes the
        fabricated region explicit.  Irrelevant when ``bounds='inner'``.

        NOTE: NaN velocities poison a fast-marching solve -- the NaN spreads
        through the whole traveltime field.  Use ``'nan'`` for inspection and
        plotting only, never for a grid you intend to run pykonal on.
    clip : (vmin, vmax), optional
        Clamp velocities into this range, in km/s, before resampling.  Useful
        for pulling sentinel or air-layer values up to something physical
        (e.g. ``clip=(1.5, None)`` to floor the model at the water velocity).
        Either bound may be None.
    verbose : bool
        Print a summary of the conversion.

    Returns
    -------
    pykonal.fields.ScalarField3D
    """
    nll = nllgrid.NLLGrid(nllfile)
    if nll.array is None:
        raise ValueError(f'No buffer data read from {nllfile}')
    if min(nll.nx, nll.ny, nll.nz) < 2:
        raise ValueError(
            f'Grid is degenerate ({nll.nx}x{nll.ny}x{nll.nz}); 2-D NLL grids '
            'cannot be converted to a 3-D spherical field.')

    x, y, z = nll_node_coords(nll)
    vel = nll_velocity(nll)
    if clip is not None:
        vmin, vmax = clip
        n_clipped = int(np.count_nonzero(
            ((vel < vmin) if vmin is not None else False) |
            ((vel > vmax) if vmax is not None else False)))
        vel = np.clip(vel, vmin, vmax)
        if verbose and n_clipped:
            print(f'  clipped {n_clipped} source nodes '
                  f'({100 * n_clipped / vel.size:.2f}%) into {clip} km/s')
    if not np.all(np.isfinite(vel)) or np.any(vel <= 0):
        warnings.warn('Source grid contains non-positive or non-finite '
                      'velocities; these will propagate to the output.')

    if bounds == 'outer':
        lat_min, lat_max, lon_min, lon_max = geographic_bounds(nll)
    elif bounds == 'inner':
        lat_min, lat_max, lon_min, lon_max = inscribed_bounds(nll)
    else:
        lat_min, lat_max, lon_min, lon_max = (float(v) for v in bounds)
    dep_min, dep_max = float(z[0]), float(z[-1])

    # -- target grid size ---------------------------------------------------
    if npts is None:
        lat_mid = 0.5 * (lat_min + lat_max)
        km_per_deg = np.pi * EARTH_RADIUS / 180.0
        dlat = nll.dy / km_per_deg
        dlon = nll.dx / (km_per_deg * max(np.cos(np.radians(lat_mid)), 1e-6))
        n_lat = int(np.ceil(oversample * (lat_max - lat_min) / dlat)) + 1
        n_lon = int(np.ceil(oversample * (lon_max - lon_min) / dlon)) + 1
        n_rad = int(np.ceil(oversample * (nll.nz - 1))) + 1
        npts = (n_rad, n_lat, n_lon)
    npts = np.asarray(npts, dtype=int)
    if np.any(npts < 2):
        raise ValueError(f'npts must be >= 2 in every dimension, got {npts}')
    n_rad, n_lat, n_lon = (int(v) for v in npts)

    # -- target axes --------------------------------------------------------
    # pykonal spherical axes are ascending: rho (up), theta (north->south),
    # phi (west->east).  So depth descends and latitude descends.
    depths = np.linspace(dep_max, dep_min, n_rad)
    lats = np.linspace(lat_max, lat_min, n_lat)
    lons = np.linspace(lon_min, lon_max, n_lon)

    # -- resample -----------------------------------------------------------
    interp = RegularGridInterpolator(
        (x, y, z), vel, method='linear',
        bounds_error=False, fill_value=None)

    LAT, LON = np.meshgrid(lats, lons, indexing='ij')       # (n_lat, n_lon)
    X, Y = nll.project(LON.ravel(), LAT.ravel())            # projection is
    X = np.asarray(X, dtype=np.float64)                     # depth-independent,
    Y = np.asarray(Y, dtype=np.float64)                     # so do it once

    outside = ((X < x[0]) | (X > x[-1]) | (Y < y[0]) | (Y > y[-1]))
    X = np.clip(X, x[0], x[-1])       # clamp to nearest edge rather than
    Y = np.clip(Y, y[0], y[-1])       # extrapolate

    if fill == 'nan':
        warnings.warn(
            'fill="nan" produces a velocity field that will break a '
            'fast-marching solve (NaN propagates through the traveltime '
            'field). Use it for inspection only.')
    if fill == 'clamp':
        fill_value = None
    elif fill == 'nan':
        fill_value = np.nan
    else:
        fill_value = float(fill)

    values = np.empty((n_rad, n_lat, n_lon), dtype=np.float64)
    zq = np.clip(depths, z[0], z[-1])
    out2d = outside.reshape(n_lat, n_lon)
    for i, zi in enumerate(zq):
        pts = np.column_stack([X, Y, np.full(X.shape, zi)])
        layer = interp(pts).reshape(n_lat, n_lon)
        if fill_value is not None:
            layer = np.where(out2d, fill_value, layer)
        values[i] = layer

    # -- build the pykonal field -------------------------------------------
    # Keep longitudes in [0, 360) as pykonal expects a positive phi axis.
    lon_ref = lon_min + 360.0 if lon_min < 0 else lon_min
    if lon_ref + (lon_max - lon_min) > 360.0 and lon_min < 0:
        warnings.warn('Grid straddles the prime meridian; the phi axis will '
                      'span negative longitudes. Check downstream use.')
        lon_ref = lon_min

    pvt = pykonal.fields.ScalarField3D(coord_sys='spherical')
    pvt.min_coords = pykonal.transformations.geo2sph(
        (lat_max, lon_ref, dep_max))           # rho_min, theta_min, phi_min
    pvt.node_intervals = np.array([
        (dep_max - dep_min) / (n_rad - 1),
        np.radians(lat_max - lat_min) / (n_lat - 1),
        np.radians(lon_max - lon_min) / (n_lon - 1),
    ])
    pvt.npts = npts
    pvt.values = np.ascontiguousarray(values)

    if os.path.isfile(pvtfile):
        os.remove(pvtfile)
    pvt.to_hdf(pvtfile)

    if verbose:
        pct = 100.0 * outside.mean()
        print(f'NLL2PVT: {nll.type} {nll.nx}x{nll.ny}x{nll.nz} '
              f'({nll.proj_name}, {nll.proj_ellipsoid}) -> '
              f'{n_rad}x{n_lat}x{n_lon}')
        print(f'  lat  {lat_min:.4f} .. {lat_max:.4f}')
        print(f'  lon  {lon_min:.4f} .. {lon_max:.4f}')
        print(f'  dep  {dep_min:.3f} .. {dep_max:.3f} km')
        print(f'  vel  {np.nanmin(values):.4f} .. {np.nanmax(values):.4f} km/s')
        how = {'clamp': 'edge-clamped'}.get(fill, f'filled with {fill}')
        print(f'  {pct:.2f}% of nodes outside the source grid ({how})')

    return pvt


# ---------------------------------------------------------------------------
# pykonal -> NLL
# ---------------------------------------------------------------------------

def PVT2NLL(pvtfile, nllfile_template, outfile=None, verbose=True):
    """
    Convert a pykonal spherical HDF5 field back to a NonLinLoc grid.

    An existing NLL grid supplies the header: geometry, projection and grid
    type.  Values are resampled from the spherical field onto the NLL nodes,
    which is the exact inverse operation of :func:`NLL2PVT` (not a transpose).

    Parameters
    ----------
    pvtfile : str
        Path to the pykonal HDF5 file.
    nllfile_template : str
        Existing NLL grid used as the header template.
    outfile : str, optional
        Output basename.  Defaults to the template basename with ``.v2``
        inserted after the first dot-separated component.
    verbose : bool

    Returns
    -------
    nllgrid.NLLGrid
    """
    nll = nllgrid.NLLGrid(nllfile_template)
    pvt = pykonal.fields.read_hdf(pvtfile)

    rho = np.linspace(pvt.min_coords[0], pvt.max_coords[0], pvt.npts[0])
    theta = np.linspace(pvt.min_coords[1], pvt.max_coords[1], pvt.npts[1])
    phi = np.linspace(pvt.min_coords[2], pvt.max_coords[2], pvt.npts[2])

    interp = RegularGridInterpolator(
        (rho, theta, phi), np.asarray(pvt.values, dtype=np.float64),
        method='linear', bounds_error=False, fill_value=None)

    x, y, z = nll_node_coords(nll)
    X, Y = np.meshgrid(x, y, indexing='ij')                 # (nx, ny)
    lon, lat = nll.iproject(X.ravel(), Y.ravel())
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)

    # Match the phi branch used by the field.
    phi_mid = np.degrees(0.5 * (phi[0] + phi[-1]))
    lon = phi_mid + (lon - phi_mid + 180.0) % 360.0 - 180.0

    th = np.radians(90.0 - lat)
    ph = np.radians(lon)
    th_c = np.clip(th, theta[0], theta[-1])
    ph_c = np.clip(ph, phi[0], phi[-1])
    outside = (th != th_c) | (ph != ph_c)

    vel = np.empty((nll.nx, nll.ny, nll.nz), dtype=np.float64)
    for k, zk in enumerate(z):
        r = np.clip(EARTH_RADIUS - zk, rho[0], rho[-1])
        pts = np.column_stack([np.full(th_c.shape, r), th_c, ph_c])
        vel[:, :, k] = interp(pts).reshape(nll.nx, nll.ny)

    nll.array = velocity_to_nll(nll, vel).astype(np.float64)

    if outfile is None:
        dirname = os.path.dirname(nll.basename)
        parts = os.path.basename(nll.basename).split('.')
        parts.insert(1, 'v2')
        outfile = os.path.join(dirname, '.'.join(parts))
    nll.basename = outfile

    nll.write_hdr_file()
    nll.write_buf_file()

    if verbose:
        pct = 100.0 * outside.mean()
        print(f'PVT2NLL: {tuple(int(n) for n in pvt.npts)} -> '
              f'{nll.nx}x{nll.ny}x{nll.nz} {nll.type}  [{outfile}]')
        print(f'  vel  {vel.min():.4f} .. {vel.max():.4f} km/s')
        print(f'  {pct:.2f}% of nodes outside the spherical field '
              '(edge-clamped)')

    return nll


# ---------------------------------------------------------------------------
# pykonal -> text
# ---------------------------------------------------------------------------

def PVT2TXT(pvtfile, output_file=None, fmt='%.5f %.5f %.3f %.6f'):
    """
    Dump a pykonal velocity field to a plain-text lon/lat/depth/velocity file,
    shallowest layer first.

    Longitudes are written in the range [-180, 180).
    """
    pvt = pykonal.fields.read_hdf(pvtfile)

    rho = np.linspace(pvt.min_coords[0], pvt.max_coords[0], pvt.npts[0])
    theta = np.linspace(pvt.min_coords[1], pvt.max_coords[1], pvt.npts[1])
    phi = np.linspace(pvt.min_coords[2], pvt.max_coords[2], pvt.npts[2])

    RHO, THETA, PHI = np.meshgrid(rho, theta, phi, indexing='ij')
    lat = 90.0 - np.degrees(THETA)
    lon = (np.degrees(PHI) + 180.0) % 360.0 - 180.0
    depth = EARTH_RADIUS - RHO

    # Shallowest first == largest rho first.
    order = slice(None, None, -1)
    table = np.column_stack([
        lon[order].ravel(),
        lat[order].ravel(),
        depth[order].ravel(),
        np.asarray(pvt.values)[order].ravel(),
    ])

    if output_file is None:
        output_file = pvtfile + '.txt'
    np.savetxt(output_file, table, fmt=fmt,
               header='lon lat depth(km) velocity(km/s)')
    return output_file