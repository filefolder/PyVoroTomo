#!/usr/bin/python3
"""
plot_model.py  –  visualise PyVoroTomo velocity / uncertainty models.
AI reworking (sorry) of some old scripts I've had sitting around

Usage
-----
Edit the CONFIG block below, then run the script.  Choose what to plot by
setting PLOT_MODE to one of:

    'velocity_map'      – horizontal depth slices (map view, Cartopy)
    'dv_map'            – dV/V % anomaly map slices
    'lat_section'       – vertical sections at fixed latitudes
    'lon_section'       – vertical sections at fixed longitudes
    'uncertainty_lat'   – uncertainty at fixed latitudes
    'uncertainty_lon'   – uncertainty at fixed longitudes
    'uncertainty_map'   – horizontal uncertainty slices (map view)
"""


import os

import cartopy.crs as ccrs
import matplotlib.colors as colors
import matplotlib.ticker as mticker
import netCDF4
import numpy as np
import pandas as pd
import pykonal
import pykonal.transformations
from fastkml import kml
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import RegularGridInterpolator, griddata


#  CONFIG  –  edit this section before running
# ============================================================

# --- model run ---
TARGET      = "output_SWAN"     # run directory
ITERATION   = "05"              # iteration number string
HVR         = "6"               # horizontal/vertical ratio used in filename

# --- what to plot ---
WAVE        = "P"               # "P" or "S"
PLOT_MODE   = "velocity_map"    # see module docstring for options
PLOT_DV_PCT = False             # if True, show % anomaly relative to layer mean

# --- depth slices (for map / lat / lon modes) ---
PLT_DEPTH   = np.array([5, 10, 15, 20, 25, 30])   # km

# --- cross-section slices ---
PLT_LAT     = [-34, -33, -32, -31, -30, -29]           # for lat_section
PLT_LON     = [115, 116, 117, 118, 119, 120]           # for lon_section
SECTION_THICKNESS = 0.5                                 # ± degrees for event selection

# --- colour scale (None = auto) ---
VMIN        = None
VMAX        = None
COLORSCHEME = "twilight_shifted_r"

# --- overlay toggles ---
PLOT_EVENTS      = True
PLOT_STATIONS    = False
PLOT_HISTORIC_EQ = False
PLOT_FAULTS      = False
PLOT_CONTOURS    = False

# --- data files ---
TOPO_FILE        = "/grds/topo_27.1.nc" # e.g. a Scripps satellite topography file for contours https://topex.ucsd.edu/pub/global_topo_1min/
STATION_FILE     = "./station.h5" #same file input to PyVoroTomo (TODO FIX THIS)
HISTORIC_EQ_FILE = "./M4plus.tlldm" # a text file of earthquakes structured tim, latitude, longitude, depth, magnitude:
HISTORIC_EQ_MINMAG = 4.0 # minimum magnitude to plot from the above
REMOVE_NETWORKS  = ["6C", "XX"] # don't plot these network codes

# Fault GMT files (add / remove as needed)
FAULT_FILES = [
    "/gmt/text.gmt",
    "/gmt/text_west.gmt",
    "/gmt/SouthwestTerranBoundary.gmt",
    "/gmt/BalingupTerrane.gmt",
]





#  Derived file paths  (built from CONFIG)
# ============================================================

def _model_path(wave, kind):
    """Return the HDF5 path for a given wave type and file kind."""
    prefix = "pwave" if wave == "P" else "swave"
    suffixes = {
        "model":    f"{TARGET}/{ITERATION}.{prefix}_model.h{HVR}.0.h5",
        "variance": f"{TARGET}/{ITERATION}.{prefix}_variance.h{HVR}.0.h5",
        #"quality":  f"{TARGET}/{ITERATION}.{prefix}_quality.h{HVR}.0.h5", # not in use
        "checker_in":  f"{TARGET}/input_checkerboard_{wave}_50.0km.h5",
        "checker_out": f"{TARGET}/recovered_checkerboard_{wave}_50.0km.h5",
    }
    return suffixes[kind]

PMODEL_FILE  = _model_path("P", "model")
SMODEL_FILE  = _model_path("S", "model")
PVAR_FILE    = _model_path("P", "variance")
SVAR_FILE    = _model_path("S", "variance")
EVENTS_FILE  = f"{TARGET}/{ITERATION}.events.h5"
EVENTS0_FILE = f"{TARGET}/00.events.h5" # this plots as grey dots to observe change from initial catalog


#  Utility helpers
# ============================================================

def find_nearest(array, value):
    """Return the index of the element in *array* closest to *value*."""
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array)
        or abs(value - array[idx - 1]) < abs(value - array[idx])
    ):
        return idx - 1
    return idx


def reinterpolate_array(model_vals, upscale_factor=4):
    """Upscale a 2-D array by *upscale_factor* using cubic interpolation."""
    M, N = model_vals.shape
    x, y = np.arange(M), np.arange(N)
    X, Y = np.meshgrid(x, y, indexing="ij")

    x_fine = np.linspace(0, M - 1, M * upscale_factor)
    y_fine = np.linspace(0, N - 1, N * upscale_factor)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine, indexing="ij")

    points = np.vstack((X.ravel(), Y.ravel())).T
    interp = griddata(points, model_vals.ravel(), (X_fine, Y_fine), method="cubic")
    return X_fine, Y_fine, interp


def label_formatter(x, _pos):
    """Gridline label: show integer values only on even degrees."""
    return f"{int(x)}°" if x % 2 == 0 else ""


#  Data loaders
# ============================================================

def load_topo(path):
    """Return (lons, lats, elevations) from a NetCDF4 topography file."""
    ds = netCDF4.Dataset(path)
    return ds["lon"][:], ds["lat"][:], ds["z"][:]


def load_stations(path, remove_networks=None):
    """
    Load a station list via PyVoroTomo HDF5 station file
 
    Returns a list of rows in the format [lon, lat, network, sta, cha, height_km].
    The 'cha' field is set to '??Z' since no channel info available - not needed here anyway
    """
    df = pd.read_hdf(path, key='stations')
    data = [
        [row.longitude, row.latitude, row.network, row.station, '??Z', row.elevation]
        for row in df.itertuples(index=False)
    ]
 
    if remove_networks:
        data = [r for r in data if r[2] not in remove_networks]
    return data


def load_historic_eq(path, minmag=None):
    """Load historic earthquake catalogue (time, lat, lon, depth, mag)."""
    dat = np.loadtxt(path, delimiter=" ", usecols=(1, 2, 3, 4))
    if minmag is not None:
        dat = dat[dat[:, 3] >= minmag]
    return dat


def load_fault_data(path):
    """Parse a GMT-style fault line file into a list of point-lists."""
    with open(path, "r") as fh:
        lines = [l for l in fh if not l.strip().startswith("#")]
    faults = []
    for segment in "".join(lines).split(">"):
        segment = segment.strip()
        if not segment:
            continue
        pts = [tuple(map(float, row.split())) for row in segment.split("\n") if row.strip()]
        if pts:
            faults.append(pts)
    return faults


def get_topo_profile(wlons, wlats, welev, start_point, end_point, num_points=200):
    """
    Interpolate a topographic profile between two (lat, lon) points.

    Returns an elevation array of length *num_points* (metres).
    """
    interp = RegularGridInterpolator((wlats, wlons), welev)
    lats = np.linspace(start_point[0], end_point[0], num_points)
    lons = np.linspace(start_point[1], end_point[1], num_points)
    return interp(np.vstack((lats, lons)).T)


def get_iso_rho(model, rho):
    """
    Build a 2-D array of depth (km) at which velocity first exceeds *rho*.

    Traverses from the bottom upward; returns 0 where the threshold is never
    reached. Needs work!
    """
    nz, ny, nx = model.values.shape
    dz = model.node_invervals[0]
    out = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz - 1, 0, -1):
                if model.values[k, j, i] >= rho:
                    out[i, j] = -k * dz
                    break
    return out


def get_topo_slices(wlons, wlats, lo0, lo1, la0, la1):
    """Return index bounds into the topo grid for the given lon/lat box."""
    i0 = find_nearest(wlons, lo0)
    i1 = find_nearest(wlons, lo1)
    j0 = find_nearest(wlats, la0)
    j1 = find_nearest(wlats, la1)
    return i0, i1, j0, j1


def model_coords(model):
    """Return (lat, lon, depth) 1-D arrays from a pykonal ScalarField3D."""
    minc = pykonal.transformations.sph2geo(model.min_coords)
    maxc = pykonal.transformations.sph2geo(model.max_coords)
    npts = model.npts
    lat   = np.linspace(minc[0], maxc[0], npts[1])
    lon   = np.linspace(minc[1], maxc[1], npts[2])
    depth = np.linspace(minc[2], maxc[2], npts[0])
    return lat, lon, depth



#  Plot helpers
# ============================================================

def _add_map_gridlines(ax, lon_step=1, lat_step=1, fontsize=9):
    gl = ax.gridlines(
        draw_labels=True, color="gray", alpha=0.5, linestyle="--",
        x_inline=False, y_inline=False, rotate_labels=False,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.MultipleLocator(lon_step)
    gl.ylocator = mticker.MultipleLocator(lat_step)
    gl.xlabel_style = {"size": fontsize}
    gl.ylabel_style = {"size": fontsize}
    gl.xformatter = mticker.FuncFormatter(label_formatter)
    gl.yformatter = mticker.FuncFormatter(label_formatter)


def _add_map_overlays(ax, d, depth, pltdepth, events0, events,
                      historic_eq, faults, stations,
                      welev=None, topo_extent=None, topo_levels=None):
    """Add events, stations, faults, contours to a map axis."""
    if PLOT_CONTOURS and welev is not None and topo_extent is not None:
        i0, i1, j0, j1 = topo_extent
        ax.contour(
            welev[j0:j1, i0:i1], levels=topo_levels or [],
            extent=(topo_extent[0], topo_extent[1],
                    topo_extent[2], topo_extent[3]),
            alpha=0.5, transform=ccrs.Geodetic(),
        )

    if PLOT_STATIONS and stations is not None:
        ax.scatter(*zip(*stations), marker="^", c="lightgreen",
                   s=15, alpha=0.5, transform=ccrs.Geodetic())

    if PLOT_FAULTS and faults:
        plot_fault_lines(ax, faults)

    if PLOT_EVENTS and events0 is not None and events is not None:
        depth0 = -5 if d == 0 else pltdepth[d - 1]
        depth1 = pltdepth[d]
        try:
            ev0 = events0[(events0["depth"] >= depth0) & (events0["depth"] <= depth1)]
            ev  = events[(events["depth"] >= depth0) & (events["depth"] <= depth1)]
            ax.plot(ev0["longitude"], ev0["latitude"], "+",
                    c="grey", markersize=0.7, transform=ccrs.Geodetic())
            ax.plot(ev["longitude"], ev["latitude"], "x",
                    c="black", markersize=0.6, transform=ccrs.Geodetic())
        except Exception:
            print("Could not overlay events.")

    if PLOT_HISTORIC_EQ and historic_eq is not None:
        ax.plot(historic_eq[:, 1], historic_eq[:, 0], "*",
                c="red", markersize=3, transform=ccrs.Geodetic())


def plot_fault_lines(ax, faults, width=1):
    """Draw fault traces on a Cartopy axis."""
    for fault in faults:
        if len(fault[0]) == 3:
            lo, la, _ = zip(*fault)
        else:
            lo, la = zip(*fault)
        ax.plot(lo, la, "k-", linewidth=width, transform=ccrs.Geodetic())


def _depth_slice(model_vals, as_dv_pct=False):
    """Optionally convert a depth slice to % anomaly relative to layer mean."""
    if as_dv_pct:
        mean = np.mean(model_vals)
        return (model_vals / mean - 1.0) * 100.0
    return model_vals



#  Map-view plots  (Cartopy)
# ============================================================

def plot_map(model, lat, lon, depth, pltdepth,
             events0=None, events=None, historic_eq=None,
             faults=None, stations=None, welev=None,
             title="", as_dv_pct=False):
    """
    6-panel horizontal depth-slice map.

    Parameters
    ----------
    as_dv_pct : bool
        If True each panel shows % deviation from layer mean.
    """
    vmin = VMIN
    vmax = VMAX

    fig, ax = plt.subplots(
        nrows=2, ncols=3, figsize=(16, 12),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    la1, lo0, _ = pykonal.transformations.sph2geo(model.min_coords)
    la0, lo1, _ = pykonal.transformations.sph2geo(model.max_coords)
    topo_ext = get_topo_slices(welev[0], welev[1], lo0, lo1, la0, la1) if welev else None

    for j in range(3):
        for i in range(2):
            d = j + i * 3
            idx = find_nearest(depth, pltdepth[d])
            vals = _depth_slice(model.values[idx, :, :], as_dv_pct)

            _vmin = vmin if vmin is not None else vals.min()
            _vmax = vmax if vmax is not None else vals.max()

            lo2d, la2d = np.meshgrid(lon, lat)
            im = ax[i, j].pcolormesh(
                lo2d, la2d, vals,
                cmap=COLORSCHEME, vmin=_vmin, vmax=_vmax,
                transform=ccrs.PlateCarree(),
            )
            ax[i, j].set_extent([lon.min(), lon.max(), lat.min(), lat.max()],
                                 crs=ccrs.PlateCarree())
            ax[i, j].coastlines(color="darkblue")
            ax[i, j].set_title(f"{pltdepth[d]} km depth", fontsize=14)
            plt.colorbar(im, ax=ax[i, j], shrink=0.5, aspect=10, pad=0.01)

            _add_map_gridlines(ax[i, j])
            _add_map_overlays(
                ax[i, j], d, depth, pltdepth,
                events0, events, historic_eq, faults, stations,
                welev=welev[2] if welev else None,
                topo_extent=topo_ext,
            )

    fig.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.show()


#  Cross-section plots  (lon–depth  or  lat–depth)
# ============================================================

def plot_section(model, lat, lon, depth, plt_vals,
                 label="latitude",
                 events0=None, events=None, historic_eq=None,
                 title=""):
    """
    6-panel vertical cross-section plot.

    Parameters
    ----------
    label : str
        "latitude" or "longitude" – which axis is fixed.
    plt_vals : array-like
        Six values (latitudes or longitudes) to slice at.
    """
    thickness = SECTION_THICKNESS
    vmin = VMIN if VMIN is not None else None
    vmax = VMAX if VMAX is not None else None
    absmin_ = vmin or model.values.min()
    absmax_ = vmax or model.values.max()

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 9))

    for j in range(3):
        for i in range(2):
            d = j + i * 3
            val = plt_vals[d]

            if label == "latitude":
                idx = find_nearest(lat, val)
                vals = model.values[:, idx, :]
                x_axis, x_arr = lon, "longitude"
            else:
                idx = find_nearest(lon, val)
                vals = model.values[:, :, idx]
                x_axis, x_arr = lat, "latitude"

            _, _, vals_up = reinterpolate_array(vals, upscale_factor=4)

            _vmin = max(absmin_, vals_up.min())
            _vmax = min(absmax_, vals_up.max())
            dc = 0.04
            boundaries = np.arange(_vmin, _vmax + dc, dc)
            norm = colors.BoundaryNorm(
                boundaries, plt.get_cmap(COLORSCHEME).N, clip=True
            )

            extent = (x_axis.min(), x_axis.max(), depth.min(), depth.max())
            im = ax[i, j].imshow(
                vals_up, cmap=COLORSCHEME, norm=norm,
                extent=extent, aspect="auto",
            )

            divider = make_axes_locatable(ax[i, j])
            cax = divider.append_axes("right", size="2%", pad=0.01)
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

            ax[i, j].set_title(f"{val:.2f}° {label}")
            ax[i, j].set_xlim([x_axis.min(), x_axis.max()])
            ax[i, j].set_ylim([depth.max(), depth.min()])

            tick_step = 1.0
            ticks = np.arange(x_axis.min(), x_axis.max(), tick_step)
            ax[i, j].set_xticks(ticks)
            ax[i, j].set_xticklabels([f"{v:.1f}" for v in ticks])

            # Overlay events
            if PLOT_EVENTS and events0 is not None and events is not None:
                if label == "latitude":
                    ev0 = events0[abs(events0["latitude"] - val) <= thickness]
                    ev  = events[abs(events["latitude"] - val) <= thickness]
                    xcol0, xcol = "longitude", "longitude"
                else:
                    ev0 = events0[abs(events0["longitude"] - val) <= thickness]
                    ev  = events[abs(events["longitude"] - val) <= thickness]
                    xcol0, xcol = "latitude", "latitude"
                ax[i, j].plot(ev0[xcol0], ev0["depth"], "+",
                              c="grey", markersize=0.8)
                ax[i, j].plot(ev[xcol], ev["depth"], "x",
                              c="black", markersize=0.8)

            if PLOT_HISTORIC_EQ and historic_eq is not None:
                if label == "latitude":
                    mask = abs(historic_eq[:, 0] - val) <= thickness
                    ax[i, j].plot(historic_eq[mask, 1], historic_eq[mask, 2],
                                  "*", c="red", markersize=5)
                else:
                    mask = abs(historic_eq[:, 1] - val) <= thickness
                    ax[i, j].plot(historic_eq[mask, 0], historic_eq[mask, 2],
                                  "*", c="red", markersize=5)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


#  Uncertainty map
# ============================================================

def plot_uncertainty_map(sigmodel, lat, lon, depth, pltdepth, title=""):
    """6-panel horizontal uncertainty slice map (map view)."""
    fig, ax = plt.subplots(
        nrows=2, ncols=3, figsize=(16, 12),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    _vmin = sigmodel.values.min()
    _vmax = sigmodel.values.max()

    for j in range(3):
        for i in range(2):
            d = j + i * 3
            idx = find_nearest(depth, pltdepth[d])
            vals = sigmodel.values[idx, :, :]

            lo2d, la2d = np.meshgrid(lon, lat)
            im = ax[i, j].pcolormesh(
                lo2d, la2d, vals,
                cmap="gray_r", vmin=_vmin, vmax=_vmax,
                transform=ccrs.PlateCarree(),
            )
            ax[i, j].set_extent(
                [lon.min(), lon.max(), lat.min(), lat.max()],
                crs=ccrs.PlateCarree(),
            )
            ax[i, j].coastlines(color="darkblue")
            ax[i, j].set_title(f"{pltdepth[d]} km depth", fontsize=14)
            plt.colorbar(im, ax=ax[i, j], shrink=0.5, aspect=10, pad=0.01)
            _add_map_gridlines(ax[i, j])

    fig.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.show()



#  Main
# ============================================================

def main():
    # --- load topo ---
    wlons, wlats, welev = load_topo(TOPO_FILE)
    topo = (wlons, wlats, welev)

    # --- load station / event / fault data ---
    station_data = load_stations(STATION_FILE, remove_networks=REMOVE_NETWORKS)
    historic_eq  = load_historic_eq(HISTORIC_EQ_FILE, minmag=HISTORIC_EQ_MINMAG)

    faults = []
    for ff in FAULT_FILES:
        if os.path.isfile(ff):
            faults += load_fault_data(ff)

    # --- load model ---
    model_file = PMODEL_FILE if WAVE == "P" else SMODEL_FILE
    var_file   = PVAR_FILE   if WAVE == "P" else SVAR_FILE
    wave_label = "P" if WAVE == "P" else "S"

    model    = pykonal.fields.read_hdf(model_file)
    sigmodel = pykonal.fields.read_hdf(var_file)
    sigmodel.values = np.sqrt(sigmodel.values)   # variance → std dev

    lat, lon, depth = model_coords(model)
    title = f"{wave_label} Velocity (km/s)"

    # --- filter stations to model extent ---
    stations = [
        [e[0], e[1]] for e in station_data
        if lon.min() <= e[0] <= lon.max()
        and lat.min() <= e[1] <= lat.max()
        and e[4][-1] == "Z"
    ]

    # --- load events ---
    events0 = events = None
    try:
        events0 = pd.read_hdf(EVENTS0_FILE, key="events")
        events  = pd.read_hdf(EVENTS_FILE,  key="events")
        # events  = events[abs(events["residual"]) < 1.0] # optionally filter
    except Exception as exc:
        print(f"Could not load events: {exc}")

    # --- dispatch plot mode ---
    mode = PLOT_MODE

    if mode in ("velocity_map", "dv_map"):
        as_dv = (mode == "dv_map") or PLOT_DV_PCT
        plot_map(
            model, lat, lon, depth, PLT_DEPTH,
            events0=events0, events=events,
            historic_eq=historic_eq if PLOT_HISTORIC_EQ else None,
            faults=faults if PLOT_FAULTS else None,
            stations=stations if PLOT_STATIONS else None,
            welev=topo if PLOT_CONTOURS else None,
            title=title if not as_dv else f"dV/V % {wave_label}",
            as_dv_pct=as_dv,
        )

    elif mode == "lat_section":
        plot_section(
            model, lat, lon, depth, PLT_LAT, label="latitude",
            events0=events0, events=events,
            historic_eq=historic_eq if PLOT_HISTORIC_EQ else None,
            title=f"{wave_label} Velocity – latitude sections",
        )

    elif mode == "lon_section":
        plot_section(
            model, lat, lon, depth, PLT_LON, label="longitude",
            events0=events0, events=events,
            historic_eq=historic_eq if PLOT_HISTORIC_EQ else None,
            title=f"{wave_label} Velocity – longitude sections",
        )

    elif mode in ("uncertainty_lat", "uncertainty_lon"):
        label = "latitude" if mode == "uncertainty_lat" else "longitude"
        vals  = PLT_LAT if label == "latitude" else PLT_LON
        plot_section(
            sigmodel, lat, lon, depth, vals, label=label,
            events0=events0, events=events,
            title=f"{wave_label} Uncertainty – {label} sections",
        )

    elif mode == "uncertainty_map":
        plot_uncertainty_map(
            sigmodel, lat, lon, depth, PLT_DEPTH,
            title=f"{wave_label} Velocity Uncertainty (km/s)",
        )

    else:
        raise ValueError(f"Unknown PLOT_MODE: {mode!r}")


if __name__ == "__main__":
    main()