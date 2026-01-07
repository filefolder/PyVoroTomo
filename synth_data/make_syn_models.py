#!/usr/bin/python3

import numpy as np
import pandas as pd
import pykonal
from pykonal.transformations import geo2sph, sph2geo

# for smoothing the vel models a little
from scipy.interpolate import interp1d

EARTH_RADIUS = 6371.0

def create_1D_velocity_model(model_bounds, model_npts):
    """ Create 3D velocity model based on AK135 reference structure """

    # AK135 reference model global avg
    ak135_data_global = np.array([
        [0.00,  1.4500, 0.0500], #added a tiny S velocity
        [3.00,  1.4500, 0.0500], 
        [3.00,  1.6500, 1.0000],
        [3.30,  1.6500, 1.0000],
        [3.30,  5.8000, 3.2000],
        [10.00, 5.8000, 3.2000],
        [10.00, 6.8000, 3.9000],
        [18.00, 6.8000, 3.9000],
        [18.00, 8.0355, 4.4839],
        [43.00, 8.0379, 4.4856],
        [80.00, 8.0400, 4.4800],
        [80.00, 8.0450, 4.4900],
        [120.0, 8.0505, 4.5000]
    ])

    # this is fudged a bit to be smoother across boundaries
    ak135_data_continent = np.array([
        [0.00,   5.700, 3.450],
        [7.00,  5.800, 3.460], 
        [21.00,  6.400, 3.800],
        [22.00,  6.500, 3.850],
        [39.00,  8.040, 4.480],
        [77.50,  8.045, 4.490],
        [120.00, 8.050, 4.500]
    ])

    ak135_data = ak135_data_continent

    def interpolate_ak135(depth, phase='P'):
        depths = ak135_data[:, 0]
        velocities = ak135_data[:, 1] if phase == 'P' else ak135_data[:, 2]
        spl = interp1d(depths, velocities, kind='linear')
        return spl(depth)

    # Create velocity models
    nz, ny, nx = model_npts

    # Create P-wave model
    pwave_model = pykonal.fields.ScalarField3D(coord_sys="spherical")
    pwave_model.min_coords = np.array([model_bounds[0][0], model_bounds[1][0], model_bounds[2][0]])
    pwave_model.node_intervals = np.array([
        (model_bounds[0][1] - model_bounds[0][0]) / (nz - 1),
        (model_bounds[1][1] - model_bounds[1][0]) / (ny - 1),
        (model_bounds[2][1] - model_bounds[2][0]) / (nx - 1)
    ])
    pwave_model.npts = np.array(model_npts)

    # Create S-wave model
    swave_model = pykonal.fields.ScalarField3D(coord_sys="spherical")
    swave_model.min_coords = pwave_model.min_coords.copy()
    swave_model.node_intervals = pwave_model.node_intervals.copy()
    swave_model.npts = pwave_model.npts.copy()

    p_velocities = np.zeros((nz, ny, nx))
    s_velocities = np.zeros((nz, ny, nx))

    for iz in range(nz):
        # Convert model coordinate to depth. model.values[0] = deepest z-slice
        rho = model_bounds[0][0] + iz * pwave_model.node_intervals[0]
        depth = EARTH_RADIUS - rho

        # Get AK135 velocities for this depth
        vp = interpolate_ak135(depth, 'P')
        vs = interpolate_ak135(depth, 'S')

        for iy in range(ny):
            for ix in range(nx):
                p_velocities[iz, iy, ix] = vp
                s_velocities[iz, iy, ix] = vs

    pwave_model.values = p_velocities
    swave_model.values = s_velocities

    return pwave_model, swave_model


def create_synthetic_test_data(model_bounds, model_npts):
    """
    Create synthetic test data with regular patterns for debugging resolution tests.

    Parameters:
    -----------
    model_bounds : tuple
        ((rho_min, rho_max), (theta_min, theta_max), (phi_min, phi_max))
    model_npts : tuple
        (nz, ny, nx) number of grid points

    Returns:
    --------
    events, stations, arrivals, velocity_model
    """

    # Convert corner points
    corner1 = sph2geo([model_bounds[0][0], model_bounds[1][0], model_bounds[2][0]])
    corner2 = sph2geo([model_bounds[0][1], model_bounds[1][1], model_bounds[2][1]])

    lat_values = [corner1[0], corner2[0]]
    lon_values = [corner1[1], corner2[1]]
    depth_values = [corner1[2], corner2[2]]

    lat_min, lat_max = min(lat_values), max(lat_values)
    lon_min, lon_max = min(lon_values), max(lon_values)
    depth_min, depth_max = min(depth_values), max(depth_values)

    # Create regular grid of stations
    n_stations_lat = 6
    n_stations_lon = 8
    station_lats = np.linspace(lat_min + 0.1, lat_max - 0.1, n_stations_lat)
    station_lons = np.linspace(lon_min + 0.1, lon_max - 0.1, n_stations_lon)

    stations = []
    for i, lat in enumerate(station_lats):
        for j, lon in enumerate(station_lons):
            stations.append({
                'network': 'SY',  # Synthetic
                'station': f'S{i:02d}{j:02d}',
                'latitude': lat,
                'longitude': lon,
                'elevation': 0.0,
                'starttime': 315532800, #1980
                'endtime': 4070908800 #2099
            })
    stations = pd.DataFrame(stations)

    # Create regular grid of events at different depths
    n_events_lat = 10
    n_events_lon = 10
    n_depths = 4

    event_lats = np.linspace(lat_min + 0.1, lat_max - 0.1, n_events_lat)
    event_lons = np.linspace(lon_min + 0.1, lon_max - 0.1, n_events_lon)
    event_depths = np.linspace(depth_min + 1, min(depth_max, 20), n_depths)  # 1-20 km depth

    events = []
    event_id = 1
    for depth in event_depths:
        for lat in event_lats:
            for lon in event_lons:
                events.append({
                    'event_id': int(event_id),
                    'latitude': lat, #+np.random.uniform(0, 0.01),
                    'longitude': lon, #+np.random.uniform(0, 0.01),
                    'depth': depth,
                    'time': 946684800,  # All simultaneous (2000-01-01)
                    'residual': 0, # np.random.uniform(0, 0.1),
                    'weight': 1.0,
                    'source_id': "event_%03d" % event_id
                })
                event_id += 1
    events = pd.DataFrame(events)

    # Create velocity models
    pwave_model, swave_model = create_1D_velocity_model(model_bounds, model_npts)

    # Generate accurate synthetic arrivals using pykonal
    arrivals = []

    for _, station in stations.iterrows():
        station_coords = geo2sph([station['latitude'], station['longitude'], station['elevation']])

        # Compute traveltime tables for this station using both P and S models
        for phase, model in [('P', pwave_model), ('S', swave_model)]:
            # Create solver
            solver = pykonal.solver.PointSourceSolver(coord_sys="spherical")
            solver.vv.min_coords = model.min_coords
            solver.vv.node_intervals = model.node_intervals
            solver.vv.npts = model.npts
            solver.vv.values = model.values
            solver.src_loc = station_coords
            solver.solve()

            # Compute traveltimes to all events
            for _, event in events.iterrows():
                event_coords = geo2sph([event['latitude'], event['longitude'], event['depth']])

                try:
                    #traveltime = solver.tt.value(event_coords) # this is what the locator uses?

                    traveltime_array = solver.tt.resample(event_coords.reshape(1, -1)) # in theory should be "perfect" since replicates exact procedure..
                    traveltime = traveltime_array[0]

                    arrivals.append({
                        'event_id': int(event['event_id']),
                        'network': station['network'],
                        'station': station['station'],
                        'phase': phase,
                        'time': event['time'] + traveltime,
                        'residual': 0, # np.random.uniform(0, 0.05), 
                        'weight': 1.0
                    })
                except:
                    # Skip if event is outside model bounds
                    continue

    arrivals = pd.DataFrame(arrivals)

    print(f"Created synthetic dataset:")
    print(f"  {len(events)} events")
    print(f"  {len(stations)} stations")  
    print(f"  {len(arrivals)} arrivals")
    print(f"  Velocity model: {model_npts} nodes")
    print(f"  P Velocity range: {pwave_model.values.min():.2f} to {pwave_model.values.max():.2f} km/s")
    print(f"  S Velocity range: {swave_model.values.min():.2f} to {swave_model.values.max():.2f} km/s")

    return events, stations, arrivals, pwave_model, swave_model


def create_model_geometry(lon_min, lon_max, lat_min, lat_max, depth_min_km, depth_max_km, resolution_km):
    """
    Create model bounds and grid dimensions for a spherical coordinate system.

    Parameters:
    -----------
    lon_min, lon_max : float
        Longitude bounds in degrees
    lat_min, lat_max : float  
        Latitude bounds in degrees
    depth_min_km, depth_max_km : float
        Depth bounds in kilometers (positive downward)
    resolution_km : float
        Target cubic resolution in kilometers

    Returns:
    --------
    model_bounds : tuple
        ((rho_min, rho_max), (theta_min, theta_max), (phi_min, phi_max))
    model_npts : tuple
        (nz, ny, nx) grid dimensions
    """

    # Use pykonal's actual coordinate transformations
    corner1 = geo2sph([lat_min, lon_min, depth_min_km])
    corner2 = geo2sph([lat_max, lon_max, depth_max_km])

    rho_min = min(corner1[0], corner2[0])
    rho_max = max(corner1[0], corner2[0])
    theta_min = min(corner1[1], corner2[1])
    theta_max = max(corner1[1], corner2[1])
    phi_min = min(corner1[2], corner2[2])
    phi_max = max(corner1[2], corner2[2])

    model_bounds = ((rho_min, rho_max), (theta_min, theta_max), (phi_min, phi_max))

    # Calculate grid dimensions for target resolution
    # Depth dimension
    depth_range_km = depth_max_km - depth_min_km
    nz = int(np.ceil(depth_range_km / resolution_km)) + 1

    # Latitude dimension 
    lat_range_deg = lat_max - lat_min
    lat_range_km = lat_range_deg * (EARTH_RADIUS * np.pi / 180)  # Arc length
    ny = int(np.ceil(lat_range_km / resolution_km)) + 1

    # Longitude dimension (varies with latitude)
    lon_range_deg = lon_max - lon_min
    avg_lat = (lat_min + lat_max) / 2
    lon_range_km = lon_range_deg * (EARTH_RADIUS * np.pi / 180) * np.cos(np.radians(avg_lat))
    nx = int(np.ceil(lon_range_km / resolution_km)) + 1

    model_npts = (nz, ny, nx)

    # Calculate actual resolution achieved
    actual_depth_res = depth_range_km / (nz - 1)
    actual_lat_res = lat_range_km / (ny - 1)
    actual_lon_res = lon_range_km / (nx - 1)

    print(f"Model geometry created:")
    print(f"  Geographic bounds: {lat_min:.2f}°N to {lat_max:.2f}°N, {lon_min:.2f}°E to {lon_max:.2f}°E")
    print(f"  Depth range: {depth_min_km:.1f} to {depth_max_km:.1f} km")
    print(f"  Grid dimensions: {nz} × {ny} × {nx} = {nz*ny*nx:,} nodes")
    print(f"  Target resolution: {resolution_km:.1f} km")
    print(f"  Actual resolution: depth={actual_depth_res:.2f}km, lat={actual_lat_res:.2f}km, lon={actual_lon_res:.2f}km")

    return model_bounds, model_npts


def extract_geographic_bounds(model_bounds):
    """
    Convert spherical model bounds to geographic coordinates using pykonal's sph2geo.
    """

    rho_bounds, theta_bounds, phi_bounds = model_bounds
    rho_min, rho_max = rho_bounds
    theta_min, theta_max = theta_bounds  
    phi_min, phi_max = phi_bounds

    # Test the 4 corners that matter for geographic bounds
    corners_sph = [
        [rho_min, theta_min, phi_min],  # Deep, north, west
        [rho_min, theta_max, phi_max],  # Deep, south, east
        [rho_max, theta_min, phi_min],  # Shallow, north, west
        [rho_max, theta_max, phi_max]   # Shallow, south, east
    ]

    corners_geo = [sph2geo(corner) for corner in corners_sph]

    lats = [corner[0] for corner in corners_geo]
    lons = [corner[1] for corner in corners_geo] 
    depths = [corner[2] for corner in corners_geo]

    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    depth_min, depth_max = min(depths), max(depths)

    return lat_min, lat_max, lon_min, lon_max, depth_min, depth_max


#get bounds (sort of roundabout way to do this)
model_bounds, model_npts = create_model_geometry(
     lon_min=113, lon_max=117, 
     lat_min=-36, lat_max=-32,
     depth_min_km=0, depth_max_km=50,
     resolution_km=3.0)




extract_geographic_bounds(model_bounds)

events, stations, arrivals, pwave_model, swave_model = create_synthetic_test_data(model_bounds,model_npts)

events_file = "synthetic_events.h5"
events.to_hdf(events_file, key="events", mode="w")
arrivals.to_hdf(events_file, key="arrivals", mode="a")

stations_file = "synthetic_stations.h5"
stations.to_hdf(stations_file, key="stations", mode="w")

pmodel_file = "synthetic_pwave_model.h5"
smodel_file = "synthetic_swave_model.h5"
pwave_model.to_hdf(pmodel_file)
swave_model.to_hdf(smodel_file)
