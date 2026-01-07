import argparse
import configparser
import logging
import mpi4py.MPI as MPI
import os
import signal
import time
import numpy as np
import pykonal
import pandas as pd

from . import _constants

# for station_dict
geo2sph = pykonal.transformations.geo2sph

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()


# get timestamp (mostly for the output dir)
if RANK == _constants.ROOT_RANK:
    stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
else:
    stamp = None
stamp = COMM.bcast(stamp, root=_constants.ROOT_RANK)



def abort():
    """ quick abort """
    shutdown_logging()
    COMM.Abort()


def signal_handler(sig, frame):
    """
    A utility function to handle interrupting signals.
    """
    try:
        shutdown_logging()
    except:
        pass
    COMM.Abort()


def configure_logger(name, log_file, verbose=False):
    """
    A utility function to configure logging. Return True on successful
    execution.
    """

    # Define the date format for logging.
    datefmt        ="%Y%jT%H:%M:%S"
    processor_name = MPI.Get_processor_name()
    rank           = MPI.COMM_WORLD.Get_rank()

    if verbose is True:
        level = logging.DEBUG
    else:
        level = logging.INFO if rank == _constants.ROOT_RANK else logging.WARNING
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if level == logging.DEBUG:
        fmt = f"%(asctime)s::%(levelname)s::%(funcName)s()::"\
              f"{processor_name}::{rank:04d}:: %(message)s"
    else:
        fmt = f"%(asctime)s::%(levelname)s::{rank:04d}:: %(message)s"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return True


def get_logger(name):
    """ Return the logger for *name* """
    return logging.getLogger(name)


def log_errors(logger):
    """ A decorator to for error logging """
    def _decorate_func(func):
        """
        An hidden decorator to permit the logger to be passed in as a
        decorator argument.
        """

        def _decorated_func(*args, **kwargs):
            try:
                return (func(*args, **kwargs))
            except Exception as exc:
                logger.error(
                    f"{func.__name__}() raised {type(exc)}: {exc}"
                )
                raise (exc)

        return _decorated_func

    return _decorate_func


def shutdown_logging():
    """ Close all logging handlers """
    try:
        for logger_name in list(logging.Logger.manager.loggerDict.keys()) + ['']:
            logger = logging.getLogger(logger_name)
            for handler in logger.handlers[:]:
                try:
                    handler.close()
                    logger.removeHandler(handler)
                except:
                    pass
    except:
        pass


def root_only(rank, default=True, barrier=True):
    """
    A decorator for functions and methods that only the root rank should execute.
    """
    def _decorate_func(func):
        """
        An hidden decorator to permit the rank to be passed in as a
        decorator argument.
        """

        def _decorated_func(*args, **kwargs):
            if rank == _constants.ROOT_RANK:
                value = func(*args, **kwargs)
                if barrier is True:
                    COMM.barrier()
                return (value)
            else:
                if barrier is True:
                    COMM.barrier()
                return (default)

        return _decorated_func

    return _decorate_func


class ArgumentParser(argparse.ArgumentParser):
    """ A simple subclass to abort all threads if argument parsing fails """
    def exit(self, status=0, message=None):

        self.print_usage()

        if message is not None:
            print(message)

        abort()


def parse_args():
    """ Parse and return command line arguments """

    parser = ArgumentParser()

    #parser.add_argument(
    #    "events",
    #    type=str,
    #    help="Input event (origins and phases) data file in HDF5 format."
    #)
    #parser.add_argument(
    #    "network",
    #    type=str,
    #    help="Input network geometry file in HDF5 format."
    #)
    parser.add_argument(
        "-c",
        "--configuration_file",
        type=str,
        default=f"{parser.prog}.cfg",
        help="Configuration file."
    )
    #parser.add_argument(
    #    "-l",
    #    "--log_file",
    #    type=str,
    #    help="Log file."
    #)
    #parser.add_argument(
    #    "-o",
    #    "--output_dir",
    #    type=str,
    #    default=f"output_{stamp}",
    #    help="Output directory."
    #)
    parser.add_argument(
        "-r",
        "--relocate_first",
        action="store_true",
        help="Relocate events before first model update."
    )
    #parser.add_argument(
    #    "-s",
    #    "--scratch_dir",
    #    type=str,
    #    help="Scratch directory."
    #)
    parser.add_argument(
        "-t",
        "--test_only",
        action="store_true",
        help="Only run a resolution test."
    )    
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging."
    )
    parser.add_argument(
        "-x",
        "--output_realizations",
        action="store_true",
        help="Save realizations to disk."
    )

    args = parser.parse_args()

    #if args.log_file is None:
    #    args.log_file = os.path.join(cfg.output_dir, f"{parser.prog}.log")

    for attr in (
        #"events",
        #"network",
        #"log_file"
        #"output_dir"
        "configuration_file",):


        _attr = getattr(args, attr)
        _attr = os.path.abspath(_attr)
        setattr(args, attr, _attr)


    #if RANK == _constants.ROOT_RANK:
    #    os.makedirs(args.output_dir, exist_ok=True)

    #if args.scratch_dir is not None:
    #    args.scratch_dir = os.path.abspath(args.scratch_dir)
    #    if RANK == _constants.ROOT_RANK:
    #        os.makedirs(args.scratch_dir, exist_ok=True)

    COMM.barrier()

    return args


def parse_cfg(configuration_file):
    """ Parse and return contents of the configuration file """

    cfg = dict()
    parser = configparser.ConfigParser()
    parser.read(configuration_file)

    _cfg = dict()

    _cfg["adaptive_data_weight"] = parser.getfloat(
        "algorithm",
        "adaptive_data_weight",
        fallback=0.5
    )
    _cfg["density_to_gradient_weight"] = parser.getfloat(
        "algorithm",
        "density_to_gradient_weight",
        fallback=0.5
    )
    _cfg["niter"] = parser.getint(
        "algorithm",
        "niter",
        fallback=1
    )
    _cfg["kvoronoi"] = parser.getfloat(
        "algorithm",
        "kvoronoi",
        fallback=5
    )
    _cfg["nvoronoi"] = parser.getint(
        "algorithm",
        "nvoronoi",
        fallback=400
    )
    _cfg["min_rays_per_cell"] = parser.getint(
        "algorithm",
        "min_rays_per_cell",
        fallback=3
    )
    _cfg["paretos_alpha"] = parser.getfloat(
        "algorithm",
        "paretos_alpha",
        fallback=1.5
    )
    _cfg["phase_order"] = [str(v).upper() for v in parser.get(
        "algorithm",
        "phase_order",
        fallback='P,S'
        ).split(",")
    ]
    _cfg["hvr"] = parser.getfloat(
        "algorithm",
        "hvr",
        fallback=3
    )
    _cfg["min_dist"] = parser.getfloat(
        "algorithm",
        "min_dist",
        fallback=1
    )
    _cfg["max_dist"] = parser.getfloat(
        "algorithm",
        "max_dist",
        fallback=155
    )
    _cfg["cutoff_depth"] = parser.getfloat(
        "algorithm",
        "cutoff_depth",
        fallback=50
    )
    _cfg["nreal"] = parser.getint(
        "algorithm",
        "nreal"
    )
    _cfg["k_medians_percent"] = parser.getfloat(
        "algorithm",
        "k_medians_percent",
        fallback=15
    )
    _cfg["min_narrival"] = parser.getint(
        "algorithm",
        "min_narrival",
        fallback=9
    )
    _cfg["narrival"] = parser.getint(
        "algorithm",
        "narrival"
    )
    _cfg["nevent"] = parser.getint(
        "algorithm",
        "nevent"
    )
    _cfg["narrival_percent"] = parser.getfloat(
        "algorithm",
        "narrival_percent",
        fallback=-1
    )
    _cfg["nevent_percent"] = parser.getfloat(
        "algorithm",
        "nevent_percent",
        fallback=-1
    )
    _cfg["outlier_removal_factor"] = parser.getfloat(
        "algorithm",
        "outlier_removal_factor",
        fallback=1.5
    )
    _cfg["max_arrival_residual"] = parser.getfloat(
        "algorithm",
        "max_arrival_residual",
        fallback=0.9
    )
    _cfg["max_event_residual"] = parser.getfloat(
        "algorithm",
        "max_event_residual",
        fallback=1.3
    )
    _cfg["max_dlat"] = parser.getfloat(
        "algorithm",
        "max_dlat",
        fallback=0.2
    )
    _cfg["max_dlon"] = parser.getfloat(
        "algorithm",
        "max_dlon",
        fallback=0.2
    )
    _cfg["max_ddepth"] = parser.getfloat(
        "algorithm",
        "max_ddepth",
        fallback=50
    )
    _cfg["max_dtime"] = parser.getfloat(
        "algorithm",
        "max_dtime",
        fallback=1
    )
    _cfg["max_lat"] = parser.getfloat(
        "algorithm",
        "max_lat",
        fallback=91
    )
    _cfg["max_lon"] = parser.getfloat(
        "algorithm",
        "max_lon",
        fallback=361
    )
    _cfg["min_lat"] = parser.getfloat(
        "algorithm",
        "min_lat",
        fallback=-91
    )
    _cfg["min_lon"] = parser.getfloat(
        "algorithm",
        "min_lon",
        fallback=-361
    )
    _cfg["min_depth"] = parser.getfloat(
        "algorithm",
        "min_depth",
        fallback=-999
    )
    _cfg["max_depth"] = parser.getfloat(
        "algorithm",
        "max_depth",
        fallback=9999
    )
    _cfg["damp"] = parser.getfloat(
        "algorithm",
        "damp"
    )
    _cfg["atol"] = parser.getfloat(
        "algorithm",
        "atol"
    )
    _cfg["btol"] = parser.getfloat(
        "algorithm",
        "btol"
    )
    _cfg["conlim"] = parser.getint(
        "algorithm",
        "conlim"
    )
    _cfg["maxiter"] = parser.getint(
        "algorithm",
        "maxiter",
        fallback=7
    )
    cfg["algorithm"] = _cfg

    _cfg = dict()

    output_label = parser.get(
        "model",
        "output_label",
        fallback='output'
    )
    output_label=output_label+f"_{stamp}"

    output_dir = parser.get(
        "model",
        "output_dir",
        fallback=output_label
    )
    output_dir = os.path.abspath(output_dir)
    _cfg["output_dir"] = output_dir

    log_file = parser.get(
        "model",
        "log_file",
        fallback='pyvorotomo.log'
    )
    _cfg["log_file"] = os.path.join(output_dir,log_file)

    scratch_dir = parser.get(
        "model",
        "scratch_dir",
        fallback=os.path.join(output_dir,"scratch")
    )
    scratch_dir = os.path.abspath(scratch_dir)
    _cfg["scratch_dir"] = scratch_dir

    if RANK == _constants.ROOT_RANK:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(scratch_dir, exist_ok=True)


    stations_path = parser.get(
        "model",
        "stations_path"
    )
    stations_path = os.path.abspath(stations_path)
    _cfg["stations_path"] = stations_path

    events_path = parser.get(
        "model",
        "events_path"
    )
    events_path = os.path.abspath(events_path)
    _cfg["events_path"] = events_path    

    initial_pwave_path = parser.get(
        "model",
        "initial_pwave_path"
    )
    initial_pwave_path = os.path.abspath(initial_pwave_path)
    _cfg["initial_pwave_path"] = initial_pwave_path

    initial_swave_path = parser.get(
        "model",
        "initial_swave_path"
    )
    initial_swave_path = os.path.abspath(initial_swave_path)
    _cfg["initial_swave_path"] = initial_swave_path

    cfg["model"] = _cfg

    map_filter_string = parser.get(
        "model",
        "map_filter",
        fallback=''
    )
    _cfg["map_filter"] = [float(x.strip()) for x in map_filter_string.split(",")] if map_filter_string else ''


    perform_res_test = parser.get(
        "model",
        "perform_res_test"
    )
    _cfg["perform_res_test"] = parser.getboolean(
        "model",
        "perform_res_test",
        fallback=False
    )

    res_test_string = parser.get(
        "model",
        "res_test_size_mag",
        fallback='100,0.08'
    )
    _cfg["res_test_size_mag"] = [float(x.strip()) for x in res_test_string.split(",")]

    res_test_layers_string = parser.get(
        "model",
        "res_test_layers",
        fallback="10,25,50,70,120,170,230"
    )
    _cfg["res_test_layers"] = (
        [float(x.strip()) for x in res_test_layers_string.split(",")]
    )

    rerun_restest = parser.get(
        "model",
        "rerun_restest",
        fallback=''
    )
    if rerun_restest.strip():
        _cfg["rerun_restest"] = os.path.abspath(rerun_restest)
    else:
        _cfg["rerun_restest"] = ''

    _cfg = dict()
    _cfg["method"] = parser.get(
        "relocate",
        "method"
    ).upper()

    if _cfg["method"] == "LINEAR":
        _cfg["atol"] = parser.getfloat(
            "linearized_relocation",
            "atol"
        )
        _cfg["btol"] = parser.getfloat(
            "linearized_relocation",
            "btol"
        )
        _cfg["maxiter"] = parser.getint(
            "linearized_relocation",
            "maxiter"
        )
        _cfg["conlim"] = parser.getint(
            "linearized_relocation",
            "conlim"
        )
        _cfg["damp"] = parser.getfloat(
            "linearized_relocation",
            "damp"
        )
    elif _cfg["method"].upper() == "DE":
        _cfg["depth_min"] = parser.getfloat(
            "de_relocation",
            "depth_min"
        )
        _cfg["dlat"] = parser.getfloat(
            "de_relocation",
            "dlat"
        )
        _cfg["dlon"] = parser.getfloat(
            "de_relocation",
            "dlon"
        )
        _cfg["ddepth"] = parser.getfloat(
            "de_relocation",
            "ddepth"
        )
        _cfg["dtime"] = parser.getfloat(
            "de_relocation",
            "dtime"
        )
    else:
        raise (
            ValueError(
                "Relocation method must be either \"linear\" or \"DE\"."
            )
        )
    cfg["relocate"] = _cfg

    return cfg


def write_cfg(argc, cfg):
    """
    Write the execution configuration to disk for later reference.
    """
    
    #output_dir = argc.output_dir
    output_dir = cfg['model']['output_dir']

    parser = configparser.ConfigParser()
    argc = vars(argc)
    argc = {key: str(argc[key]) for key in argc}
    cfg["argc"] = argc
    parser.read_dict(cfg)
    path = os.path.join(output_dir, "pyvorotomo.cfg")
    with open(path, "w") as configuration_file:
        parser.write(configuration_file)

    return True


############ utilitiy functions used (and not used) elsewhere

def dist_deg(lat1, lon1, lat2, lon2):
    """
    Vectorized calculation of spherical distance in DEGREES.
    Works with both single values and arrays.
    """
    # Convert inputs to arrays for vectorization
    lat1, lon1, lat2, lon2 = map(np.asarray, (lat1, lon1, lat2, lon2))

    # Convert to radians
    phi1 = lat1 * _constants.DEG_TO_RAD
    phi2 = lat2 * _constants.DEG_TO_RAD

    # Pre-compute trigonometric functions
    cos_phi1 = np.cos(phi1)
    cos_phi2 = np.cos(phi2)

    dlon = (lon2 - lon1) * _constants.DEG_TO_RAD
    dlat = (lat2 - lat1) * _constants.DEG_TO_RAD

    # Use sine squared directly
    sin_dlat_2 = np.sin(0.5 * dlat)
    sin_dlon_2 = np.sin(0.5 * dlon)

    # Optimized haversine
    a = sin_dlat_2 * sin_dlat_2 + cos_phi1 * cos_phi2 * sin_dlon_2 * sin_dlon_2
    a = np.minimum(a, 1.0)  # ensure a doesn't exceed 1 due to floating point errors

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return c * _constants.RAD_TO_DEG


def dist_km(lat1, lon1, lat2, lon2):
    """
    Vectorized calculation of distance in KILOMETERS.
    Works with both single values and arrays.
    """
    lat1, lon1, lat2, lon2 = map(np.asarray, (lat1, lon1, lat2, lon2))

    # Convert to radians
    phi1 = lat1 * _constants.DEG_TO_RAD
    phi2 = lat2 * _constants.DEG_TO_RAD

    # Pre-compute trigonometric functions
    cos_phi1 = np.cos(phi1)
    cos_phi2 = np.cos(phi2)

    dlon = (lon2 - lon1) * _constants.DEG_TO_RAD
    dlat = (lat2 - lat1) * _constants.DEG_TO_RAD

    # Use sine squared directly
    sin_dlat_2 = np.sin(0.5 * dlat)
    sin_dlon_2 = np.sin(0.5 * dlon)

    # Optimized haversine
    a = sin_dlat_2 * sin_dlat_2 + cos_phi1 * cos_phi2 * sin_dlon_2 * sin_dlon_2
    a = np.minimum(a, 1.0)  # ensure a doesn't exceed 1 due to floating point errors

    return _constants.EARTH_RADIUS * 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def remove_outliers(dataframe, tukey_k, column, max_resid=None):
    """
    Return DataFrame with outliers removed using Tukey fences.
    ALSO remove any arrival or event beyond maxresid (first)
    Note that "column" is always "residual" in our case
    """

    # Toss max residuals for both arrivals and events
    if max_resid:
        dataframe = dataframe[
             (dataframe[column] <= max_resid)
            &(dataframe[column] >= -max_resid)]

    # Do not Tukey the events
    if tukey_k and 'phase' not in dataframe.keys():
        q1, q3 = dataframe[column].quantile(q=[0.25, 0.75])
        iqr = q3 - q1
        vmin = q1 - tukey_k * iqr
        vmax = q3 + tukey_k * iqr
        dataframe = dataframe[
             (dataframe[column] >= vmin)
            &(dataframe[column] <= vmax)]

    return dataframe


def station_dict(dataframe):
    """
    Return a dictionary with network geometry suitable for passing to
    the EQLocator constructor.

    Returned dictionary has "station_id" keys, where "station_id" =
    f"{network}.{station}", and values are spherical coordinates of
    station locations.
    """

    if np.any(dataframe[["network", "station"]].duplicated()):
        raise IOError("Multiple coordinates supplied for single station(s)")

    dataframe = dataframe.set_index(["network", "station"])

    _station_dict = {
        (network, station): geo2sph(
            dataframe.loc[
                (network, station),
                ["latitude", "longitude", "depth"]
            ].values
        ) for network, station in dataframe.index
    }

    return _station_dict


def arrival_dict(dataframe, event_id):
    """
    Return a dictionary with phase-arrival data suitable for passing to
    the EQLocator.add_arrivals() method.

    Returned dictionary has ("station_id", "phase") keys, where
    "station_id" = f"{network}.{station}", and values are
    phase-arrival timestamps.
    """

    dataframe = dataframe.set_index("event_id")
    fields = ["network", "station", "phase", "time"]
    dataframe = dataframe.loc[event_id, fields]

    # If dataframe has only 1 item, it is converted to a Series
    #  this ensures it remains a DataFrame
    if not isinstance(dataframe,pd.DataFrame):
        dataframe = dataframe.to_frame().T

    # Failsafe against weirdness or if stations have their start/end times set incorrectly
    #  need to revisit first <=1 part, unclear if that ever happens normally
    if len(dataframe) <= 1:
        _arrival_dict = {} if len(dataframe) == 0 else {
        (dataframe.iloc[0, 0], dataframe.iloc[0, 1],
         dataframe.iloc[0, 2]): dataframe.iloc[0, 3]
        }
    else:
        try:
            _arrival_dict = {
                (network, station, phase): timestamp
                for network, station, phase, timestamp in dataframe.values
            }
        except:
            print("issue with setting arrival dict event_id=", event_id)
            print(dataframe.values)
            _arrival_dict = {}

    return _arrival_dict


# not used 
def fibonacci(n):
    """ Return the n-th number in the Fibonacci sequence """
    return pow(2 << n, n+1, (4 << 2 * n) - (2 << n)-1) % (2 << n)


# no longer in use
def eq_angle(eq_distkm,eq_depth):
    """
    Returns the angle in degrees from station to event.
    primarily to reduce shallow events with crustal reflections 
    but still allow deep teleseismic events through
    """
    theta = np.arctan2(eq_distkm, eq_depth)
    return 90 - np.abs(np.degrees(theta))


# not in use
def estimate_noise_from_residuals(residuals, method='mad'):
    """
    Estimate noise level from residual distribution.

    Parameters:
    -----------
    residuals : array
        Travel time residuals
    method : str
        'mad' - Median Absolute Deviation (robust)
        'std' - Standard deviation
        'iqr' - Interquartile range
    """
    if method == 'mad':
        median = np.median(residuals)
        mad = np.median(np.abs(residuals - median))
        return mad * 1.4826  # Convert to std equivalent

    elif method == 'std':
        # Remove outliers first (Tukey fence)
        q1, q3 = np.percentile(residuals, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (residuals >= lower) & (residuals <= upper)
        return np.std(residuals[mask])

    elif method == 'iqr':
        q1, q3 = np.percentile(residuals, [25, 75])
        return (q3 - q1) / 1.349  # convert std equivalent


# not in use but fun idea
from scipy.stats import gaussian_kde
def kde_stack(stack, bw_method='scott', return_uncertainty=False):
    """
    Find the mode (peak probability) of stack at each grid cell using KDE.

    Parameters:
    -----------
    stack : h5py Dataset or numpy array, shape (150, nz, ny, nx)
    bw_method : bandwidth selection ('scott', 'silverman', or float)
    return_uncertainty : if True, also return std or IQR as uncertainty measure

    Returns:
    --------
    delta_slowness : array of mode values at each cell
    uncertainty : (optional) uncertainty estimate at each cell
    """

    # Get shape - works for both h5py and numpy
    n_realizations = stack.shape[0]
    grid_shape = stack.shape[1:]
    n_cells = np.prod(grid_shape)

    delta_slowness_flat = np.zeros(n_cells)

    if return_uncertainty:
        uncertainty_flat = np.zeros(n_cells)

    # Iterate through spatial indices directly (avoid reshape with h5py)
    cell_idx = 0
    for iz in range(grid_shape[0]):
        for iy in range(grid_shape[1]):
            for ix in range(grid_shape[2]):
                # Extract all realizations at this spatial point
                cell_values = stack[:, iz, iy, ix]

                # Remove NaNs and invalid values
                valid_values = cell_values[~np.isnan(cell_values)]

                if len(valid_values) < 3:
                    delta_slowness_flat[cell_idx] = np.median(valid_values) if len(valid_values) > 0 else np.nan
                    if return_uncertainty:
                        uncertainty_flat[cell_idx] = 0
                    cell_idx += 1
                    continue

                try:
                    kde = gaussian_kde(valid_values, bw_method=bw_method)

                    v_min, v_max = valid_values.min(), valid_values.max()
                    v_range = np.linspace(v_min, v_max, 60) # cap at 60... maybe even lower is good
                    density = kde(v_range)

                    delta_slowness_flat[cell_idx] = v_range[np.argmax(density)]

                    if return_uncertainty:
                        uncertainty_flat[cell_idx] = np.percentile(valid_values, 75) - np.percentile(valid_values, 25)

                except (np.linalg.LinAlgError, ValueError):
                    delta_slowness_flat[cell_idx] = valid_values[0]
                    if return_uncertainty:
                        uncertainty_flat[cell_idx] = 0

                cell_idx += 1

    delta_slowness = delta_slowness_flat.reshape(grid_shape)

    if return_uncertainty:
        uncertainty = uncertainty_flat.reshape(grid_shape)
        return delta_slowness, uncertainty

    return delta_slowness
