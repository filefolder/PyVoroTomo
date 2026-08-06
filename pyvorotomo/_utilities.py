import argparse
import configparser
import logging
import mpi4py.MPI as MPI
import os
import signal
import time
import traceback
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
    """A utility function to handle interrupting signals"""
    try:
        shutdown_logging()
    except:
        pass
    COMM.Abort()


def configure_logger(name, log_file, verbose=False):
    """A utility function to configure logging. Return True with success."""

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
                logger.error(traceback.format_exc())
                raise exc

        return _decorated_func

    return _decorate_func


def shutdown_logging():
    """Close all logging handlers"""
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
    """Parse and return command line arguments"""

    parser = ArgumentParser()

    parser.add_argument(
        "-c",
        "--configuration_file",
        type=str,
        default=f"{parser.prog}.cfg",
        help="Configuration file."
    )
    parser.add_argument(
        "-r",
        "--relocate_first",
        action="store_true",
        help="Relocate events before first model update."
    )
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

    for attr in (
        #"events",
        #"network",
        #"log_file"
        #"output_dir"
        "configuration_file",):


        _attr = getattr(args, attr)
        _attr = os.path.abspath(_attr)
        setattr(args, attr, _attr)

    COMM.barrier()

    return args


def parse_cfg(configuration_file):
    """Parse and return contents of the configuration file"""

    cfg = dict()
    parser = configparser.ConfigParser()
    parser.read(configuration_file)

    # ALGORITHM section
    _cfg = dict()
    _cfg["niter"] = parser.getint("algorithm", "niter", fallback=4)
    _cfg["phase_order"] = [str(v).upper() for v in parser.get("algorithm", "phase_order", fallback='P,S').split(",")]
    _cfg["min_dist"] = parser.getfloat("algorithm", "min_dist", fallback=1)
    _cfg["max_dist"] = parser.getfloat("algorithm", "max_dist", fallback=155)
    raypath_bottom_mask_string = parser.get("algorithm", "raypath_bottom_mask", fallback="-1,-1")
    _cfg["raypath_bottom_mask"] = [float(x.strip()) for x in raypath_bottom_mask_string.split(",")]
    _cfg["nreal"] = parser.getint("algorithm", "nreal")
    _cfg["min_narrival"] = parser.getint("algorithm", "min_narrival", fallback=9)
    _cfg["narrival"] = parser.getint("algorithm", "narrival")
    _cfg["nevent"] = parser.getint("algorithm", "nevent")
    _cfg["narrival_percent"] = parser.getfloat("algorithm", "narrival_percent", fallback=-1)
    _cfg["nevent_percent"] = parser.getfloat("algorithm", "nevent_percent", fallback=-1)
    _cfg["outlier_removal_factor"] = parser.getfloat("algorithm", "outlier_removal_factor", fallback=1.5)
    _cfg["max_arrival_residual"] = parser.getfloat("algorithm", "max_arrival_residual", fallback=3.0)
    _cfg["max_event_residual"] = parser.getfloat("algorithm", "max_event_residual", fallback=2.0)
    _cfg["solver_weight_start"] = parser.getfloat("algorithm", "solver_weight_start", fallback=0.0)
    _cfg["solver_weight_end"] = parser.getfloat("algorithm", "solver_weight_end", fallback=0.8)
    _cfg["solver_weight_method"] = parser.get("algorithm", "solver_weight_method", fallback='huber').lower()
    _cfg["solver_weight_tuning"] = parser.getfloat("algorithm", "solver_weight_tuning", fallback=-1)
    _cfg["stack_type"] = parser.get("algorithm", "stack_type", fallback='mean')
    _cfg["stack_trim_percent"] = parser.getfloat("algorithm", "stack_trim_percent", fallback=20.0)
    _cfg["max_dlat"] = parser.getfloat("algorithm", "max_dlat", fallback=0.2)
    _cfg["max_dlon"] = parser.getfloat("algorithm", "max_dlon", fallback=0.2)
    _cfg["max_ddepth"] = parser.getfloat("algorithm", "max_ddepth", fallback=50)
    _cfg["max_dtime"] = parser.getfloat("algorithm", "max_dtime", fallback=1)
    _cfg["max_lat"] = parser.getfloat("algorithm", "max_lat", fallback=91)
    _cfg["max_lon"] = parser.getfloat("algorithm", "max_lon", fallback=361)
    _cfg["min_lat"] = parser.getfloat("algorithm", "min_lat", fallback=-91)
    _cfg["min_lon"] = parser.getfloat("algorithm", "min_lon", fallback=-361)
    _cfg["min_depth"] = parser.getfloat("algorithm", "min_depth", fallback=-999)
    _cfg["max_depth"] = parser.getfloat("algorithm", "max_depth", fallback=9999)
    _cfg["damp"] = parser.getfloat("algorithm", "damp",fallback=-1)
    _cfg["sigma_station"] = parser.getfloat("algorithm", "sigma_station",fallback=0.001)
    _cfg["atol"] = parser.getfloat("algorithm", "atol",fallback=1e-5)
    _cfg["btol"] = parser.getfloat("algorithm", "btol",fallback=0.01)
    _cfg["conlim"] = parser.getfloat("algorithm", "conlim",fallback=1000)
    _cfg["maxiter"] = parser.getint("algorithm", "maxiter", fallback=7)
    _cfg["tt_crop_km"] = parser.getfloat("algorithm", "tt_crop_km", fallback=-1)
    # Adaptive early stopping of realizations: after each realization
    # (once >= stack_convergence_min_nreal are done, checked every
    # stack_convergence_check_every), the running stacked model (same
    # trim/mean/median statistic as update_model) is compared to the
    # previous check; if the relative RMS change stays below
    # stack_convergence_tol for stack_convergence_patience consecutive
    # checks, remaining realizations are skipped. 0 disables (default).
    # A tolerance of ~0.02 (2% change per check block) is a sane start.
    _cfg["stack_convergence_tol"] = parser.getfloat(
        "algorithm", "stack_convergence_tol", fallback=0)
    _cfg["stack_convergence_min_nreal"] = parser.getint(
        "algorithm", "stack_convergence_min_nreal", fallback=30)
    _cfg["stack_convergence_check_every"] = parser.getint(
        "algorithm", "stack_convergence_check_every", fallback=5)
    _cfg["stack_convergence_patience"] = parser.getint(
        "algorithm", "stack_convergence_patience", fallback=3)
    # mask nodes beyond the radius with NaN inside the cropped box.
    # Default False for PyVoroTomo: ray tracing descends the traveltime
    # gradient and must never encounter NaN nodes near the crop edge.
    _cfg["tt_crop_mask"] = parser.getboolean("algorithm", "tt_crop_mask", fallback=False)
    cfg["algorithm"] = _cfg

    # MESHING section
    _cfg = dict()
    _cfg["adaptive_data_weight"] = parser.getfloat("meshing", "adaptive_data_weight", fallback=0.6)
    _cfg["density_to_gradient_weight"] = parser.getfloat("meshing", "density_to_gradient_weight", fallback=0.5)
    _cfg["hvr"] = parser.getfloat("meshing", "hvr", fallback=3)
    _cfg["target_rays_per_cell"] = parser.getint("meshing", "target_rays_per_cell", fallback=25)
    _cfg["min_cell_width_km"] = parser.getfloat("meshing", "min_cell_width_km", fallback=20)
    _cfg["max_cell_width_km"] = parser.getfloat("meshing", "max_cell_width_km", fallback=150)
    _cfg["enable_backfill"] = parser.getboolean("meshing", "enable_backfill", fallback=True)
    _cfg["min_rays_per_cell"] = parser.getint("meshing", "min_rays_per_cell", fallback=10)
    cfg["meshing"] = _cfg

    # ANALYZE section
    _cfg = dict()
    _cfg["pick_start_iter"] = parser.getint("analyze", "pick_start_iter", fallback=3)
    _cfg["pick_min_iters_present"] = parser.getint("analyze", "pick_min_iters_present", fallback=3)
    _cfg["pick_drop"] = parser.getboolean("analyze", "pick_drop", fallback=False)
    _cfg["pick_median_threshold"] = parser.getfloat("analyze", "pick_median_threshold", fallback=0.5)
    _cfg["pick_mad_max"] = parser.getfloat("analyze", "pick_mad_max", fallback=0.5)
    _cfg["pick_station_excess"] = parser.getfloat("analyze", "pick_station_excess", fallback=2.5)
    _cfg["pick_scale_k"] = parser.getfloat("analyze", "pick_scale_k", fallback=3.5)
    _cfg["pick_max_drop_fraction"] = parser.getfloat("analyze", "pick_max_drop_fraction", fallback=0.10)
    _cfg["event_residual_threshold"] = parser.getfloat("analyze", "event_residual_threshold", fallback=0.6)
    _cfg["event_weight_threshold"] = parser.getfloat("analyze", "event_weight_threshold", fallback=0.6)
    _cfg["event_std_threshold"] = parser.getfloat("analyze", "event_std_threshold", fallback=0.5)
    _cfg["station_residual_threshold"] = parser.getfloat("analyze", "station_residual_threshold", fallback=0.5)
    _cfg["station_std_threshold"] = parser.getfloat("analyze", "station_std_threshold", fallback=0.5)
    cfg["analyze"] = _cfg

    # MODEL section
    _cfg = dict()
    output_label = parser.get("model", "output_label", fallback='output')
    output_label = output_label + f"_{stamp}"

    output_dir = parser.get("model", "output_dir", fallback=output_label)
    output_dir = os.path.abspath(output_dir)
    _cfg["output_dir"] = output_dir

    log_file = parser.get("model", "log_file", fallback='pyvorotomo.log')
    _cfg["log_file"] = os.path.join(output_dir, log_file)

    scratch_dir = parser.get("model", "scratch_dir", fallback=os.path.join(output_dir, "scratch"))
    scratch_dir = os.path.abspath(scratch_dir)
    _cfg["scratch_dir"] = scratch_dir

    if RANK == _constants.ROOT_RANK:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(scratch_dir, exist_ok=True)

    stations_path = parser.get("model", "stations_path")
    stations_path = os.path.abspath(stations_path)
    _cfg["stations_path"] = stations_path

    events_path = parser.get("model", "events_path")
    events_path = os.path.abspath(events_path)
    _cfg["events_path"] = events_path

    initial_pwave_path = parser.get("model", "initial_pwave_path")
    initial_pwave_path = os.path.abspath(initial_pwave_path)
    _cfg["initial_pwave_path"] = initial_pwave_path

    initial_swave_path = parser.get("model", "initial_swave_path")
    initial_swave_path = os.path.abspath(initial_swave_path)
    _cfg["initial_swave_path"] = initial_swave_path
 
    map_filter_string = parser.get("model", "map_filter", fallback='')
    _cfg["map_filter"] = [float(x.strip()) for x in map_filter_string.split(",")] if map_filter_string else '' # note HAS to be an empty string, not None
 

    truepicks_path = parser.get("model", "truepicks_path", fallback='')
    _cfg["truepicks_path"] = os.path.abspath(truepicks_path) if truepicks_path.strip() else ''
    _cfg["always_include_truepicks"] = parser.getboolean("algorithm", "always_include_truepicks", fallback=True)

    _cfg["output_1d_model"] = parser.getboolean("model", "output_1d_model", fallback=True)
    _cfg["perform_res_test"] = parser.getboolean("model", "perform_res_test", fallback=False)
    res_test_string = parser.get("model", "res_test_size_mag", fallback='100,0.08')
    _cfg["res_test_size_mag"] = [float(x.strip()) for x in res_test_string.split(",")]
 
    res_test_layers_string = parser.get("model", "res_test_layers", fallback="10,25,50,70,120,170,230")
    _cfg["res_test_layers"] = [float(x.strip()) for x in res_test_layers_string.split(",")]
    rerun_restest = parser.get("model", "rerun_restest", fallback='')
    if rerun_restest.strip():
        _cfg["rerun_restest"] = os.path.abspath(rerun_restest)
    else:
        _cfg["rerun_restest"] = ''
    cfg["model"] = _cfg
 
    # RELOCATION section
    _cfg = dict()
    _cfg["depth_min"] = parser.getfloat("relocation", "depth_min", fallback=-99)
    _cfg["dlat"] = parser.getfloat("relocation", "dlat", fallback=0.3)
    _cfg["dlon"] = parser.getfloat("relocation", "dlon", fallback=0.3)
    _cfg["ddepth"] = parser.getfloat("relocation", "ddepth", fallback=50)
    _cfg["dtime"] = parser.getfloat("relocation", "dtime", fallback=3)
    _cfg["pick_uncert"] = parser.getfloat("relocation", "pick_uncert", fallback=0.02)
    _cfg["tt_error"] = parser.getfloat("relocation", "tt_error", fallback=0.015)
    # "edt" = NLL-style Equal Differential Time objective (robust to outlier
    # picks, origin time decoupled); "l1" = legacy weighted-L1 joint search.
    _cfg["method"] = parser.get("relocation", "method", fallback="edt").lower()
    # pick error assigned to arrivals manually flagged in truepicks_path (seconds);
    # near-zero => these picks dominate the relocation likelihood
    _cfg["truepick_error"] = parser.getfloat("relocation", "truepick_error", fallback=0.001) # not in cfg / TODO
    cfg["relocation"] = _cfg
 
    return cfg



def write_cfg(argc, cfg):
    """
    Write the execution configuration to disk for later reference.
    """
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

    Rows flagged truepick=True are NEVER removed: they are protected from
    both the max_resid cut and the Tukey fences (fences are still computed
    from the full distribution). Rescued rows are logged.
    """

    protected = None
    if "truepick" in dataframe.columns:
        is_true = dataframe["truepick"].fillna(False).astype(bool)
        if is_true.any():
            protected = dataframe[is_true]
            dataframe = dataframe[~is_true]

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

    if protected is not None:
        would_cut = 0
        if max_resid:
            would_cut = int((protected[column].abs() > max_resid).sum())
        if would_cut > 0:
            logger = get_logger(f"__main__.{__name__}")
            logger.info(
                f"remove_outliers: protected {len(protected)} truepicks "
                f"({would_cut} would otherwise have been culled)   ###"
            )
        dataframe = pd.concat([dataframe, protected])

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


def pick_error_dict(dataframe, event_id, default_error, truepick_error):
    """
    Return {(network, station, phase): pick_error_seconds} for one event,
    for EQLocator.add_pick_errors(). Arrivals flagged truepick=True get
    truepick_error (near-zero => they dominate the EDT likelihood);
    everything else gets default_error.
    """

    dataframe = dataframe.set_index("event_id")
    if "truepick" not in dataframe.columns:
        dataframe = dataframe.assign(truepick=False)
    fields = ["network", "station", "phase", "truepick"]
    dataframe = dataframe.loc[[event_id], fields]

    return {
        (network, station, phase):
            truepick_error if bool(is_true) else default_error
        for network, station, phase, is_true in dataframe.values
    }


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

    try:
        dataframe = dataframe.loc[event_id, fields]
    except Exception as e:
        logger = get_logger(f"__main__.{__name__}")
        logger.warning(f"arrival_dict could not access event_id: {event_id}."
            f"Error: {e}"
            " This should not happen!!")
        _arrival_dict = {}

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


def compute_residual_weights(residuals, method="huber", scale=None, tuning_param=-1):
    """
    Compute solver weights based on arrival residuals to down-weight outliers.

    Args:
        residuals: Array of traveltime residuals.
        method: Weighting method:
            - "huber"    : convex, never zero; w = 1 for |u|<=k, w = k/|u| beyond.
                           Best for fat-tailed but not catastrophic data.
                           For a +2s bad pick, weighs to 0.27 (SOFT)
            - "cauchy"   : convex, never zero; w = 1/(1 + (u/k)^2).
                           Smooth alternative to Huber, stronger tail attenuation.
                           For a +2s bad pick, weights to 0.18 (MEDIUM)
            - "bisquare" : redescending; w = (1 - (u/k)^2)^2 for |u|<=k, w = 0 beyond.
                           Best for data with gross outliers (e.g. bad picks).
                           For a +2s bad pick, weighs to 0.02 (HARD)

        scale: Robust scale estimate. If None, uses MAD x 1.4826.
        tuning_param: Method-specific. If <= 0, uses standard defaults:
            huber 1.345, bisquare 4.685, cauchy 2.385, soft_l1 1.0,
            linear 80 (percentile).

    Returns:
        weights: Array of weights in [0, 1], where 1 = full weight.
    """
    residuals = np.asarray(residuals)
    if scale is None:
        mad = np.median(np.abs(residuals - np.median(residuals)))
        scale = max(mad * 1.4826, 1e-6)
    abs_u = np.abs(residuals / scale)
    abs_u = np.maximum(abs_u, 1e-9)

    defaults = {"huber": 1.345, "bisquare": 4.685, "cauchy": 2.385}
    k = tuning_param if tuning_param > 0 else defaults[method]

    if method == "huber":
        weights = np.where(abs_u <= k, 1.0, k / abs_u)
    elif method == "cauchy":
        weights = 1.0 / (1.0 + (abs_u / k)**2)
    elif method == "bisquare":
        # Tukey's bisquare (biweight): redescending, hits zero at |u| = k
        u_over_k = abs_u / k
        weights = np.where(abs_u <= k, (1.0 - u_over_k**2)**2, 0.0)
    else:
        raise ValueError(f"Unknown method: {method}. Use: huber, cauchy, or bisquare")

    return weights


def blend_weights(weights, blend_factor):
    """
    Args:
        weights: weights from compute_residual_weights
        blend_factor: 0 = uniform weights, 1 = full weighting

    Returns blended weights from 0 to 1
    """
    blend_factor = np.clip(blend_factor, 0.0, 1.0)
    return (1 - blend_factor) + blend_factor * weights


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


import warnings
def stack_modal(stack, trim=0.25, sep=1.5, minor_frac=0.25, min_valid=3):
    """Gated-bimodal trimmed-mean stack over a masked (NUM, nz, ny, nx) array.
    Per cell: a symmetric trimmed mean of the valid realizations by default;
    the trimmed mean of the dominant mode only where the cell is clearly bimodal.
    NaN/masked-safe; cells with < min_valid valid samples are masked.
    Returns a masked array (parity with the np.ma.mean/median branches)."""
    raw = np.ma.filled(np.ma.asarray(stack).astype(float), np.nan)
    NUM, nz = raw.shape[0], raw.shape[1]
    out = np.full(raw.shape[1:], np.nan)
    if NUM == 0:
        return np.ma.masked_invalid(out)

    def take(arr, idx):                         # gather (L,ny,nx) at per-cell idx (ny,nx)
        return np.take_along_axis(arr, idx[None], axis=0)[0]

    def range_tmean(a, b, P):                   # trimmed mean of sorted positions [a, b)
        nd = b - a
        kd = np.floor(nd * trim).astype(np.intp)
        kd = np.where((kd > 0) & (nd > 2 * kd), kd, 0)
        return (take(P, b - kd) - take(P, a + kd)) / np.maximum(nd - 2 * kd, 1)

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for iz in range(nz):
            sl = raw[:, iz]                                  # (NUM, ny, nx)
            finite = np.isfinite(sl)
            m = finite.sum(axis=0).astype(np.intp)           # valid count per cell

            s = np.sort(np.where(finite, sl, np.inf), axis=0)        # valid first, +inf last
            vmask = np.arange(NUM)[:, None, None] < m[None]
            sv = np.where(vmask, s, 0.0)

            zero = np.zeros((1,) + sl.shape[1:])
            P  = np.concatenate([zero, np.cumsum(sv,      axis=0)], axis=0)   # P[i]=sum first i
            P2 = np.concatenate([zero, np.cumsum(sv * sv, axis=0)], axis=0)
            total, total2 = P[NUM], P2[NUM]

            # default: trimmed mean over all valid values
            default = range_tmean(np.zeros_like(m), m, P)

            # best two-class split (Otsu) over valid positions
            t   = np.arange(1, NUM)[:, None, None]
            Pt  = P[1:NUM]                                    # = P[t]
            n0, n1 = t.astype(float), (m[None] - t).astype(float)
            between = n0 * n1 * (Pt / n0 - (total[None] - Pt) / n1) ** 2
            between[~(t <= (m[None] - 1))] = -np.inf          # only splits with both sides valid
            split = np.clip(np.argmax(between, axis=0).astype(np.intp) + 1,
                            1, np.maximum(m - 1, 1))

            # gate stats for lo = [0, split), hi = [split, m)
            n_lo, n_hi = split.astype(float), (m - split).astype(float)
            Ps, P2s = take(P, split), take(P2, split)
            mu_lo = Ps / np.maximum(n_lo, 1)
            mu_hi = (total - Ps) / np.maximum(n_hi, 1)
            spread = (np.sqrt(np.maximum(P2s / np.maximum(n_lo, 1) - mu_lo ** 2, 0)) +
                      np.sqrt(np.maximum((total2 - P2s) / np.maximum(n_hi, 1) - mu_hi ** 2, 0)))
            minor = np.minimum(n_lo, n_hi) / np.maximum(m, 1)
            bimodal = ((m >= 8) & (minor >= minor_frac) &
                       (spread > 0) & ((mu_hi - mu_lo) >= sep * spread))

            # dominant-cluster trimmed mean
            lo_big = n_lo >= n_hi
            a = np.where(lo_big, 0, split).astype(np.intp)
            b = np.where(lo_big, split, m).astype(np.intp)

            dom = range_tmean(a, b, P)

            cell = np.where(bimodal, dom, default)
            cell[m < min_valid] = np.nan
            cell[~np.isfinite(cell)] = np.nan
            out[iz] = cell

    return np.ma.masked_invalid(out)



# not in use and poorly implemented but fun idea
# it seems that nothing really beats a good trimmed mean
from scipy.stats import gaussian_kde
def kde_stack(stack, bw_method='scott', return_uncertainty=False):
    """
    Find the mode (peak probability) of stack at each grid cell using KDE.

    Parameters:
    -----------
    stack : h5py Dataset or numpy array, shape (NUM, nz, ny, nx)
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
                    v_range = np.linspace(v_min, v_max, n_realizations)
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
