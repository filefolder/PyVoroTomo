import numpy as np
import pandas as pd
import pykonal
import os

from . import _constants
from . import _picklabel
from . import _utilities

logger = _utilities.get_logger(f"__main__.{__name__}")

def parse_event_data(cfg):
    """
    Parse and return event data (origins and phases) specified in the
    config file.

    Data are returned as a two-tuple of pandas.DataFrame objects. The
    first entry is the origin data and the second is the phase data.

    The input file is expected to be a HDF5 file readable using
    pandas.HDFStore. The input file should have two tables: "events"
    and "arrivals".

    The "events" table needs to have "latitude",
    "longitude", "depth", "time", "event_id", and "source_id" columns.
    If source_id is not present, we make a generic one.

    The "arrivals" table needs to have "network", "station", "phase",
    "time", and "event_id" columns.
    """

    path = cfg["model"]["events_path"]
    if not os.path.isfile(path):
        raise RuntimeError(f'No catalog exists at {path}!')

    try:
        events   = pd.read_hdf(path, key="events")
        arrivals = pd.read_hdf(path, key="arrivals")
    except Exception as e:
        raise RuntimeError(
            f"Could not load event file '{path}'. "
            f"This may be due to an incompatible version of PyTables. "
            f"Original error: {e}"
        ) from e

    if 'arrival_id' not in arrivals.keys():
        arrivals['arrival_id'] = range(len(arrivals))

    if 'source_id' not in events.keys():
        events['source_id'] = "event_" + events['event_id'].astype(str).str.zfill(6)

    for field in _constants.EVENT_FIELDS:
        if field not in events.columns:
            error = ValueError(
                f"Input event data must have the following fields: "
                f"{_constants.EVENT_FIELDS}"
            )
            raise error

    for field in _constants.ARRIVAL_FIELDS:
        if field not in arrivals.columns:
            error = ValueError(
                f"Input arrival data must have the following fields: "
                f"{_constants.ARRIVAL_FIELDS}"
            )
            raise error

    arrivals = flag_truepicks(cfg, arrivals)

    return events, arrivals


def flag_truepicks(cfg, arrivals):
    """
    Add a boolean "truepick" column to the arrivals DataFrame.

    If cfg["model"]["truepicks_path"] points to a CSV with columns
    (event_id | eq_id), network, station, phase, the matching arrivals
    are flagged True. These picks are treated as ground truth throughout
    the inversion: exempt from outlier culling, always included in
    arrival sampling, given maximum inversion weight, and assigned a
    near-zero pick uncertainty during relocation.
    """

    arrivals = arrivals.copy()
    arrivals["truepick"] = False

    path = cfg["model"].get("truepicks_path", "")
    if not path:
        return arrivals

    truepicks = pd.read_csv(path, skipinitialspace=True, comment="#")
    truepicks.columns = [c.strip().lower() for c in truepicks.columns]
    if "eq_id" in truepicks.columns and "event_id" not in truepicks.columns:
        truepicks = truepicks.rename(columns={"eq_id": "event_id"})

    required = ["event_id", "network", "station", "phase"]
    missing = [c for c in required if c not in truepicks.columns]
    if missing:
        raise ValueError(
            f"truepicks file '{path}' is missing columns {missing}; "
            f"expected (event_id|eq_id), network, station, phase."
        )

    # Normalize for matching
    truepicks = truepicks[required].copy()
    truepicks["event_id"] = truepicks["event_id"].astype(
        arrivals["event_id"].dtype
    )
    for col in ("network", "station", "phase"):
        truepicks[col] = truepicks[col].astype(str).str.strip()
        arrivals[col] = arrivals[col].astype(str).str.strip()
    truepicks["phase"] = truepicks["phase"].str.upper()

    truepicks = truepicks.drop_duplicates()

    key_cols = ["event_id", "network", "station", "phase"]
    arr_keys = pd.MultiIndex.from_frame(arrivals[key_cols])
    true_keys = pd.MultiIndex.from_frame(truepicks[key_cols])
    arrivals["truepick"] = arr_keys.isin(true_keys)

    n_flagged = int(arrivals["truepick"].sum())
    n_unmatched = int((~true_keys.isin(arr_keys)).sum())
    logger.info(
        f"truepicks: flagged {n_flagged} of {len(arrivals)} arrivals as "
        f"ground truth ({len(truepicks)} rows in {path})   ###"
    )
    if n_unmatched > 0:
        unmatched = truepicks[~true_keys.isin(arr_keys)]
        logger.warning(
            f"truepicks: {n_unmatched} rows matched NO arrival "
            f"(check ids/codes). First few:\n"
            f"{unmatched.head(5).to_string(index=False)}   ###"
        )
    if n_flagged == 0 and len(truepicks) > 0:
        logger.warning("truepicks: file provided but nothing matched!   ###")

    return arrivals

# TODO rename this to "stations" ?
def parse_network_geometry(cfg):
    """
    Parse and return network-geometry file specified in the
    config file.

    Data are returned as a pandas.DataFrame object.

    The input file is expected to be a HDF5 file readable using
    pandas.HDFStore. The input file needs to have one table: "stations"."
    The "stations" table needs to have "network", "station", "latitude",
    "longitude", and "elevation" fields. "latitude" and "longitude" are
    in degrees and "elevation" is in kilometers. The returned DataFrame
    has "network", "station", "latitude", "longitude", and "depth"
    columns.
    """

    path = cfg["model"]["stations_path"]
    if not os.path.isfile(path):
        raise RuntimeError(f'No stations exist at {path}!')

    network = pd.read_hdf(path, key="stations")
    network["depth"] = -network["elevation"] # TODO make this a bit more flexible
    network = network.drop(columns=["elevation"])

    return network


def parse_velocity_models(cfg):
    """
    Parse and return velocity models specified in configuration.

    Velocity models are returned as a two-tuple of
    _picklabel.ScalarField3D objects. The first entry is the P-wave and
    the second is the S-wave model.
    """

    pwave_model = _picklabel.ScalarField3D(coord_sys="spherical")
    swave_model = _picklabel.ScalarField3D(coord_sys="spherical")


    path = cfg["model"]["initial_pwave_path"]
    if not os.path.isfile(path):
        raise RuntimeError(f'No P model exists at {path}!')

    try:
        _pwave_model = pykonal.fields.read_hdf(path)
    except Exception as e:
        raise RuntimeError(
            f"Could not load P model file '{path}'"
            f"Original error: {e}"
        ) from e

    path = cfg["model"]["initial_swave_path"]
    if not os.path.isfile(path):    
        raise RuntimeError(f'No S model exists at {path}!')

    try:
        _swave_model = pykonal.fields.read_hdf(path)
    except Exception as e:
        raise RuntimeError(
            f"Could not load S model file '{path}'"
            f"Original error: {e}"
        ) from e


    models  = pwave_model, swave_model
    _models = _pwave_model, _swave_model
    for model, _model in zip(models, _models):
        model.min_coords = _model.min_coords
        model.node_intervals = _model.node_intervals
        model.npts = _model.npts
        model.values = _model.values

    return pwave_model, swave_model
