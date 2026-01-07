import numpy as np
import pandas as pd
import pykonal

from . import _constants
from . import _picklabel

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

    events = pd.read_hdf(path, key="events")
    arrivals = pd.read_hdf(path, key="arrivals")

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

    return events, arrivals

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
    _pwave_model = pykonal.fields.read_hdf(path)

    path = cfg["model"]["initial_swave_path"]
    _swave_model = pykonal.fields.read_hdf(path)

    models  = pwave_model, swave_model
    _models = _pwave_model, _swave_model
    for model, _model in zip(models, _models):
        model.min_coords = _model.min_coords
        model.node_intervals = _model.node_intervals
        model.npts = _model.npts
        model.values = _model.values

    return pwave_model, swave_model
