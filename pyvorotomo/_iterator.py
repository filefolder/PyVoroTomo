import glob
import h5py
import KDEpy as kp
import mpi4py.MPI as MPI
import numpy as np
import os
import pandas as pd
import pykonal
import shutil
import tempfile
import time

import scipy.sparse
from scipy.stats import iqr
from scipy.spatial import cKDTree

from . import _dataio
from . import _clustering
from . import _constants
from . import _picklabel
from . import _utilities
from . import _restesting

# Get logger handle.
logger = _utilities.get_logger(f"__main__.{__name__}")

# Define aliases.
TraveltimeInventory = pykonal.inventory.TraveltimeInventory
PointSourceSolver = pykonal.solver.PointSourceSolver
geo2sph = pykonal.transformations.geo2sph
sph2geo = pykonal.transformations.sph2geo
sph2xyz = pykonal.transformations.sph2xyz
xyz2sph = pykonal.transformations.xyz2sph

COMM       = MPI.COMM_WORLD
RANK       = COMM.Get_rank()
WORLD_SIZE = COMM.Get_size()
ROOT_RANK  = _constants.ROOT_RANK


class InversionIterator(object):
    """
    A class providing core functionality for iterating inversion
    procedure.
    """

    def __init__(self, argc):

        self._argc = argc
        self._arrivals = None
        self._arrivals_history = None
        self._cfg = None
        self._events = None
        self._iiter = 0
        self._ireal = 0
        self._phases = None
        self._projection_matrix = None
        self._pwave_model = None
        self._swave_model = None
        self._pwave_realization_stack = None
        self._swave_realization_stack = None
        self._pwave_variance = None
        self._swave_variance = None
        self._pwave_quality = None
        self._swave_quality = None
        self._pqual_realization_stack = None #new! try to assess and track quality of each inversion for weighting
        self._squal_realization_stack = None
        self._residuals = None
        self._sensitivity_matrix = None
        self._stations = None
        self._step_size = None
        self._sampled_arrivals = None
        self._sampled_events = None
        self._voronoi_cells = None
        self._model_lat_center = 0
        # self._max_variance_km_s = 0 # barometer to tell when enough iterations are enough / TODO not in use

        if RANK == ROOT_RANK:
            scratch_dir = argc.scratch_dir
            self._scratch_dir_obj = tempfile.TemporaryDirectory(dir=scratch_dir)
            self._scratch_dir = self._scratch_dir_obj.name

            _tempfile = tempfile.TemporaryFile(dir=argc.scratch_dir)
            self._f5_workspace = h5py.File(_tempfile, mode="w")

        self.synchronize(attrs=["scratch_dir"])

    def __del__(self):
        if RANK == ROOT_RANK:
            self._f5_workspace.close()
            shutil.rmtree(self.scratch_dir)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.__del__()

    @property
    def argc(self):
        return self._argc

    @property
    def arrivals(self):
        return self._arrivals

    @property
    def arrivals_history(self):
        return self._arrivals_history

    @arrivals_history.setter
    def arrivals_history(self, value):
        self._arrivals_history = value

    @arrivals.setter
    def arrivals(self, value):
        self._arrivals = value

    @property
    def cfg(self):
        return self._cfg

    @cfg.setter
    def cfg(self, value):
        self._cfg = value

    @property
    def events(self):
        return self._events

    @events.setter
    def events(self, value):
        value = value.sort_values("event_id")
        value = value.reset_index(drop=True)
        self._events = value

    @property
    def events_history(self):
        return self._events_history

    @events_history.setter
    def events_history(self, value):
        self._events_history = value

    @property
    def iiter(self):
        return self._iiter

    @iiter.setter
    def iiter(self, value):
        self._iiter = value

    @property
    def ireal(self):
        return self._ireal

    @ireal.setter
    def ireal(self, value):
        self._ireal = value

    @property
    def phases(self):
        return self._phases

    @phases.setter
    def phases(self, value):
        self._phases = value

    @property
    def projection_matrix(self):
        return self._projection_matrix

    @projection_matrix.setter
    def projection_matrix(self, value):
        self._projection_matrix = value

    @property
    def pqual_realization_stack(self):
        if RANK == ROOT_RANK:
            if "pqual_stack" not in self._f5_workspace:
                self._f5_workspace.create_dataset(
                    "pqual_stack",
                    shape=(self.cfg["algorithm"]["nreal"], *self.pwave_model.npts),
                    dtype=_constants.DTYPE_REAL,
                    fillvalue=np.nan,chunks=True
                )
            return self._f5_workspace["pqual_stack"]
        return None

    @property
    def squal_realization_stack(self):
        if RANK == ROOT_RANK:
            if "squal_stack" not in self._f5_workspace:
                self._f5_workspace.create_dataset(
                    "squal_stack",
                    shape=(self.cfg["algorithm"]["nreal"], *self.swave_model.npts),
                    dtype=_constants.DTYPE_REAL,
                    fillvalue=np.nan,chunks=True
                )
            return self._f5_workspace["squal_stack"]
        return None

    @property
    def pwave_model(self) -> _picklabel.ScalarField3D:
        return self._pwave_model

    @pwave_model.setter
    def pwave_model(self, value):
        self._pwave_model = value

    @property
    def pwave_quality(self) -> _picklabel.ScalarField3D:
        return self._pwave_quality

    @pwave_model.setter
    def pwave_quality(self, value):
        self._pwave_quality = value

    @property
    def pwave_realization_stack(self):
        if RANK == ROOT_RANK:
            if "pwave_stack" not in self._f5_workspace:
                self._f5_workspace.create_dataset(
                    "pwave_stack",
                    shape=(self.cfg["algorithm"]["nreal"], *self.pwave_model.npts),
                    dtype=_constants.DTYPE_REAL,
                    fillvalue=np.nan,chunks=True
                )
            return self._f5_workspace["pwave_stack"]
        return None

    @property
    def pwave_variance(self) -> _picklabel.ScalarField3D:
        field = _picklabel.ScalarField3D(coord_sys="spherical")
        field.min_coords = self.pwave_model.min_coords
        field.node_intervals = self.pwave_model.node_intervals
        field.npts = self.pwave_model.npts
        stack = self._f5_workspace["pwave_stack"]
        stack = np.ma.masked_invalid(stack)
        var = np.ma.var(stack, axis=0)
        field.values = var
        return field

    @property
    def raypath_dir(self):
        return os.path.join(self.scratch_dir, "raypaths")

    @property
    def residuals(self):
        return self._residuals

    @residuals.setter
    def residuals(self, value):
        self._residuals = value

    @property
    def sampled_arrivals(self):
        return self._sampled_arrivals

    @sampled_arrivals.setter
    def sampled_arrivals(self, value):
        self._sampled_arrivals = value

    @property
    def sampled_events(self):
        return self._sampled_events

    @sampled_events.setter
    def sampled_events(self, value):
        self._sampled_events = value

    @property
    def scratch_dir(self):
        return self._scratch_dir

    @scratch_dir.setter
    def scratch_dir(self, value):
        self._scratch_dir = value

    @property
    def sensitivity_matrix(self):
        return self._sensitivity_matrix

    @sensitivity_matrix.setter
    def sensitivity_matrix(self, value):
        self._sensitivity_matrix = value

    @property
    def stations(self):
        return self._stations

    @stations.setter
    def stations(self, value):
        self._stations = value

    @property
    def step_size(self):
        return self._step_size

    @step_size.setter
    def step_size(self, value):
        self._step_size = value

    @property
    def swave_model(self):
        return self._swave_model

    @swave_model.setter
    def swave_model(self, value):
        self._swave_model = value

    @property
    def swave_quality(self) -> _picklabel.ScalarField3D:
        return self._swave_quality

    @pwave_model.setter
    def swave_quality(self, value):
        self._swave_quality = value

    @property
    def swave_realization_stack(self):
        if RANK == ROOT_RANK:
            if "swave_stack" not in self._f5_workspace:
                self._f5_workspace.create_dataset(
                    "swave_stack",
                    shape=(self.cfg["algorithm"]["nreal"], *self.swave_model.npts),
                    dtype=_constants.DTYPE_REAL,
                    fillvalue=np.nan, chunks=True
                )
            return self._f5_workspace["swave_stack"]

        return None

    @property
    def swave_variance(self) -> _picklabel.ScalarField3D:
        field = _picklabel.ScalarField3D(coord_sys="spherical")
        field.min_coords = self.swave_model.min_coords
        field.node_intervals = self.swave_model.node_intervals
        field.npts = self.swave_model.npts
        stack = self._f5_workspace["swave_stack"]
        stack = np.ma.masked_invalid(stack)
        var = np.ma.var(stack, axis=0)
        field.values = var
        return field

    @property
    def traveltime_dir(self):
        return os.path.join(self.scratch_dir, "traveltimes")

    @property
    def traveltime_inventory_path(self):
        return os.path.join(self.scratch_dir, "traveltime_inventory.h5")

    @property
    def voronoi_cells(self):
        return self._voronoi_cells

    @voronoi_cells.setter
    def voronoi_cells(self, value):
        self._voronoi_cells = value

    @_utilities.log_errors(logger)
    @_utilities.root_only(RANK)
    def _compute_model_update(self, phase, min_rays=3):
        """
        Compute the model update for a single realization and appends
        the results to the realization stack.
        Only the root rank performs this operation.
        """
        logger.debug(f"Computing {phase}-wave model update")


        if phase == "P":
            model = self.pwave_model
        elif phase == "S":
            model = self.swave_model
        else:
            raise (ValueError(f"Unrecognized phase ({phase}) supplied."))
        
        atol = self.cfg["algorithm"]["atol"]
        btol = self.cfg["algorithm"]["btol"]
        conlim = self.cfg["algorithm"]["conlim"]
        maxiter = self.cfg["algorithm"]["maxiter"]
        nvoronoi = len(self.voronoi_cells)

        # Get ray counts for masking
        sensitivity_voronoi = self.sensitivity_matrix.tocsr()[:len(self.residuals), :nvoronoi]
        sensitivity_coo = sensitivity_voronoi.tocoo()
        ray_counts = np.bincount(sensitivity_coo.col, minlength=nvoronoi)
        valid_cells = (ray_counts >= min_rays)

        # Attempt adaptive damping if set to -1
        damp = self.cfg["algorithm"]["damp"]
        if damp < 0:

            base_damp = np.std(self.residuals.data) / np.median(np.abs(self.sensitivity_matrix.data)) 

            df = pd.DataFrame({'col': sensitivity_coo.col, 'data': np.abs(sensitivity_coo.data)})
            sensitivity_per_cell = df.groupby('col')['data'].median().reindex(range(nvoronoi), fill_value=0).values
            norm_sensitivity = sensitivity_per_cell / np.max(sensitivity_per_cell)
            norm_ray_count = ray_counts / np.max(ray_counts)
            cell_quality = norm_sensitivity/2 + norm_ray_count/2  # should go from 0-1

            if RANK == ROOT_RANK:
                percentiles = np.percentile(cell_quality, [0, 25, 50, 75, 90])
                logger.info(f"Cell quality distribution - "
                            f"0%: {percentiles[0]:.2f}, 25%: {percentiles[1]:.2f}, "
                            f"50%: {percentiles[2]:.2f}, 75%: {percentiles[3]:.2f}, 90%: {percentiles[4]:.2f}")

            # good = 50% reduction, bad = 100% increase
            damp = base_damp * (0.5 + 1.5 * (1 - cell_quality))

            # Infer dimensions from sensitivity matrix
            total_cols = self.sensitivity_matrix.shape[1]
            nstation = total_cols - nvoronoi

            # Create damping matrix for Voronoi cells only
            voronoi_damping = scipy.sparse.diags(damp, format='csr')
            # No damping for station parameters
            station_damping = scipy.sparse.diags(np.zeros(nstation), format='csr')

            # Combine into full damping matrix matching sensitivity matrix columns
            full_damping = scipy.sparse.block_diag([voronoi_damping, station_damping])

            # Augment the system: [G; D] * x = [d; 0]
            augmented_G = scipy.sparse.vstack([
                self.sensitivity_matrix, 
                full_damping
            ])
            augmented_d = np.concatenate([
                self.residuals, 
                np.zeros(nvoronoi + nstation)
            ])

            result = scipy.sparse.linalg.lsmr(
                augmented_G,
                augmented_d, 
                damp=0,
                atol=atol,
                btol=btol,
                conlim=conlim,
                maxiter=maxiter,
                show=False
            )
            x, istop, itn, normr, normar, norma, conda, normx = result

            damp = np.mean(damp) # just for logging to get a sense of scale

        # use the literal value given in cfg
        else:
            result = scipy.sparse.linalg.lsmr(
                self.sensitivity_matrix,
                self.residuals, 
                damp = damp,
                atol = atol,
                btol = btol,
                conlim = conlim,
                maxiter = maxiter,
                show=False
            )
            x, istop, itn, normr, normar, norma, conda, normx = result

        # set cells with insufficient rays to ZERO
        x[:nvoronoi][ray_counts < min_rays] = 0

        logger.info(f" ||G||         = {norma:8.2f}  (sensitivity matrix mag)")
        logger.info(f" ||Gm-d||      = {normr:8.2f}  (residual norm)")
        logger.info(f" ||m||         = {normx:8.3f}  (solution norm)")
        logger.info(f" ||G^-1||||G|| = {conda:8.2f}  (condition estimate)")
        logger.info(f" med(std) G,r  = {np.median(self.sensitivity_matrix.data):.3f}({np.std(self.sensitivity_matrix.data):.3f}), {np.median(self.residuals.data):.3f}({np.std(self.residuals.data):.3f})")
        logger.info(f" est/used damp = {np.std(self.residuals.data)/conda:.4e} / {damp:.4e}")
        logger.info(f" used/nvoronoi = {np.sum(ray_counts >= min_rays):.0f}/{nvoronoi:.0f}  ({itn} LSMR iterations)")

        #note that damping should be about std(residuals)/conda
        #btol should be our target resolution (or maybe median residual) / normr

        delta_slowness = self.projection_matrix * x[:nvoronoi]
        delta_slowness = delta_slowness.reshape(model.npts)

        # can we save inversion quality metrics to weight the stack also?

        ray_coverage = np.zeros(nvoronoi)
        ray_coverage[ray_counts >= min_rays] = ray_counts[ray_counts >= min_rays] / np.max(ray_counts)
        local_quality_coverage = self.projection_matrix * ray_coverage
        local_quality_coverage = local_quality_coverage.reshape(model.npts)


        # increases with decreasing residual norm
        quality_normr = 1.0 / (normr + 1e-8)
        # increases with decreasing condition number
        quality_conda = 1.0 / (conda + 1e-8)
        # raypath coverage ratio relative to number of cells
        quality_coverage = np.sum(ray_counts >= min_rays) / nvoronoi

        # Combine global factors into a single global quality score (you can adjust weights)
        global_quality = (quality_normr * 0.4 + 
                          quality_conda * 0.4 + 
                          quality_coverage * 0.2)

        # Add to our solution stack
        if phase == "P":
            self.pwave_realization_stack[self.ireal] = delta_slowness
            self.pqual_realization_stack[self.ireal] = global_quality # not sure this is working. seems to be the same as pwave?
        else:
            self.swave_realization_stack[self.ireal] = delta_slowness
            self.squal_realization_stack[self.ireal] = global_quality

        return True


    @_utilities.log_errors(logger)
    def _compute_sensitivity_matrix(self, phase, hvr):
        """
        Compute the sensitivity matrix.
        """

        logger.debug(f"Computing {phase}-wave sensitivity matrix")

        raypath_dir = self.raypath_dir

        index_keys = ["network", "station"]
        arrivals = self.sampled_arrivals.set_index(index_keys)

        arrivals = arrivals.sort_index()

        stationused = self.sampled_arrivals[index_keys]
        stationused = stationused.drop_duplicates().reset_index()
        stationused['idx'] = range(len(stationused))
        stationused = stationused.set_index(index_keys)
        nstation = stationused['idx'].max()+1


        if RANK == ROOT_RANK:

            nvoronoi = len(self.voronoi_cells)

            ids = arrivals.index.unique()
            self._dispatch(ids)

            logger.debug("Compiling sensitivity matrix")
            column_idxs = COMM.gather(None, root=ROOT_RANK)
            nsegments = COMM.gather(None, root=ROOT_RANK)
            nonzero_values = COMM.gather(None, root=ROOT_RANK)
            residuals = COMM.gather(None, root=ROOT_RANK)

            column_idxs = np.concatenate([x for x in column_idxs if x is not None])
            nonzero_values = np.concatenate([x for x in nonzero_values if x is not None])
            residuals = np.concatenate([x for x in residuals if x is not None])
            nsegments = np.concatenate([x for x in nsegments if x is not None])

            row_idxs = np.repeat(np.arange(len(nsegments)), nsegments)

            matrix = scipy.sparse.coo_matrix(
                (nonzero_values, (row_idxs, column_idxs)),
                shape=(len(nsegments), nvoronoi+nstation)
            )

            if matrix.nnz < 1:
                logger.warning("*** G matrix is ~empty! Diagnostics:")
                logger.warning(f"    column_idxs length: {len(column_idxs)}")
                logger.warning(f"    nonzero_values length: {len(nonzero_values)}")
                logger.warning(f"    row_idxs length: {len(row_idxs)}")

            self.sensitivity_matrix = matrix
            self.residuals = residuals

        else:

            nvoronoi = len(self.voronoi_cells)
            column_idxs = np.array([], dtype=_constants.DTYPE_INT)
            nsegments = np.array([], dtype=_constants.DTYPE_INT)
            nonzero_values = np.array([], dtype=_constants.DTYPE_REAL)
            residuals = np.array([], dtype=_constants.DTYPE_REAL)

            step_size = self.step_size
            events = self.events.set_index("event_id")
            events["idx"] = range(len(events))

            while True:
                item = self._request_dispatch()

                if item is None:
                    logger.debug("Sentinel received. Gathering sensitivity matrix.")

                    column_idxs = COMM.gather(column_idxs, root=ROOT_RANK)
                    nsegments = COMM.gather(nsegments, root=ROOT_RANK)
                    nonzero_values = COMM.gather(nonzero_values, root=ROOT_RANK)
                    residuals = COMM.gather(residuals, root=ROOT_RANK)

                    break

                network, station = item

                # Get the subset of arrivals belonging to this station.
                _arrivals = arrivals.loc[[(network, station)]]
                _arrivals = _arrivals.set_index("event_id")

                station_idxs = stationused['idx'].loc[[(network,station)]]+nvoronoi

                # Open the raypath file.
                filename = f"{network}.{station}.{phase}.h5"
                path = os.path.join(raypath_dir, filename)
                #raypath_file = h5py.File(path, mode="r")
                with h5py.File(path, mode="r") as raypath_file:
                    for event_id, arrival in _arrivals.iterrows():

                        event = events.loc[event_id]
                        idx = int(event["idx"])

                        raypath = raypath_file[phase][:, idx]
                        raypath = np.stack(raypath).T

                        if len(raypath) < 1:
                            logger.warning("raypath is 0??")

                        _column_idxs, counts = self._projected_ray_idxs(raypath,hvr) # raypath HVR scaling done in this function
                        _column_idxs = np.append(_column_idxs,station_idxs)
                        column_idxs = np.append(column_idxs, _column_idxs)
                        nsegments = np.append(nsegments, len(_column_idxs))
                        _nonzero_values = counts * step_size
                        _nonzero_values = np.append(_nonzero_values,1)
                        nonzero_values = np.append(nonzero_values, _nonzero_values)
                        residuals = np.append(residuals, arrival["residual"])      

        COMM.barrier()

        return True


    @_utilities.log_errors(logger)
    def _dispatch(self, ids, sentinel=None):
        """
        Dispatch ids to hungry workers, then dispatch sentinels.
        """

        logger.debug(f"_dispatch called with {len(list(ids))} items")

        for _id in ids:
            requesting_rank = COMM.recv(
                source=MPI.ANY_SOURCE,
                tag=_constants.DISPATCH_REQUEST_TAG
            )
            COMM.send(
                _id,
                dest=requesting_rank,
                tag=_constants.DISPATCH_TRANSMISSION_TAG
            )

        logger.debug("Distribute sentinel")
        for irank in range(WORLD_SIZE - 1):
            requesting_rank = COMM.recv(
                source=MPI.ANY_SOURCE,
                tag=_constants.DISPATCH_REQUEST_TAG
            )
            COMM.send(
                sentinel,
                dest=requesting_rank,
                tag=_constants.DISPATCH_TRANSMISSION_TAG
            )

        return True


    def _estimate_voronoi_cell_widths_simple(self, voronoi_cells):
        """
        Fast estimate of average horizontal cell width using KDTree.
        """
        # Convert to geographic
        cells_geo = np.array([sph2geo(cell) for cell in voronoi_cells])
        lats = cells_geo[:, 0]
        lons = cells_geo[:, 1]

        # Project to local Cartesian (km)
        mean_lat = np.mean(lats)
        x_km = (lons - np.mean(lons)) * 111 * np.cos(np.radians(mean_lat))
        y_km = (lats - np.mean(lats)) * 111

        points_2d = np.column_stack([x_km, y_km])

        # Use KDTree for fast nearest neighbor search
        tree = cKDTree(points_2d)

        # Find nearest neighbor for each point (k=2 to skip self)
        distances, indices = tree.query(points_2d, k=2)
        nearest_distances = distances[:, 1]

        # Convert to cell width estimate
        mean_spacing = np.mean(nearest_distances)

        return mean_spacing * 1.5


    @_utilities.log_errors(logger)
    def _generate_voronoi_cells(self, phase, kvoronoi, nvoronoi, alpha, adaptive_weight):
        """
        Generate Voronoi cells using k-medians clustering of raypaths with optional
        adaptive data-driven adjustment.

        Args:
            phase: The seismic phase to consider
            kvoronoi: Number of cells to generate using k-medians clustering
            nvoronoi: Total number of Voronoi cells to generate
            alpha: Parameter controlling surface bias (0 for uniform, higher for more surface bias)
            adaptive_weight: Degree of adaptive meshing. Anything > 0 changes mode
        """
        if RANK == ROOT_RANK:

            logger.debug(
                f"Generating {nvoronoi} Voronoi cells ({kvoronoi} k-medians, "
                f"{nvoronoi-kvoronoi} random) with alpha={alpha}, adaptive_weight={adaptive_weight}"
            )

            # Get model parameters
            if phase == "P":
                model = self.pwave_model
            elif phase == "S":
                model = self.swave_model
            else:
                raise ValueError(f"Unrecognized phase ({phase}) supplied.")

            min_coords = model.min_coords
            max_coords = model.max_coords
            delta = max_coords - min_coords

            # Initialize cells array
            base_cells = None

            # STEP 1: Generate k-medians cells FIRST (if requested)
            if kvoronoi > 0:
                points = self._sample_raypaths(phase)
                # i can't see it happening, but kvoronoi needs to be less then the len(points)! we might could cap it at 90% or something

                if len(points) == 0:
                    logger.warning("No raypath points available for k-medians clustering")
                else:
                    # Apply sampling size control
                    k_medians_percent = self.cfg["algorithm"].get("k_medians_percent", 15)
                    max_points = int(len(self.sampled_arrivals) * k_medians_percent / 100)

                    # Add some (20%?) variability to avoid identical clustering
                    points_variance = np.random.randint(-max_points//5, max_points//5)
                    max_points = max(10, max_points + points_variance)

                    if len(points) > max_points:
                        idxs = np.random.choice(len(points), max_points, replace=False)
                        points = points[idxs]

                    # Ensure points are within model bounds before clustering
                    valid_mask = (
                        (points[:, 0] >= min_coords[0]) & (points[:, 0] <= max_coords[0]) &
                        (points[:, 1] >= min_coords[1]) & (points[:, 1] <= max_coords[1]) &
                        (points[:, 2] >= min_coords[2]) & (points[:, 2] <= max_coords[2])
                    )
                    points = points[valid_mask]

                    if len(points) >= 3 and kvoronoi >= 3:  # bare minimum

                        # Run k-medians clustering in model coordinates
                        model_bounds = (min_coords, max_coords)
                        medians = _clustering.k_medians(kvoronoi, points, model_bounds)

                        # Verify medians are within bounds
                        medians = np.clip(medians, min_coords, max_coords)
                        base_cells = medians

                        logger.debug(f"K-medians used {len(points)} raypath points")
                        logger.info(
                            f"Depth ranges: Raypath {6371-points[:,0].min():.1f} to {6371-points[:,0].max():.1f} km, "
                            f"K-medians {6371-medians[:,0].min():.1f} to {6371-medians[:,0].max():.1f} km"
                        )
                    else:
                        logger.warning(f"Insufficient valid raypath points ({len(points)}) for k-medians clusters")

            # STEP 2: Generate random/adaptive cells for the remainder
            n_random = nvoronoi - kvoronoi
            
            if n_random > 0:

                if alpha == 0:
                    # Uniform depth distribution
                    rho = np.random.rand(n_random, 1) * delta[0] + min_coords[0]
                else:
                    # Higher cell density at surface
                    beta_vals = np.random.beta(1, alpha, size=(n_random, 1))
                    rho = max_coords[0] - beta_vals * delta[0]

                # Generate random lateral coordinates
                theta_phi = np.random.rand(n_random, 2) * delta[1:] + min_coords[1:]
                random_cells = np.hstack([rho, theta_phi])

                # STEP 3: Apply adaptive meshing ONLY to random cells
                if adaptive_weight > 0:
                    density_3d, edges = self._estimate_data_density(phase, adaptive_weight)

                    grid_coords = [
                        (edges[i][:-1] + edges[i][1:]) / 2 
                        for i in range(3)
                    ]

                    interpolator = scipy.interpolate.RegularGridInterpolator(
                        grid_coords,
                        density_3d,
                        bounds_error=False,
                        fill_value=0.0
                    )

                    # Apply adjustment to each random cell
                    for i in range(n_random):
                        cell_pos = random_cells[i]

                        # Sample density in local neighborhood
                        search_radius = delta * 0.15
                        n_search = 10

                        # Generate search points around current position
                        search_offsets = np.random.uniform(-1, 1, (n_search, 3)) * search_radius
                        search_points = cell_pos + search_offsets

                        # Clip to model bounds
                        search_points = np.clip(search_points, min_coords, max_coords)

                        # Evaluate density at search points
                        search_densities = interpolator(search_points)
                        current_density = interpolator(cell_pos.reshape(1, -1))[0]

                        # Find best position (highest density)
                        best_idx = np.argmax(search_densities)
                        best_density = search_densities[best_idx]

                        # Move toward best position ...if better
                        if best_density > current_density * 1.05:
                            move_vector = search_points[best_idx] - cell_pos

                            new_pos = cell_pos + move_vector * adaptive_weight
                            random_cells[i] = new_pos

                # Ensure random cells are within bounds
                random_cells = np.clip(random_cells, min_coords, max_coords)

                # STEP 4: Combine k-medians cells (first) with random/adaptive cells (after)
                if base_cells is not None:
                    base_cells = np.vstack([base_cells, random_cells])
                else:
                    base_cells = random_cells

            # STEP 5: Remove cells that are too close together
            if self.cfg["algorithm"].get("remove_close_cells", True):
                min_distance_rad = float(self.cfg["algorithm"]["min_dist"]) / 6371. * 2 # so smallest size is 2x mindist
                original_count = len(base_cells)

                tree = cKDTree(base_cells)
                close_pairs = tree.query_pairs(min_distance_rad)

                # Important: preserve k-medians cells preferentially
                to_remove = set()
                for i, j in close_pairs:
                    # If one is a k-medians cell and the other isn't, remove the random one
                    if i < kvoronoi and j >= kvoronoi:
                        to_remove.add(j)
                    elif j < kvoronoi and i >= kvoronoi:
                        to_remove.add(i)
                    else:
                        # Both are same type, remove higher index
                        to_remove.add(max(i, j))

                keep_indices = [i for i in range(len(base_cells)) if i not in to_remove]
                base_cells = base_cells[keep_indices]

                if len(to_remove) > 0:
                    logger.info(f"generate_voronoi_cells: removed {len(to_remove)} too-close cells")
                    # Update kvoronoi if any k-medians cells were removed
                    kvoronoi = sum(1 for i in keep_indices if i < kvoronoi)

            self.voronoi_cells = base_cells
            
            # Print diagnostics
            cell_widths_km = self._estimate_voronoi_cell_widths_simple(base_cells)
            n_rays = len(self.sampled_arrivals)
            n_cells = len(self.voronoi_cells)
            rays_per_cell = n_rays / n_cells

            logger.info(f"Cell count: {n_cells} ({kvoronoi} k-medians, {n_cells-kvoronoi} random/adaptive, {len(points)} points)")
            logger.info(f"Average horizontal cell width: {cell_widths_km:.1f} km & rays per cell: {rays_per_cell:.1f}")

            if rays_per_cell < self.cfg["algorithm"]["min_rays_per_cell"]:
                logger.warning(f"Low ray density! Consider reducing nvoronoi and/or increasing events & arrivals")
            elif rays_per_cell > 3 * self.cfg["algorithm"]["min_rays_per_cell"]:
                logger.info(f"High ray density! - Could increase nvoronoi for better resolution")

        self.synchronize(attrs=["voronoi_cells"])
        return True


    def _estimate_data_density(self, phase, adaptive_weight, nbins=None):
        """
        Estimate data density based on arrival counts in model coordinate system.
        
        Parameters:
        -----------
        phase : str
            Phase type
        nbins : int, optional
            Number of bins per dimension
        adaptive_weight : float
            If > 0, return 3D density field and edges for interpolation
            else, return flattened 1D array (legacy behavior)
        
        Returns:
        --------
        If adaptive_weight <= 0 : 1D array of flattened density values
        If adaptive_weight > 0: tuple of (density_3d, edges) where edges are bin edges
        """

        arrivals = self.sampled_arrivals
        events = self.events
        stations = self.stations

        # Join arrivals with stations to get coordinates
        arrival_coords = arrivals.merge(
            stations[['network', 'station', 'depth', 'latitude', 'longitude']], # n.b. now depth, not elevation
            on=['network', 'station'],
            how='left'
        )

        # Join with events using event_id column
        points_data = arrival_coords.merge(
            events,
            on='event_id',
            how='inner'
        )

        if len(points_data) == 0:
            logger.warning("No valid points found for density estimation")
            if adaptive_weight > 0:
                # Return uniform density in 3D
                dummy_bins = nbins if nbins else 5
                uniform_density = np.ones((dummy_bins, dummy_bins, dummy_bins))
                dummy_edges = [np.linspace(0, 1, dummy_bins+1) for _ in range(3)]
                return uniform_density, dummy_edges
            else:
                dummy_size = (nbins**3) if nbins else 125
                return np.ones(dummy_size) / dummy_size

        # Convert coordinates to model spherical coordinates
        event_coords = np.column_stack([
            points_data['latitude_y'].values,   # event lat
            points_data['longitude_y'].values,  # event lon  
            points_data['depth_y'].values       # event depth
        ])
        event_coords_sph = np.array([geo2sph(coord) for coord in event_coords])

        station_coords = np.column_stack([
            points_data['latitude_x'].values,   # station lat
            points_data['longitude_x'].values,  # station lon
            points_data['depth_x'].values       # station depth (not elevation; same scale as event)
        ])
        station_coords_sph = np.array([geo2sph(coord) for coord in station_coords])

        # Compute midpoints in model coordinate system
        midpoints = (event_coords_sph + station_coords_sph) / 2.0

        # Handle NaN values
        valid_points = ~np.isnan(midpoints).any(axis=1)
        if not np.any(valid_points):
            logger.warning("All midpoints are NaN after coordinate conversion")
            if adaptive_weight > 0:
                dummy_bins = nbins if nbins else 5
                uniform_density = np.ones((dummy_bins, dummy_bins, dummy_bins))
                dummy_edges = [np.linspace(0, 1, dummy_bins+1) for _ in range(3)]
                return uniform_density, dummy_edges
            else:
                dummy_size = (nbins**3) if nbins else 125
                return np.ones(dummy_size) / dummy_size

        midpoints = midpoints[valid_points]
        min_coords = self.pwave_model.min_coords
        max_coords = self.pwave_model.max_coords
        
        # Filter points within model bounds
        tolerance = (max_coords - min_coords) * 0.01
        bounded_mask = np.all(
            (midpoints >= min_coords - tolerance) & 
            (midpoints <= max_coords + tolerance), 
            axis=1
        )

        if not np.any(bounded_mask):
            logger.warning("No midpoints fall within model bounds")
            if adaptive_weight > 0:
                dummy_bins = nbins if nbins else 5
                uniform_density = np.ones((dummy_bins, dummy_bins, dummy_bins))
                dummy_edges = [np.linspace(0, 1, dummy_bins+1) for _ in range(3)]
                return uniform_density, dummy_edges
            else:
                dummy_size = (nbins**3) if nbins else 125
                return np.ones(dummy_size) / dummy_size

        midpoints = midpoints[bounded_mask]

        # Determine bins using Scott's rule if not provided
        if not nbins:
            n_points = len(midpoints)
            if n_points < 10:
                nbins = 5
            else:
                std_per_dim = np.std(midpoints, axis=0)
                range_per_dim = max_coords - min_coords
                scott_width = 3.5 * std_per_dim * n_points**(-1/3)
                scott_width = np.where(scott_width > 0, scott_width, range_per_dim * 0.1)
                nbins_per_dim = range_per_dim / scott_width
                nbins = max(5, int(np.mean(nbins_per_dim)))
                nbins = min(nbins, 20)

        # Create 3D histogram
        try:
            hist, edges = np.histogramdd(
                midpoints, 
                bins=nbins,
                range=[(min_coords[i], max_coords[i]) for i in range(3)]
            )

            # Normalize density
            max_density = hist.max()
            if max_density > 0:
                density_3d = hist / max_density
            else:
                density_3d = np.ones_like(hist) / hist.size

            if adaptive_weight > 0:
                return density_3d, edges
            else:
                # Legacy behavior: return flattened
                return density_3d.flatten()

        except Exception as e:
            logger.warning(f"Histogram creation failed: {e}")
            if adaptive_weight > 0:
                dummy_bins = nbins
                uniform_density = np.ones((dummy_bins, dummy_bins, dummy_bins))
                dummy_edges = [np.linspace(min_coords[i], max_coords[i], dummy_bins+1) for i in range(3)]
                return uniform_density, dummy_edges
            else:
                return np.ones(nbins**3) / nbins**3


    def _sample_raypaths(self, phase):
        """Get raypath points from stored HDF5 files."""
        points = np.empty((0, 3))

        logger.debug(f"Model depth range: {6371-self.pwave_model.min_coords[0]} to {6371-self.pwave_model.max_coords[0]}")

        arrivals = self.sampled_arrivals.set_index(["network", "station"]).sort_index()
        index = arrivals.index.unique()
        events = self.events.set_index("event_id")
        events["idx"] = np.arange(len(events))

        for network, station in index:
            # Read raypath file
            filename = f"{network}.{station}.{phase}.h5"
            path = os.path.join(self.raypath_dir, filename)
            with h5py.File(path, mode="r") as raypath_file:
                event_ids = arrivals.loc[[(network, station)], "event_id"]
                idxs = events.loc[event_ids, "idx"]
                idxs = np.sort(idxs).astype(int)
                raypoints = raypath_file[phase][:, idxs]

                if raypoints.ndim > 1:
                    raypoints = np.apply_along_axis(np.concatenate, 1, raypoints)
                else:
                    raypoints = np.stack(raypoints)
                raypoints = raypoints.T
                points = np.vstack([points, raypoints])

        return points


    @_utilities.log_errors(logger)
    def _projected_ray_idxs(self, raypath, hvr):
        """
        Return the cell IDs (column IDs) of each segment of the given
        raypath and the length of each segment in counts.
        """

        min_coords = self.pwave_model.min_coords
        max_coords = self.pwave_model.max_coords
        center = (min_coords + max_coords) / 2

        voronoi_cells = self.voronoi_cells
        voronoi_cells = center + (voronoi_cells - center) / [1, hvr, hvr] #n.b. dividing here is correct. hvr > 1 makes wider cells 

        voronoi_cells = sph2xyz(voronoi_cells)
        tree = cKDTree(voronoi_cells)

        raypath = center + (raypath - center) / [1, hvr, hvr]
        raypath = sph2xyz(raypath)

        _, column_idxs = tree.query(raypath)
        column_idxs, counts = np.unique(column_idxs, return_counts=True)

        logger.info(f"Ray query results: {len(column_idxs)} points, counts range: {counts.min()}-{counts.max()}")

        return (column_idxs, counts)

    @_utilities.log_errors(logger)
    def _request_dispatch(self):
        """
        Request, receive, and return item from dispatcher.
        """
        COMM.send(
            RANK,
            dest=ROOT_RANK,
            tag=_constants.DISPATCH_REQUEST_TAG
        )
        item = COMM.recv(
            source=ROOT_RANK,
            tag=_constants.DISPATCH_TRANSMISSION_TAG
        )

        return item


    @_utilities.log_errors(logger)
    @_utilities.root_only(RANK)
    def _reset_realization_stack_old(self, phase):
        """
        Reset the realization stack values to np.nan for the given phase.
        """
        logger.info("Resetting realization stacks...") 
        phase = phase.lower()
        stack = getattr(self, f"{phase}wave_realization_stack")
        stack[:] = np.nan # < this was redundant

        # also do the quality stacks (new! experimental!)
        stack = getattr(self, f"{phase}qual_realization_stack")
        stack[:] = np.nan

        return


    @_utilities.log_errors(logger)
    @_utilities.root_only(RANK)
    def _reset_realization_stack(self, phase):
        """
        Reset the realization stack by deleting and recreating
        """
        phase = phase.lower()

        for prefix in ["wave", "qual"]:
            handle = f"{phase}{prefix}_realization_stack"
            dataset_name = f"{phase}{prefix}_stack"
            if dataset_name in self._f5_workspace:
                del self._f5_workspace[dataset_name]

            stack = getattr(self,handle)

        return


    @_utilities.log_errors(logger)
    def _sample_arrivals(self, phase, useall=False, do_remove_outliers=True):
        """
        Draw a random sample of arrivals and update the sampled_arrivals attribute
        """
        if RANK == ROOT_RANK:

            # Filter by phase first, then by event_id
            arrivals = self.arrivals[self.arrivals["phase"] == phase]
            event_ids = self.sampled_events["event_id"]
            arrivals = arrivals[arrivals["event_id"].isin(event_ids)].copy()

            if self.cfg["algorithm"]["narrival_percent"] > 0:
                narrival = int(len(arrivals) * self.cfg["algorithm"]["narrival_percent"]/100)
            else:
                narrival = self.cfg["algorithm"]["narrival"]

            if do_remove_outliers:
                tukey_k = self.cfg["algorithm"]["outlier_removal_factor"]
                max_arr_resid = self.cfg["algorithm"]["max_arrival_residual"]
                n0 = len(arrivals)
                arrivals = remove_outliers(arrivals, tukey_k, "residual", max_arr_resid)
                if len(arrivals) < 0.75*n0:
                    logger.warn(f"FYI: remove_outliers removed {(n0-len(arrivals))/n0*100:.1f}% of your arrivals!")

            if not useall and narrival < len(arrivals):
                if 'weight' not in arrivals.columns:
                    arrivals['weight'] = 1.0
                    logger.warning("arrivals['weight'] column missing in sample_arrivals??")
                arrivals = arrivals.sample(n=narrival, weights="weight")

            self.sampled_arrivals = arrivals

        self.synchronize(attrs=["sampled_arrivals"])

        return


    @_utilities.log_errors(logger)
    def _sample_events(self, useall=False, do_remove_outliers=True):
        """
        Draw a random sample of events and update the sampled_events attribute
        Note that events aren't subject to Tukey fencing.
        """
        if RANK == ROOT_RANK:

            if self.cfg["algorithm"]["nevent_percent"] > 0:
                nevent = int(len(self.events) * self.cfg["algorithm"]["nevent_percent"]/100)
            else:
                nevent = self.cfg["algorithm"]["nevent"]

            if do_remove_outliers:
                max_evt_resid = self.cfg["algorithm"]["max_event_residual"]
                n0 = len(self.events)
                events = remove_outliers(self.events, None, "residual", max_evt_resid)
                if len(self.events) < 0.75*n0:
                    logger.warn(f"FYI: remove_outliers removed {(n0-len(self.events))/n0*100:.1f}% of your events!")
            else:
                events = self.events

            if not useall and nevent < len(events):
                if 'weight' not in events.columns:
                    events['weight'] = 1.0
                    logger.warning("events['weight'] column missing in sample_events??")
                events = events.sample(n=nevent, weights='weight')

            self.sampled_events = events

        self.synchronize(attrs=["sampled_events"])

        return


    @_utilities.log_errors(logger)
    def _trace_rays(self, phase):
        """
        Trace rays for all arrivals in self.sampled_arrivals and store
        in HDF5 file. Only trace non-existent raypaths to save time.
        """

        raypath_dir = self.raypath_dir
        arrivals = self.sampled_arrivals
        arrivals = arrivals.set_index(["network", "station"])
        arrivals = arrivals.sort_index()

        if RANK == ROOT_RANK:
            logger.debug("Tracing rays")
            os.makedirs(raypath_dir, exist_ok=True)
            index = arrivals.index.unique()
            self._dispatch(index)

        else:
            events = self.events
            events = events.set_index("event_id")
            events["idx"] = range(len(events))

            _path = self.traveltime_inventory_path
            with TraveltimeInventory(_path, mode="r") as traveltime_inventory:
                while True:

                    item = self._request_dispatch()

                    if item is None:
                        break

                    network, station = item
                    handle = "/".join([network, station, phase])

                    traveltime = traveltime_inventory.read(handle)

                    filename = ".".join([network, station, phase])
                    path = os.path.join(raypath_dir, filename + ".h5")
                    raypath_file = h5py.File(path, mode="a")

                    if phase not in raypath_file:
                        dtype = h5py.vlen_dtype(_constants.DTYPE_REAL)
                        dataset = raypath_file.create_dataset(
                            phase,
                            (3, len(events),),
                            dtype=dtype
                        )
                    else:
                        dataset = raypath_file[phase]

                    event_ids = arrivals.loc[[(network, station)], "event_id"].values

                    for event_id in event_ids:

                        event = events.loc[event_id]
                        idx = int(event["idx"])

                        if np.stack(dataset[:, idx]).size != 0:
                            continue

                        columns = ["latitude", "longitude", "depth"]
                        coords = event[columns]
                        coords = geo2sph(coords)

                        # trace_ray does not handle bad events very well.. skipping them should be OK?
                        try:
                            raypath = traveltime.trace_ray(coords)
                            dataset[:, idx] = raypath.T

                            # check for empty paths
                            if len(raypath) < 1:
                                logger.warning(f"Empty raypath for event_id {event_id}, {network}, {station}, {phase}")
                                continue

                            # ensure consistent shape
                            if raypath.ndim != 2 or raypath.shape[1] != 3:
                                logger.warning(f"Invalid raypath shape {raypath.shape} for event_id {event_id}")
                                continue

                        except Exception as e:
                            logger.warning(f"traveltime issue with event_id {event_id}, coords {coords}, {network}, {station}, {phase}")
                            print(e)

                            try:
                                min_val = np.min(traveltime.values[~np.isinf(traveltime.values)])
                                # replacing inf tt's to a hair under the min should be functionally equivalent
                                traveltime.values[traveltime.values == -np.inf] = 0.97 * min_val
                                raypath = traveltime.trace_ray(coords)
                                dataset[:, idx] = raypath.T
                                logger.info("...success on second try!")
                            except Exception as e:
                                logger.warning("...couldn't fix traveltime: ", e)
                                continue

                    raypath_file.close()

        COMM.barrier()
        return True


    @_utilities.log_errors(logger)
    def track_residual_improvement(self, min_improvement=0.01, safe_residual=0.3):
        """
        Track arrival residuals per iteration,
        remove if above safe_residual AND not improving
        min_improvement is a fraction (.01 = 1%), safe_residual in seconds
        """

        if RANK == ROOT_RANK:
            logger.debug(f"Tracking residuals for iteration {self.iiter}...")

            current_arrival_ids = set(self.arrivals['arrival_id'])
            current_event_ids = set(self.events['event_id'])

            # Update only for arrivals/events that still exist
            history_arrival_mask = self.arrival_history['arrival_id'].isin(current_arrival_ids)
            self.arrival_history = self.arrival_history[history_arrival_mask].reset_index(drop=True)

            history_event_mask = self.event_history['event_id'].isin(current_event_ids)
            self.event_history = self.event_history[history_event_mask].reset_index(drop=True)

            # Add residuals for current iteration / create a mapping for efficient lookup
            arrival_residual_map = dict(zip(self.arrivals['arrival_id'], self.arrivals['residual'].abs()))
            event_residual_map = dict(zip(self.events['event_id'], self.events['residual'].abs()))

            # Update histories
            self.arrival_history[f'iter_{self.iiter}'] = (
                self.arrival_history['arrival_id'].map(arrival_residual_map)
            )
            self.event_history[f'iter_{self.iiter}'] = (
                self.event_history['event_id'].map(event_residual_map)
            )

            # Check improvements (only for items with valid previous values)
            prev_col = f'iter_{self.iiter - 1}'
            curr_col = f'iter_{self.iiter}'

            # Arrival improvements
            prev_arrival_residuals = self.arrival_history[prev_col]
            curr_arrival_residuals = self.arrival_history[curr_col]

            # Only check improvement for arrivals with residuals above safe_residual
            significant_arrival_mask = prev_arrival_residuals > safe_residual
            arrival_improvement = (prev_arrival_residuals - curr_arrival_residuals) / (prev_arrival_residuals + 1e-6)

            # Only remove if residual is above safe_residual AND not improving
            arrival_mask_to_remove = significant_arrival_mask & (arrival_improvement < min_improvement)
            arrivals_to_remove = self.arrival_history.loc[arrival_mask_to_remove, 'arrival_id'].values

            # Calculate mean improvement using mean of residuals
            mean_prev_arrival = self.arrival_history[prev_col].mean()
            mean_curr_arrival = self.arrival_history[curr_col].mean()
            mean_arrival_improvement = ((mean_prev_arrival - mean_curr_arrival) / mean_prev_arrival) * 100 if mean_prev_arrival > 0 else 0
            logger.info(f"Mean arrival residual reduction: {mean_arrival_improvement:.2f}% "
                   f"({mean_prev_arrival:.4f} -> {mean_curr_arrival:.4f})")

            # Event improvements
            prev_event_residuals = self.event_history[prev_col]
            curr_event_residuals = self.event_history[curr_col]

            # Only check improvement for events with residuals above safe_residual
            significant_event_mask = prev_event_residuals > safe_residual
            event_improvement = (prev_event_residuals - curr_event_residuals) / (prev_event_residuals + 1e-6)

            # Only remove if residual is above safe_residual AND not improving
            event_mask_to_remove = significant_event_mask & (event_improvement < min_improvement)
            events_to_remove = self.event_history.loc[event_mask_to_remove, 'event_id'].values

            # Calculate mean improvement using mean of residuals
            mean_prev_event = self.event_history[prev_col].mean()
            mean_curr_event = self.event_history[curr_col].mean()
            mean_event_improvement = ((mean_prev_event - mean_curr_event) / mean_prev_event) * 100 if mean_prev_event > 0 else 0
            logger.info(f"Mean event residual reduction: {mean_event_improvement:.2f}% "
                   f"({mean_prev_event:.4f} -> {mean_curr_event:.4f})")

            # Remove non-improving events FIRST (and their associated arrivals)
            if len(events_to_remove) > 0:
                # Remove events
                self.events = self.events[~self.events['event_id'].isin(events_to_remove)].reset_index(drop=True)
                self.event_history = self.event_history[~event_mask_to_remove].reset_index(drop=True)

                # Remove all arrivals associated with removed events
                arrivals_before = len(self.arrivals)
                self.arrivals = self.arrivals[~self.arrivals['event_id'].isin(events_to_remove)].reset_index(drop=True)
                arrivals_removed_by_event = arrivals_before - len(self.arrivals)

                # Update arrival history to remove arrivals from deleted events
                arrival_event_mask = ~self.arrival_history['event_id'].isin(events_to_remove)
                self.arrival_history = self.arrival_history[arrival_event_mask].reset_index(drop=True)

                logger.info(f"Removed {len(events_to_remove)} non-improving events (< {min_improvement*100}%) "
                           f"and {arrivals_removed_by_event} associated arrivals")

            # Remove individual non-improving arrivals (only those not already removed)
            if len(arrivals_to_remove) > 0:
                # Filter out arrival_ids that were already removed with their events
                remaining_arrival_ids = set(self.arrivals['arrival_id'])
                arrivals_to_remove = [aid for aid in arrivals_to_remove if aid in remaining_arrival_ids]

                if len(arrivals_to_remove) > 0:
                    if len(arrivals_to_remove) > 0.5 * len(self.arrivals):
                        logger.warning(f"We are over-removing arrivals and probably need to stop now")

                    self.arrivals = self.arrivals[~self.arrivals['arrival_id'].isin(arrivals_to_remove)].reset_index(drop=True)

                    # Update arrival history
                    arrival_mask_still_valid = self.arrival_history['arrival_id'].isin(self.arrivals['arrival_id'])
                    self.arrival_history = self.arrival_history[arrival_mask_still_valid].reset_index(drop=True)
                    logger.info(f"Removed {len(arrivals_to_remove)} additional non-improving arrivals (< {min_improvement*100}%)")

        self.synchronize(attrs=["arrivals","arrival_history","events","event_history"])
        return True


    @_utilities.log_errors(logger)
    def update_event_weights(self,npts=16):
        """
        Update events weights using KDE for homogeneous raypath sampling

        Args:
            npts: Number of points for KDE grid evaluation (16 is fine)
        """
        logger.info("Updating event KDE weights for raypath sampling")

        if RANK == ROOT_RANK:
            events = self.events
            kde_columns = ["latitude", "longitude", "depth"]
            ndim = len(kde_columns)
            data = events[kde_columns].values

            # IQR normalization
            data_iqr = iqr(data, axis=0)
            data_median = np.median(data, axis=0)

            # handle zero-variance dimensions
            mask_zero_iqr = data_iqr == 0
            if np.any(mask_zero_iqr):
                logger.warning(f"Zero variance detected in dimensions: {kde_columns[mask_zero_iqr]}")
                data_iqr[mask_zero_iqr] = np.median(data_iqr[~mask_zero_iqr])

            # normalize
            data_normalized = (data - data_median) / data_iqr

            # compute robust bandwidth using Scott's rule (which seems to be a bit larger than silverman)
            n, d = data_normalized.shape
            sigma = np.std(data_normalized, ddof=1)
            bandwidth_scott = (4 / (n * (2 * d + 1)))**(1 / (d + 4)) * sigma 
            #bandwidth_silverman = (n * (d + 2) / 4)**(-1. / (d + 4)) * sigma
            bandwidth = bandwidth_scott

            try:

                # Fit and evaluate KDE
                kde = kp.FFTKDE(kernel='gaussian',bw=bandwidth).fit(data_normalized)
                points, values = kde.evaluate(npts)

                # Reshape grid points and values
                points = [np.unique(points[:,i]) for i in range(ndim)]
                values = values.reshape((npts,) * ndim)

                # Create interpolator with more robust error handling
                interpolator = scipy.interpolate.RegularGridInterpolator(
                    points,
                    values,
                    method='linear',
                    bounds_error=False,
                    fill_value=np.min(values)
                )

                # Compute densities at data points
                densities = interpolator(data_normalized)

                # let's just keep it simple
                if self.iiter < 7:
                    weights = 1.0 / densities
                #elif self.iiter >= 7:
                else:
                    # equal weighting
                    weights = np.ones_like(densities)
                #else:
                #    # steps 2-4: linear transition between the two extremes
                #    progress = (self.iiter - 1) / 4  # ie from 0.25 (step 2) to 0.75 (step 4)
                #    uniform_weight = np.ones_like(densities)
                #    weights = (1 - progress) * (1/densities) + progress * uniform_weight


                # set any problem infinite or NaN values to 0..
                n_infinite = np.sum(~np.isfinite(weights))
                weights[~np.isfinite(weights)] = 0
                if n_infinite > 0:
                    logger.warning(f"Removed {n_infinite} infinite/NaN event weights")

                events["weight"] = weights
                self.events = events

                logger.info(f"  (event bandwidth = {bandwidth:.2f})")

                # >>> COMPLETELY SEPARATE RESIDUAL ANALYSIS but may as well warn about it here

                # check for NaN values and warn
                residuals = events['residual']
                nan_count = residuals.isna().sum()
                if nan_count > 0:
                    logger.warning(f"Found {nan_count} events with NaN residuals")

                # calculate/print residual statistics
                valid_residuals = residuals.dropna()
                if len(valid_residuals) > 0:
                    logger.info(
                        f"mean event residual (s): {valid_residuals.mean():.4f} "
                        f"({valid_residuals.std():.4f} std)"
                    )
                else:
                    logger.warning("No valid event residuals found - all are NaN!!!")

            except Exception as e:
                logger.error(f"KDE fitting failed: {str(e)}")
                events["weight"] = 1.0
                self.events = events
                return False

        self.synchronize(attrs=["events"])
        return True


    @_utilities.log_errors(logger)
    def update_arrival_weights(self,phase,npts=16):
        """
        Update arrival weights using KDE for homogeneous raypath sampling

        Args:
            phase: Phase type to process
            npts: Number of points for KDE grid evaluation (default 16)
        """
        logger.info(f"Updating {phase} arrival KDE weights for raypath sampling")

        if RANK == ROOT_RANK:
            try:
                # get parameters
                max_arr_resid = self.cfg["algorithm"]["max_arrival_residual"]

                arrivals = self.arrivals[self.arrivals["phase"] == phase]

                # merge event data with renamed columns
                events = self.events.rename(columns={
                    "latitude": "event_latitude",
                    "longitude": "event_longitude",
                    "depth": "event_depth"
                })
                merge_columns = ["event_latitude", "event_longitude", "event_depth", "event_id"]
                arrivals = arrivals.merge(events[merge_columns], on="event_id")

                # merge station data with renamed columns
                stations = self.stations.rename(columns={
                    "latitude": "station_latitude",
                    "longitude": "station_longitude"
                })
                merge_columns = ["station_latitude", "station_longitude", "network", "station"]
                merge_keys = ["network", "station"]
                arrivals = arrivals.merge(stations[merge_columns], on=merge_keys)

                # compute ray geometry
                dlat = arrivals["event_latitude"] - arrivals["station_latitude"]
                dlon = arrivals["event_longitude"] - arrivals["station_longitude"]
                arrivals["azimuth"] = np.arctan2(dlat, dlon)
                arrivals["delta"] = dist_deg(
                    arrivals["event_latitude"],
                    arrivals["event_longitude"],
                    arrivals["station_latitude"],
                    arrivals["station_longitude"]
                )

                # prepare data for KDE
                kde_columns = ["event_latitude", "event_longitude", "event_depth", "azimuth", "delta"]
                ndim = len(kde_columns)
                data = arrivals[kde_columns].values

                # IQR normalization
                data_iqr = iqr(data, axis=0)
                data_median = np.median(data, axis=0)

                # handle zero-variance dimensions
                mask_zero_iqr = data_iqr == 0
                if np.any(mask_zero_iqr):
                    logger.warning(f"Zero variance detected in dimensions: {np.array(kde_columns)[mask_zero_iqr]}")
                    data_iqr[mask_zero_iqr] = np.median(data_iqr[~mask_zero_iqr])

                data_normalized = (data - data_median) / data_iqr

                # compute Scott's bandwidth
                n, d = data_normalized.shape
                sigma = np.std(data_normalized, ddof=1)
                bandwidth_scott = (4 / (n * (2 * d + 1)))**(1 / (d + 4)) * sigma 
                #bandwidth_silverman = (n * (d + 2) / 4)**(-1. / (d + 4)) * sigma
                bandwidth = bandwidth_scott


                # Fit and evaluate KDE
                kde = kp.FFTKDE(kernel='gaussian',bw=bandwidth).fit(data_normalized)
                points, values = kde.evaluate(npts)
                points = [np.unique(points[:,i]) for i in range(ndim)]
                values = values.reshape((npts,) * ndim)

                # Interpolate densities
                interpolator = scipy.interpolate.RegularGridInterpolator(
                    points,
                    values,
                    method='linear',
                    bounds_error=False,
                    fill_value=np.min(values)
                )
                densities = interpolator(data_normalized)

                # calculate weights... inversely proportional to density of raypaths!
                # np.exp (at least!) needed to add some separation
                dataweight = 1 / np.exp(densities)

                # set any infinite or NaN values to 0
                dataweight[~np.isfinite(dataweight)] = 0

                arrivals["weight"] = dataweight

                # update self.arrivals with new weights
                index_columns = ["network", "station", "event_id", "phase"]
                arrivals = arrivals.set_index(index_columns)
                _arrivals = self.arrivals.set_index(index_columns).sort_index()
                _arrivals.loc[arrivals.index, "weight"] = arrivals["weight"]
                self.arrivals = _arrivals.reset_index()

                # log statistics
                valid_arrivals = arrivals[abs(arrivals['residual']) <= max_arr_resid]['residual']
                logger.info(f"  ({phase} arrival bandwidth = {bandwidth:.2f})")
                logger.info(
                    f"mean {phase} arrival residual (s)     : "
                    f"{valid_arrivals.mean():.4f} ({valid_arrivals.std():.4f} std)"
                )
                logger.info(
                    f"   mean abs arrival residual (s): "
                    f"{valid_arrivals.abs().mean():.4f} ({valid_arrivals.abs().std():.4f} std)"
                )

            except Exception as e:
                logger.error(f"Failed to update arrival weights: {str(e)}")
                return False

        self.synchronize(attrs=["arrivals"])
        return True


    @_utilities.log_errors(logger)
    def _update_projection_matrix(self, phase, hvr):
        """
        Update the projection matrix using the current Voronoi cells.

        Args:
            phase: P or S, but shouldn't be needed necessarily
            hvr: Horizontal to vertical ratio for scaling
        """

        if RANK == ROOT_RANK:
            logger.debug("Updating projection matrix")

            if phase == "P":
                model = self.pwave_model
            elif phase == "S":
                model = self.swave_model
            else:
                raise (ValueError(f"Unrecognized phase ({phase}) supplied."))

            nvoronoi = len(self.voronoi_cells)
            min_coords = model.min_coords
            max_coords = model.max_coords
            center = (min_coords + max_coords) / 2

            # transform voronoi cells
            voronoi_cells = self.voronoi_cells
            voronoi_cells = center + (voronoi_cells - center) / [1, hvr, hvr]
            voronoi_cells = sph2xyz(voronoi_cells)
            tree = cKDTree(voronoi_cells)

            # transform nodes
            nodes = model.nodes
            nodes = center + (nodes - center) / [1, hvr, hvr]
            nodes = nodes.reshape(-1, 3)
            nodes = sph2xyz(nodes)

            # get mapping from nodes to voronoi cells
            _, column_ids = tree.query(nodes)
            nnodes = np.prod(model.nodes.shape[:-1])
            row_ids = np.arange(nnodes)
            values = np.ones(nnodes,)

            # create base projection matrix
            proj_matrix = scipy.sparse.coo_matrix(
                (values, (row_ids, column_ids)),
                shape=(nnodes, nvoronoi)
            )

            self.projection_matrix = proj_matrix

        self.synchronize(attrs=["projection_matrix"])
        return True

    @_utilities.log_errors(logger)
    def compute_traveltime_lookup_tables(self,run_phases=None):
        """
        Compute traveltime-lookup tables for both (default) or individual phases
        """

        logger.info(f"Computing traveltime tables for {len(self.stations)} stations...")
        traveltime_dir = self.traveltime_dir

        # sometimes makes sense to just calculate the specific phase tables
        run_phases = run_phases or self.phases

        if RANK == ROOT_RANK:
            logger.info(f"  Building traveltimes here: {traveltime_dir}")
            os.makedirs(traveltime_dir, exist_ok=True)
            ids = zip(self.stations["network"], self.stations["station"])
            self._dispatch(sorted(ids))
        else:
            geometry = self.stations
            geometry = geometry.set_index(["network", "station"])

            while True:
                # request an event
                item = self._request_dispatch()

                if item is None:
                    break

                network, station = item

                keys = ["latitude", "longitude", "depth"]
                coords = geometry.loc[(network, station), keys]
                coords = geo2sph(coords)

                for phase in run_phases:
                    handle = f"{phase.lower()}wave_model"
                    model = getattr(self, handle)
                    solver = PointSourceSolver(coord_sys="spherical")
                    solver.vv.min_coords = model.min_coords
                    solver.vv.node_intervals = model.node_intervals
                    solver.vv.npts = model.npts
                    solver.vv.values = model.values
                    solver.src_loc = coords
                    solver.solve() # we're seeing -inf traveltimes

                    if solver.tt.values.min() == -np.inf:  # fixes the few trouble arrivals
                        logger.warn("-inf values found in solver.tt ...no problem, setting these to a safe value")
                        min_val = np.min(solver.tt.values[~np.isinf(solver.tt.values)])
                        solver.tt.values[solver.tt.values == -np.inf] = 0.95 * min_val
                    path = os.path.join(traveltime_dir,f"{network}.{station}.{phase}.h5")
                    solver.tt.to_hdf(path)

        COMM.barrier()

        if RANK == ROOT_RANK:
            _path = self.traveltime_inventory_path
            if os.path.isfile(_path):
                os.remove(_path)
            with TraveltimeInventory(_path, mode="w") as tt_inventory:
                pattern = os.path.join(traveltime_dir, "*.h5")
                paths = glob.glob(pattern)
                paths = sorted(paths)
                tt_inventory.merge(paths)

            shutil.rmtree(self.traveltime_dir)

        COMM.barrier()
        return True


    @_utilities.log_errors(logger)
    def iterate(self):
        """
        Execute one iteration the entire inversion procedure including
        updating velocity models, event locations, and arrival residuals.
        """
        output_dir = self.argc.output_dir

        niter = self.cfg["algorithm"]["niter"]
        hvr = self.cfg["algorithm"]["hvr"] # note this WAS a list, now just a float
        nvoronoi = self.cfg["algorithm"]["nvoronoi"]
        kvoronoi_percent = self.cfg["algorithm"].get("kvoronoi",5)

        alpha = self.cfg["algorithm"]["paretos_alpha"]
        nreal = self.cfg["algorithm"]["nreal"]
        relocation_method = self.cfg["relocate"]["method"]
        min_rays_per_cell = self.cfg["algorithm"]["min_rays_per_cell"]

        adaptive_weight = self.cfg["algorithm"].get("adaptive_data_weight", 0.0)
        adaptive_weight = max(0,min(adaptive_weight,1.0))

        # are we ONLY doing the resolution test? (via -t flag)
        if self.argc.test_only:
            return self.run_resolution_test()

        self.iiter += 1

        try:
            phase_order = self.cfg["algorithm"]["phase_order"]
        except:
            phase_order =  ['P', 'S']

        logger.info(f"Iteration #{self.iiter} (/{niter}) with hvr = {hvr}")

        if self.cfg["argc"]["relocate_first"]=="False":
            self.sanitize_data()
        else:
            self.resanitize_data()

        self.check_event_bounds()

        for phase in phase_order:
            logger.info(f" >>> Starting {phase}-wave iteration {self.iiter}/{niter} <<<")

            if self.cfg["argc"]["relocate_first"]=="False" and self.iiter == 1: # nb relocate_first arg is a string, not bool
                self.update_arrival_weights(phase)
            self._reset_realization_stack(phase)

            for self.ireal in range(nreal):
                logger.info(f"{phase} Realization # {self.ireal+1}/{nreal} | Iteration # {self.iiter}/{niter}")
                self._sample_events()
                self._sample_arrivals(phase)
                self._trace_rays(phase)

                # add some minor stochastic variability
                mod_nvoronoi = int(nvoronoi*np.random.uniform(low=0.70, high=1.15)) # mostly dip lower but sometimes higher (TODO set in params?)
                kvoronoi = int(mod_nvoronoi * kvoronoi_percent/100)
                mod_kvoronoi = min(kvoronoi,int(mod_nvoronoi*0.7)) # sanity cap k's at 70% total n's

                self._generate_voronoi_cells(phase,mod_kvoronoi,mod_nvoronoi,alpha,adaptive_weight)

                self._compute_sensitivity_matrix(phase,hvr)
                self._update_projection_matrix(phase,hvr)
                self._compute_model_update(phase,min_rays=min_rays_per_cell)

            self.update_model(phase)

            if not self.argc.test_only:
                self.save_model(phase, tag=f"h{hvr}")

        self.compute_traveltime_lookup_tables() # without an argument, computes both phases
        self.relocate_events(method=relocation_method) # also calls update_arrival_residuals, update_event_weights, and update_arrival_weights
        self.track_residual_improvement() # track improvement of residuals and boot any gremlins TODO: can also be used to prematurely stop iterations?

        if self.iiter <= 3:
            self.check_event_migration() # implement a check to see if EQs have migrated a great deal (located very poorly to begin with!)

        self.purge_raypaths()
        self.resanitize_data()
        self.save_events() #n.b. the first 00.events.h5 is the initial relocated (-r) model, may be faster to start from this in the future
        #self.save_stations() # not in use.. yet?


    @_utilities.log_errors(logger)
    def check_event_bounds(self):
        """
        Remove events that have been runaway migrated beyond some boundary depth, latitude, or longitude
        """
        logger.info("Checking for out of bounds events...")

        if RANK == ROOT_RANK:
            max_lat   = self.cfg["algorithm"]["max_lat"]
            min_lat   = self.cfg["algorithm"]["min_lat"]
            max_lon   = self.cfg["algorithm"]["max_lon"]
            min_lon   = self.cfg["algorithm"]["min_lon"]
            max_depth = self.cfg["algorithm"]["max_depth"]
            min_depth = self.cfg["algorithm"]["min_depth"]

            events = self.events
            n0 = len(events)
            events0 = self.events0 # e.g. input catalog
            merged = pd.merge(events0, events, on='event_id', suffixes=('_0', ''))

            filtered = merged.copy()
            filters = [
                ('lat', lambda df: (df['latitude'] >= min_lat) & (df['latitude'] <= max_lat)),
                ('lon', lambda df: (df['longitude'] >= min_lon) & (df['longitude'] <= max_lon)),
                ('depth', lambda df: (df['depth'] >= min_depth) & (df['depth'] <= max_depth))
            ]

            for filter_name, condition in filters:
                before_count = len(filtered)
                filtered = filtered[condition]
                after_count = len(filtered)
                dropped_count = before_count - after_count
                if dropped_count > 0:
                    logger.info(" %10s bounds filter: %5d events dropped (%5d remaining)" % (filter_name,dropped_count,after_count))

            dropped_event_ids = set(events['event_id']) - set(filtered['event_id'])

            events = filtered[events.columns]
            self.events = events

            if len(dropped_event_ids) > 0:
                dn = len(dropped_event_ids)
                # also have to toss any arrivals referencing these dropped events
                arrivals = self.arrivals
                arrivals = arrivals[~arrivals['event_id'].isin(dropped_event_ids)]
                self.arrivals = arrivals
                logger.info(f"   ...Dropped {dn} total events which are out of bounds. {n0-dn} remain.")

        self.synchronize(attrs=['events','arrivals'])
        return True


    @_utilities.log_errors(logger)
    def check_event_migration(self):
        """
        Remove events that have been runaway migrated beyond some tolerance
        """
        if RANK == ROOT_RANK:
            logger.info("Removing runaway event migrations.")

            events = self.events
            n0 = len(events)
            events0 = self.events0 # e.g. input catalog
            merged = pd.merge(events0, events, on='event_id', suffixes=('_0', ''))

            # calculate absolute differences
            merged['dlat'] = np.abs(merged['latitude'] - merged['latitude_0'])
            merged['dlon'] = np.abs(merged['longitude'] - merged['longitude_0'])
            merged['ddepth'] = np.abs(merged['depth'] - merged['depth_0'])
            merged['dtime'] = np.abs(merged['time'] - merged['time_0'])

            # Not sure if wise to do this.. for now just set factor to a huge number
            def get_mad_threshold(x, factor=4): #factor ~ # std
                median = np.median(x)
                mad = np.median(np.abs(x - median))
                return median + factor * mad * 1.4826  # 1.4826 scales MAD to equivalent std

            max_dlat = self.cfg['algorithm']['max_dlat']
            max_dlon = self.cfg['algorithm']['max_dlon']
            max_ddepth = self.cfg['algorithm']['max_ddepth']
            max_dtime = self.cfg['algorithm']['max_dtime']
            max_evt_resid = self.cfg['algorithm']['max_event_residual']

            # apply filters and log results
            filtered = merged.copy()
            filters = [
                ('dlat', lambda df: df['dlat'] <= max_dlat),
                ('dlon', lambda df: df['dlon'] <= max_dlon),
                ('ddepth', lambda df: df['ddepth'] <= max_ddepth),
                ('dtime', lambda df: df['dtime'] <= max_dtime),
                ('residual', lambda df: df['residual'] <= max_evt_resid)]

            for filter_name, condition in filters:
                before_count = len(filtered)
                filtered = filtered[condition]
                after_count = len(filtered)
                dropped_count = before_count - after_count
                logger.info(" %10s migration filter: %5d events dropped (%5d remaining)" % (filter_name,dropped_count,after_count))

            dropped_event_ids = set(events['event_id']) - set(filtered['event_id'])

            events = filtered[events.columns]
            self.events = events

            if len(dropped_event_ids) > 0:
                dn = len(dropped_event_ids)
                # also have to toss any arrivals referencing these dropped events (now done elsewhere)
                arrivals = self.arrivals
                arrivals = arrivals[~arrivals['event_id'].isin(dropped_event_ids)]
                self.arrivals = arrivals
                logger.info(f"   ...Dropped {dn} events which have migrated too far from original position. {n0-dn} remain.")
                for ele in dropped_event_ids:
                    logger.debug(f"dropped event: %6d" % ele)

        self.synchronize(attrs=['events','arrivals'])
        return len(self.events) > 0


    @_utilities.log_errors(logger)
    def load_cfg(self):
        """
        Parse and store configuration-file parameters.

        ROOT_RANK parses configuration file and broadcasts contents to all
        other processes.
        """
        if RANK == ROOT_RANK:
            logger.info("Loading configuration parameters")

            # parse configuration-file parameters.
            self.cfg = _utilities.parse_cfg(self.argc.configuration_file)
            _utilities.write_cfg(self.argc, self.cfg)

        self.synchronize(attrs=["cfg"])

        return True


    @_utilities.log_errors(logger)
    def load_event_data(self):
        """
        Parse and return event data from file.

        ROOT_RANK parses file and broadcasts contents to all other processes.
        """
        if RANK == ROOT_RANK:
            logger.info("Loading event data")

            data = _dataio.parse_event_data(self.argc)
            self.events, self.arrivals = data

            self.arrival_history = pd.DataFrame({
                'arrival_id': self.arrivals['arrival_id'],
                'event_id': self.arrivals['event_id'],
                f'iter_{self.iiter}': self.arrivals['residual'].abs()
            })

            self.event_history = pd.DataFrame({
                'event_id': self.events['event_id'],
                f'iter_{self.iiter}': self.events['residual'].abs()
            })

            # register the available phase types also just in case
            phases = self.arrivals["phase"]
            phases = phases.unique()
            self.phases = sorted(phases)

        self.synchronize(attrs=["events", "arrivals", "event_history", "arrival_history", "phases"])

        return True


    @_utilities.log_errors(logger)
    def load_network_geometry(self):
        """
        Parse and return network geometry from file.

        ROOT_RANK parses file and broadcasts contents to all other processes.
        """
        if RANK == ROOT_RANK:
            logger.info("Loading stations / network geometry")

            # parse station data.
            stations = _dataio.parse_network_geometry(self.argc)
            self.stations = stations

        self.synchronize(attrs=["stations"])
        return True


    @_utilities.log_errors(logger)
    def load_velocity_models(self):
        """
        Parse and return velocity models from file.

        ROOT_RANK parses file and broadcasts contents to all other processes.
        """

        if RANK == ROOT_RANK:
            logger.info("Loading velocity models")

            velocity_models = _dataio.parse_velocity_models(self.cfg)
            self.pwave_model, self.swave_model = velocity_models
            self.step_size = self.pwave_model.step_size

            # Calculate & store latitude center for proper distance scaling
            minlat, _, _ = sph2geo(self.pwave_model.max_coords)
            maxlat, _, _ = sph2geo(self.pwave_model.min_coords)
            self._model_lat_center = (minlat + maxlat) / 2

        self.synchronize(attrs=["pwave_model", "swave_model", "step_size", "_model_lat_center"])
        return True


    @_utilities.log_errors(logger)
    @_utilities.root_only(RANK)
    def purge_raypaths(self):
        """
        Destroys all stored raypaths.
        """
        logger.debug(f"Purging raypath directory: {self.raypath_dir}")

        shutil.rmtree(self.raypath_dir)
        os.makedirs(self.raypath_dir)
        return True


    @_utilities.log_errors(logger)
    def relocate_events(self, method="DE", weightsonly=False):
        if not weightsonly:
            if method == "LINEAR":
                self._relocate_events_linear()
            elif method == "DE":
                self._relocate_events_de()
            else:
                raise (ValueError("Relocation method must be either 'linear' or 'DE'"))

        # after relocating events, update arrival residuals also
        self.update_arrival_residuals()

        # then update arrival and event KDE weights
        phase_order = self.cfg["algorithm"]["phase_order"]
        for phase in phase_order:
            self.update_arrival_weights(phase)

        self.update_event_weights()

        return True


    @_utilities.log_errors(logger)
    def _relocate_events_linear(self, niter_linloc=1):
        """
        Relocate all events based on linear inversion and update the "events" attribute.
        """
        raypath_dir = self.raypath_dir

        if RANK == ROOT_RANK:
            events = self.events.set_index("event_id")
            events["idx"] = range(len(events))
            arrivals = self.arrivals

            logger.info(f"Relocating {len(events)} events with linear inversion")

            for iter_loc in range(niter_linloc):
                column_idxs = np.array([],dtype=int)
                nonzero_values = np.array([],dtype=float)
                nsegments = np.array([],dtype=int)
                residuals = np.array([],dtype=float)

                for phase in self.phases:
                    if phase == "P":
                        model = self.pwave_model
                    elif phase == "S":
                        model = self.swave_model

                    arrivalssub = arrivals[arrivals["phase"]==phase]
                    arrivalssub = arrivalssub.set_index(["network", "station"])
                    idx_reloc = arrivalssub.index.unique()
                    for network, station in idx_reloc:

                        _arrivals = arrivalssub.loc[(network, station)]
                        _arrivals = _arrivals.set_index("event_id")

                        filename = f"{network}.{station}.{phase}.h5"
                        path = os.path.join(raypath_dir, filename)
                        if not os.path.isfile(path):
                            continue

                        with h5py.File(path, mode="r") as raypath_file:

                            for event_id, arrival in _arrivals.iterrows():

                                event = events.loc[event_id]
                                idx = int(event["idx"])

                                raypath = raypath_file[phase][:, idx]
                                raypath = np.stack(raypath).T
                                if (len(raypath)) < 7:
                                    logger.info(f"skipping raypath for eid {event_id}, {phase}, {network}.{station}")
                                    continue

                                dpos = np.zeros(3,)
                                dpos[0] = raypath[-2,0]-raypath[-1,0]
                                dpos[1] = raypath[-1,0]*(raypath[-2,1]-raypath[-1,1])
                                dpos[2] = raypath[-1,0]*(raypath[-2,2]-raypath[-1,2])*np.cos(raypath[-1,1])

                                dpos = dpos/np.sqrt(np.sum(dpos**2))
                                event_coords = events.loc[event_id, ["latitude", "longitude", "depth"]]
                                event_coords = geo2sph(event_coords)
                                vel_hypo = model.value(event_coords)
                                dtdx = np.zeros(4,)
                                dtdx[:-1] = dpos/vel_hypo # may be causing issue near ocean water boundaries
                                dtdx[-1] = 1.0
                                _column_idxs = np.arange(idx*4,idx*4+4)
                                _nonnzero_values = dtdx

                                column_idxs = np.append(column_idxs, _column_idxs)
                                nonzero_values = np.append(nonzero_values,_nonnzero_values)
                                nsegments = np.append(nsegments, len(_column_idxs))
                                residuals = np.append(residuals, arrival["residual"])

                row_idxs = [
                    i for i in range(len(nsegments))
                      for j in range(nsegments[i])
                ]
                row_idxs = np.array(row_idxs)

                ncol = (events["idx"].max()+1)*4

                Gmatrix = scipy.sparse.coo_matrix(
                    (nonzero_values, (row_idxs, column_idxs)),
                    shape=(len(nsegments), ncol)
                )

                # call lsmr for relocating
                # add three more parameters into cfg file,
                # "niter_linloc","damp_reloc" and "maxiter"
                atol    = self.cfg["relocate"]["atol"]
                btol    = self.cfg["relocate"]["btol"]
                conlim  = self.cfg["relocate"]["conlim"]
                damp    = self.cfg["relocate"]["damp"]
                maxiter = self.cfg["relocate"]["maxiter"]

                result = scipy.sparse.linalg.lsmr(
                    Gmatrix,
                    residuals,
                    damp = damp,
                    atol = atol,
                    btol = btol,
                    conlim = conlim,
                    maxiter = maxiter,
                    show=False
                )
                x, istop, itn, normr, normar, norma, conda, normx = result

                # change radians to degrees
                drad = x[::4]
                dlat = x[1::4]*180.0/(np.pi*(_constants.EARTH_RADIUS-events["depth"]))
                dlon = x[2::4]*180.0/(np.pi*(_constants.EARTH_RADIUS-events["depth"]))/np.cos(np.radians(events["latitude"]))
                dorigin = x[3::4]

                # update events
                events["latitude"] = events["latitude"]+dlat
                events["longitude"] = events["longitude"]-dlon
                events["depth"] = events["depth"]+drad
                events["time"] = events["time"]+dorigin
            events = events.reset_index()
            self.events = events

        self.synchronize(attrs=["events"])
        return True

    @_utilities.log_errors(logger)
    def _relocate_events_de(self):
        """
        Relocate all events and update the "events" attribute.
        """

        if RANK == ROOT_RANK:
            ids = self.events["event_id"]
            logger.info("Relocating %d events." % len(ids))
            self._dispatch(sorted(ids))

            # get info about the model bounds (here?) if we want to adjust delta to remain inside of them

            logger.debug("Dispatch complete. Gathering events.")
            # Gather and concatenate events from all workers.
            events = COMM.gather(None, root=ROOT_RANK)
            events = pd.concat(events, ignore_index=True)

            self.events = events

            if len(self.events) == 0:
                logger.error("Events DataFrame is empty! (did you over-filter?)")
                return False

        else:
            # Define columns to output.
            columns = ["latitude","longitude","depth","time","residual","event_id"]

            # Initialize EQLocator object.
            _path = self.traveltime_inventory_path
            _station_dict = station_dict(self.stations)

            with pykonal.locate.EQLocator(_path) as locator: ## fwiw the numpy error is in EQLocator

                # Create some aliases for configuration-file parameters.
                depth_min = self.cfg["relocate"]["depth_min"]
                dlat = self.cfg["relocate"]["dlat"]
                dlon = self.cfg["relocate"]["dlon"]
                dz = self.cfg["relocate"]["ddepth"]
                dt = self.cfg["relocate"]["dtime"]

                #for teleseismic events (only 1 arrival) we should limit what we can change here just to dt

                # Convert configuration-file parameters from geographic to spherical coordinates
                rho_max = _constants.EARTH_RADIUS - depth_min
                dtheta = np.radians(dlat)
                #dphi = np.radians(dlon) 
                dphi = np.radians(dlon) * np.cos(np.radians(self._model_lat_center)) # better scaled
                delta = np.array([dz, dtheta, dphi, dt])
                # slightly nonzero dlat and dlon for the quasi teleseisms.. within error anyway
                #   if zero, pykonal can have a tantrum,
                #     if too small then ||G|| zero. 1e-4 seems to work OK
                delta_tele = np.array([.1,.0001,.0001,dt])

                events = self.events
                events = events.set_index("event_id")
                relocated_events = pd.DataFrame()

                while True:
                    event_id = self._request_dispatch()

                    if event_id is None:
                        logger.debug("Received sentinel, gathering events.")
                        COMM.gather(relocated_events, root=ROOT_RANK)
                        break

                    logger.debug(f"Received event ID #{event_id}")

                    # Extract the initial event location and convert to
                    # spherical coordinates
                    _columns = ["latitude", "longitude", "depth", "time"]
                    initial = events.loc[event_id, _columns].values

                    initial[:3] = geo2sph(initial[:3])

                    locator.clear_arrivals()

                    # update EQLocator with arrivals for this event
                    _arrivals = arrival_dict(self.arrivals, event_id)
                    locator.add_arrivals(_arrivals)

                    # relocate the event
                    try:
                        if len(_arrivals) == 1:
                            # only let the teleseisms shift via time dimension since 1D (seems to be sensitive to the value. 1e-4 works)
                            loc = locator.locate(initial, delta_tele)
                            logger.info("locating TELESEISM")
                        else:
                            loc = locator.locate(initial, delta)

                    except Exception as e:
                        logger.debug(f"Location failed for event {event_id}: {str(e)}")
                        raise

                    # get residual RMS, reformat, append to relocated_events dataframe
                    loc[0] = min(loc[0], rho_max) # cap the depths
                    rms = locator.rms(loc)
                    loc[:3] = sph2geo(loc[:3])

                    event = pd.DataFrame(
                        [np.concatenate((loc, [rms, event_id]))],
                        columns=columns
                    )
                    relocated_events = pd.concat([relocated_events, event])

        self.synchronize(attrs=["events"])

        # TODO: would be useful to further log how catalog improves after relocating?

        return True


    @_utilities.log_errors(logger)
    def sanitize_data(self, for_res_test=False):
        """
        Clean up stations, events, and arrivals. Also adds necessary keys etc.
        """

        if RANK == ROOT_RANK:
            logger.info("Sanitizing data")

            # drop events where residual is NaN
            n0 = len(self.events)
            self.events = self.events.dropna(subset='residual')
            dn = n0 - len(self.events)
            if dn > 0:
                logger.info(f"Dropped {dn} NaNs from events (tends to happen if contains an arrival with NaN residual)")

            # drop events where weight is 0 (probably shouldn't be happening here)
            if 'weight' in self.events.columns:
                n0 = len(self.events)
                self.events = self.events[self.events['weight'] > 0]
                dn = n0 - len(self.events)
                if dn > 0:
                    logger.info(f"Dropped {dn} events with zero weights. {n0-dn} remain.")

            # drop arrivals where residual is NaN
            n0 = len(self.arrivals)
            self.arrivals = self.arrivals.dropna(subset='residual')
            dn = n0 - len(self.arrivals)
            if dn > 0:
                logger.info(f"Dropped {dn} NaNs from arrivals (arrivals get NaN residuals if too near model boundary)")

            # drop arrivals where weight is 0 (probably shouldn't be happening here)
            if 'weight' in self.arrivals.columns:
                n0 = len(self.arrivals)
                self.arrivals = self.arrivals[self.arrivals['weight'] > 0]
                dn = n0 - len(self.arrivals)
                if dn > 0:
                    logger.info(f"Dropped {dn} arrivals with zero weights. {n0-dn} remain.")

            # drop duplicate arrivals.
            keys = ["network", "station", "phase", "event_id"]
            n0 = len(self.arrivals)
            self.arrivals = self.arrivals.drop_duplicates(keys)
            dn = n0 - len(self.arrivals)
            if dn > 0:
                logger.info(f"Dropped {dn} duplicate arrivals {n0-dn} remain.")

            # drop duplicate stations.
            keys = ["network", "station"]
            n0 = len(self.stations)
            self.stations = self.stations.drop_duplicates(keys)
            dn = n0 - len(self.stations)
            if dn > 0:
                logger.info(f"Dropped {dn} duplicate stations {n0-dn} remain.")

            # drop duplicate events.
            keys = ["latitude", "longitude", "depth", "time"]
            n0 = len(self.events)
            self.events = self.events.drop_duplicates(keys)
            dn = n0 - len(self.events)
            if dn > 0:
                logger.info(f"Dropped {dn} duplicate events {n0-dn} remain.")


            # drop events outside of the velocity model
            velmodel = self.pwave_model
            minlat,maxlon,mindepth = sph2geo(velmodel.max_coords) # a little confusing but confirmed correct
            maxlat,minlon,maxdepth = sph2geo(velmodel.min_coords)

            events = self.events
            n0 = len(self.events)
            idx_keep = events[ (minlon <= events['longitude'])
                             & (events['longitude']<= maxlon)
                             & (minlat <= events['latitude'])
                             & (events['latitude']<= maxlat)].index
            self.events = events.loc[idx_keep]
            dn = n0 - len(self.events)
            if dn > 0:
                logger.info(f"Dropped {dn} events outside of velocity model. {n0-dn} remain.")

            # ..and drop stations outside of velocity model
            stations = self.stations
            n0 = len(self.stations)
            idx_keep = stations[ (minlon <= stations['longitude'])
                             & (stations['longitude']<= maxlon)
                             & (minlat <= stations['latitude'])
                             & (stations['latitude']<= maxlat)].index
            self.stations = stations.loc[idx_keep]
            dn = n0 - len(self.stations)
            if dn > 0:
                logger.info(f"Dropped {dn} stations outside of velocity model. {n0-dn} remain.")

            # drop stations outside of map_filter (if it exists)
            map_filter = self.cfg["model"]["map_filter"]
            if map_filter:
                map_min_lat, map_min_lon, map_max_lat, map_max_lon = map_filter
                stations = self.stations
                n0 = len(self.stations)
                idx_keep = stations[ (map_min_lon <= stations['longitude'])
                                 & (stations['longitude']<= map_max_lon)
                                 & (map_min_lat <= stations['latitude'])
                                 & (stations['latitude']<= map_max_lat)].index
                self.stations = stations.loc[idx_keep]
                dn = n0 - len(self.stations)
                if dn > 0:
                    logger.info(f"Dropped {dn} stations outside of map_filter. {n0-dn} remain.")

            # .... and drop arrivals linked to those dropped stations
            n0 = len(self.arrivals)
            stations_set = set(zip(self.stations['network'], self.stations['station']))
            arrival_mask = self.arrivals.apply(lambda x: (x['network'], x['station']) in stations_set, axis=1)
            self.arrivals = self.arrivals[arrival_mask]
            dn = n0 - len(self.arrivals)
            if dn > 0:
                logger.info(f"Dropped {dn} arrivals with stations outside velocity model. {n0-dn} remain.")

            #the extra metadata seems to be causing an issue (TODO)
            if 'mag' in self.events.keys() and 'sc_eqid' in self.events.keys():
                self.events.drop(['mag','sc_eqid'],axis=1,inplace=True)

            if not for_res_test:
                # drop events without minimum number of arrivals
                min_narrival = self.cfg["algorithm"]["min_narrival"]
                n0 = len(self.events)
                counts = self.arrivals["event_id"].value_counts()
                counts = counts[(counts >= min_narrival) | (counts == 1)] #allow singular teleseisms-as-synthetic-events to remain
                event_ids = counts.index
                self.events = self.events[self.events["event_id"].isin(event_ids)]
                dn = n0 - len(self.events)
                if dn > 0:
                    logger.info(f"Dropped {dn} events with < {min_narrival} arrivals. {n0-dn} remain.")

                # drop arrivals without events
                n0 = len(self.arrivals)
                bool_idx = self.arrivals["event_id"].isin(self.events["event_id"])
                self.arrivals = self.arrivals[bool_idx]
                dn = n0 - len(self.arrivals)
                if dn > 0:
                    logger.info(f"Dropped {dn} arrivals without associated events. {n0-dn} remain.")

            # drop arrivals out of desired distance range
            arrivals   = self.arrivals
            max_dist   = self.cfg["algorithm"]["max_dist"]
            min_dist   = self.cfg["algorithm"]["min_dist"]
            #dist_angle = self.cfg["algorithm"]["dist_angle"]
            cutoff_depth = self.cfg["algorithm"]["cutoff_depth"]

            #merge event data.
            events = self.events.rename(
                columns={
                    "latitude": "event_latitude",
                    "longitude": "event_longitude",
                    "depth": "event_depth"
                }
            )

            merge_columns = [
                "event_latitude",
                "event_longitude",
                "event_depth",
                "event_id"
            ]

            arrivals = arrivals.merge(events[merge_columns], on="event_id")

            #merge station data.
            stations = self.stations.rename(
                columns={
                    "latitude": "station_latitude",
                    "longitude": "station_longitude"
                }
            )

            merge_columns = [
                "station_latitude",
                "station_longitude",
                "network",
                "station"
            ]

            merge_keys = ["network", "station"]
            arrivals = arrivals.merge(stations[merge_columns], on=merge_keys)

            # apply distance filters if events are shallower than cutoff_depth
            dist = dist_km(arrivals["event_latitude"],arrivals["event_longitude"],
                                   arrivals["station_latitude"],arrivals["station_longitude"])
            arrivals["delta"] = dist

            idx_keep = arrivals[
                # within normal distance bounds
                ((arrivals['delta'] >= min_dist) & (arrivals['delta'] <= max_dist))
                | (arrivals['event_depth'] >= cutoff_depth)
            ].index

            n0 = len(self.arrivals)
            self.arrivals = arrivals.loc[idx_keep]
            dn = n0 - len(self.arrivals)
            if dn > 0:
                logger.info(f"Dropped {dn} arrivals outside of requested lateral range. {n0-dn} remain.")

            # drop events without arrivals
            n0 = len(self.events)
            bool_idx = self.events["event_id"].isin(self.arrivals["event_id"])
            self.events = self.events[bool_idx]
            self.events0 = self.events.copy() # also save a copy of the original to track total drift over time
            dn = n0 - len(self.events)
            if dn > 0:
                logger.info(f"Dropped {dn} events without associated arrivals. {n0-dn} remain.")

            # drop stations without arrivals
            n0 = len(self.stations)
            arrivals = self.arrivals.set_index(["network", "station"])
            idx_keep = arrivals.index.unique()
            stations = self.stations.set_index(["network", "station"])
            stations = stations.loc[idx_keep]
            stations = stations.reset_index()
            arrivals = arrivals.reset_index()
            self.stations = stations
            dn = n0 - len(self.stations)
            if dn > 0:
                logger.info(f"Dropped {dn} stations without associated arrivals. {n0-dn} remain.")

            # drop arrivals without stations
            n0 = len(self.arrivals)
            stations = self.stations.set_index(["network", "station"])
            idx_keep = stations.index.unique()
            arrivals = self.arrivals.set_index(["network", "station"])
            arrivals = arrivals.loc[idx_keep]
            arrivals = arrivals.reset_index()
            self.arrivals = arrivals
            dn = n0 - len(self.arrivals)
            if dn > 0:
                logger.info(f"Dropped {dn} arrivals without associated stations. {n0-dn} remain.")

            if len(self.stations) == 0:
                logger.error("All stations were dropped!!")
            if len(self.events) == 0:
                logger.error("All events were dropped!!")
            if len(self.arrivals) == 0:
                logger.error("All arrivals were dropped!!")


        self.synchronize(attrs=["stations", "events", "arrivals"])

        return True

    @_utilities.log_errors(logger)
    def resanitize_data(self, do_remove_outliers=True):
        """
        RE-Sanitize data as we iterate
        """

        if RANK == ROOT_RANK:
            logger.info("RE-sanitizing data")

            # Drop events where residual is NaN
            n0 = len(self.events)
            self.events = self.events.dropna(subset='residual')
            dn = n0 - len(self.events)
            if dn > 0:
                logger.info(f"Dropped {dn} NaNs from events (shouldn't be happening...)")

            # Drop events where weight is 0 (what causes this??)
            if 'weight' in self.events.columns:
                n0 = len(self.events)
                self.events = self.events[self.events['weight'] > 0]
                dn = n0 - len(self.events)
                if dn > 0:
                    logger.info(f"Dropped {dn} events with zero weights. {n0-dn} remain. (shouldn't happen!)")

            # Drop arrivals where residual is NaN (!)
            n0 = len(self.arrivals)
            self.arrivals = self.arrivals.dropna(subset='residual')
            dn = n0 - len(self.arrivals)
            if dn > 0:
                logger.info(f"Dropped {dn} NaNs from arrivals (shouldn't be happening...)")

            # Drop arrivals where weight is 0
            if 'weight' in self.arrivals.columns:
                n0 = len(self.arrivals)
                self.arrivals = self.arrivals[self.arrivals['weight'] > 0]
                dn = n0 - len(self.arrivals)
                if dn > 0:
                    logger.info(f"Dropped {dn} arrivals with zero weights. {n0-dn} remain. (shouldn't happen!)")

            # Drop events or arrivals with bad residuals
            if do_remove_outliers:
                max_evt_resid = self.cfg["algorithm"]["max_event_residual"]
                max_arr_resid = self.cfg["algorithm"]["max_arrival_residual"]
                n0 = len(self.arrivals)
                self.arrivals = remove_outliers(self.arrivals,None,"residual", max_arr_resid)
                if len(self.arrivals) < n0:
                    dn = n0 - len(self.arrivals)
                    logger.info(f"Dropped {dn} arrivals with residual > {max_arr_resid}. {n0-dn} remain.")
                n0 = len(self.events)
                self.events = remove_outliers(self.events,None,"residual",max_evt_resid)
                if len(self.events) < n0:
                    dn = n0 - len(self.events)
                    logger.info(f"Dropped {dn} events with residual > {max_evt_resid}. {n0-dn} remain.")

            # Drop events without minimum number of arrivals (should we be removing events that fall lower than min arrivals AFTER the initial check?)
            """
            min_narrival = self.cfg["algorithm"]["min_narrival"]
            n0 = len(self.events)
            counts = self.arrivals["event_id"].value_counts()
            counts = counts[(counts >= min_narrival) | (counts == 1)] # also let singular (e.g. teleseisms) pass. can probably also check by depth RCP
            event_ids = counts.index
            self.events = self.events[self.events["event_id"].isin(event_ids)]
            dn = n0 - len(self.events)
            if dn > 0:
                logger.info(f"Dropped {dn} events with < {min_narrival} arrivals. {n0-dn} remain.")
            """

            # Drop arrivals without events
            n0 = len(self.arrivals)
            bool_idx = self.arrivals["event_id"].isin(self.events["event_id"])
            self.arrivals = self.arrivals[bool_idx]
            dn = n0 - len(self.arrivals)
            if dn > 0:
                logger.info(f"Dropped {dn} arrivals without associated events. {n0-dn} remain.")

            # Drop events without arrivals 
            n0 = len(self.events)
            bool_idx = self.events["event_id"].isin(self.arrivals["event_id"])
            self.events = self.events[bool_idx]
            dn = n0 - len(self.events)
            if dn > 0:
                logger.info(f"Dropped {dn} events without associated arrivals. {n0-dn} remain.")


            if len(self.stations) == 0:
                logger.error("All stations were dropped!!") # I guess we aren't really testing stations for RE- but maybe we should be
            if len(self.events) == 0:
                logger.error("All events were dropped!!")
            if len(self.arrivals) == 0:
                logger.error("All arrivals were dropped!!")

        self.synchronize(attrs=["stations","events","arrivals"])

        return True


    @_utilities.log_errors(logger)
    @_utilities.root_only(RANK)
    def save_events(self):
        """
        Save the current "events", and "arrivals" to and HDF5 file using
        pandas.HDFStore.
        """
        logger.info(f"Saving event data from iteration #{self.iiter}")

        path = os.path.join(self.argc.output_dir, f"{self.iiter:02d}")

        events       = self.events
        EVENT_DTYPES = _constants.EVENT_DTYPES
        for column in EVENT_DTYPES:

            events[column] = events[column].astype(EVENT_DTYPES[column])

        arrivals       = self.arrivals
        ARRIVAL_DTYPES = _constants.ARRIVAL_DTYPES
        for column in ARRIVAL_DTYPES:
            arrivals[column] = arrivals[column].astype(ARRIVAL_DTYPES[column])

        events.to_hdf(f"{path}.events.h5", key="events")
        arrivals.to_hdf(f"{path}.events.h5", key="arrivals")

        return True


    @_utilities.log_errors(logger)
    @_utilities.root_only(RANK)
    def save_stations(self):
        """
        Save the current stations to and HDF5 file using
        pandas.HDFStore. Usually just the first!
        """
        logger.info(f"Saving filtered station data from iteration #{self.iiter}")

        path = os.path.join(self.argc.output_dir, f"{self.iiter:02d}")

        self.stations.to_hdf(f"{path}.stations.h5", key="stations")

        return True


    @_utilities.log_errors(logger)
    @_utilities.root_only(RANK)
    def save_model(self, phase, tag=None):
        """
        Save model data to disk for single phase.

        Return True upon successful completion.
        """
        logger.info(f"Saving {phase}-wave model for iteration #{self.iiter}")

        phase = phase.lower()
        path = os.path.join(self.argc.output_dir, f"{self.iiter:02d}")

        # save velocity
        handle = f"{phase}wave_model"
        label = f"{handle}.{tag}" if tag is not None else handle
        model = getattr(self, handle)
        model.to_hdf(path + f".{label}.h5")

        if self.iiter == 0:
            return True

        # save variance
        handle = f"{phase}wave_variance"
        model = getattr(self, handle)
        label = f"{handle}.{tag}" if tag is not None else handle
        model.to_hdf(path + f".{label}.h5")

        # iteration "quality"
        handle = f"{phase}wave_quality"
        model = getattr(self, handle)
        label = f"{handle}.{tag}" if tag is not None else handle
        model.to_hdf(path + f".{label}.h5")

        if self.argc.output_realizations is True:
            handle = f"{phase}wave_realization_stack"
            label = f"{handle}.{tag}" if tag is not None else handle
            stack = getattr(self, handle)
            with h5py.File(path + f".{label}.h5", mode="w") as f5:
                f5.create_dataset(f"{phase}wave_stack",data=stack[:])

            # also do the quality stack (new! not working? stupid?)
            handle = f"{phase}qual_realization_stack"
            label = f"{handle}.{tag}" if tag is not None else handle
            stack = getattr(self, handle)
            with h5py.File(path + f".{label}.h5", mode="w") as f5:
                f5.create_dataset(f"{phase}qual_stack",data=stack[:])

        return True


    @_utilities.log_errors(logger)
    def synchronize(self, attrs="all"):
        """
        Synchronize input data across all processes.
        'attrs' may be an iterable of attribute names to synchronize.
        """
        _all = (
            "arrivals",
            "arrival_history",
            "cfg",
            "events",
            "event_history",
            "projection_matrix",
            "pwave_model",
            "swave_model",
            "sampled_arrivals",
            "sampled_events",
            "stations",
            "step_size",
            "voronoi_cells"
        )

        if attrs == "all":
            attrs = _all

        for attr in attrs:
            value = getattr(self, attr) if RANK == ROOT_RANK else None
            value = COMM.bcast(value, root=ROOT_RANK)
            setattr(self, attr, value)

        COMM.barrier()

        return True


    @_utilities.log_errors(logger)
    def update_arrival_residuals(self, run_phases=None):
        """
        Compute arrival-time residuals based on current event locations
        and velocity models, and update "residual" columns of "arrivals"
        attribute.
        """
        arrivals = self.arrivals.set_index(["network", "station", "phase"])
        logger.info("Updating %d arrival residuals" % len(arrivals))
        arrivals = arrivals.sort_index()

        if RANK == ROOT_RANK:
            ids = arrivals.index.unique()
            self._dispatch(ids)
            logger.debug("Dispatch complete. Gathering arrivals.")
            arrivals = COMM.gather(None, root=ROOT_RANK)
            arrivals = pd.concat(arrivals, ignore_index=True)

            # sometimes NaNs sneak in as residuals,
            #  usually if the source or station is near or eclipses model boundary?
            n0 = len(arrivals)
            arrivals = arrivals.dropna(subset=['residual'])
            dn = n0 - len(arrivals)
            if dn > 0:
                logger.info(f"Dropped {dn} arrivals with NaN residuals (likely near/past model bounds)")

            self.arrivals = arrivals

        else:
            events = self.events.set_index("event_id")
            updated_arrivals = pd.DataFrame()

            last_handle = None
            processed = 0

            run_phases = run_phases or self.phases

            _path = self.traveltime_inventory_path
            with TraveltimeInventory(_path, mode="r") as traveltime_inventory:

                while True:
                    item = self._request_dispatch()

                    if item is None:
                        logger.debug("Received sentinel. Gathering arrivals.")
                        COMM.gather(updated_arrivals, root=ROOT_RANK)
                        break

                    network, station, phase = item
                    # we could skip past phases that aren't needed here?
                    if phase.upper() in run_phases:

                        handle = "/".join([network, station, phase])
                        if handle != last_handle:

                            traveltime = traveltime_inventory.read(handle)
                            last_handle = handle

                        _arrivals = arrivals.loc[(network, station, phase)]
                        _events = events.loc[_arrivals["event_id"].values]
                        arrival_times = _arrivals["time"].values

                        origin_times = _events["time"].values
                        coords = _events[["latitude", "longitude", "depth"]].values
                        coords = geo2sph(coords)
                        residuals = arrival_times - (origin_times + traveltime.resample(coords))
                        _arrivals = dict(
                            network=network,
                            station=station,
                            phase=phase,
                            event_id=_arrivals["event_id"].values,
                            arrival_id=_arrivals["arrival_id"].values,
                            time=arrival_times,
                            residual=residuals
                        )
                        _arrivals = pd.DataFrame(_arrivals)
                        updated_arrivals = pd.concat([updated_arrivals, _arrivals])

        self.synchronize(attrs=["arrivals"])

        return True


    @_utilities.log_errors(logger)
    def update_model(self, phase):
        """
        Obtain median and variance of model stackto update our model
        """
        logger.info(f"Running update_model for {phase}")
        phase = phase.lower()

        if RANK == ROOT_RANK:
            # Get slowness and quality stacks
            stack = getattr(self, f"{phase}wave_realization_stack")
            quality_stack = getattr(self, f"{phase}qual_realization_stack")

            # get our values. median is better than mean here, and allows for sharper features to be resolved
            variance = np.ma.var(stack,axis=0)
            delta_slowness = np.ma.median(stack,axis=0)

            # grab the model we're updating (which should be in velocity)
            model = getattr(self, f"{phase}wave_model")

            # hold onto the original copy to restore certain very low velocity (i.e. oceans or magma) areas
            orig_model = model.values.copy()
            watermask = model.values <= 0.2
            wateridx = np.where(watermask)

            # apply velocity guardrails here? +/- 20% or something? TODO

            # update model in slowness, then convert back to velocity
            values = np.power(model.values, -1) + delta_slowness
            velocities = np.power(values, -1)
            model.values = velocities
            # restore water velocity
            model.values[watermask] = orig_model[wateridx]


            # Update variance also (work back to get this in variance in VELOCITY)
            model = getattr(self, f"{phase}wave_variance")
            #model.values = variance # n.b. this is variance of SLOWNESS

            ## but what if we want to convert variance from slowness to velocity?
            # Var(1/s) ≈ Var(s) / s^4 = Var(s) * v^4
            slowness_values = np.power(velocities, -1)
            velocity_variance = variance * np.power(velocities, 4)

            # store variance as velocity variance (in (km/s)^2 -- take SQRT when interpreting!)
            model.values = velocity_variance

            # keep track of mean? median? so we can monitor throughout the iterations
            self._max_variance_km_s = np.mean( np.sqrt(velocity_variance) )
            logger.info(f"Mean {phase.upper()} velocity variance (km/s): {self._max_variance_km_s:0.6f}")

        self.synchronize(attrs=[f"{phase}wave_model"])
        return True


    @_utilities.log_errors(logger)
    def run_resolution_test(self):
        """
        Execute resolution test.
        """
        if not self.cfg["model"]["perform_res_test"]:
            return True

        need_to_load_data = False

        if RANK == ROOT_RANK:
            logger.info(">>>  Starting Resolution Test  <<<")

            # Check if we need to load data from files
            rerun_dir = self.cfg["model"].get("rerun_restest", "")
            need_to_load_data = (
                self.argc.test_only and
                rerun_dir and
                os.path.exists(rerun_dir) and
                rerun_dir != "." and
                not hasattr(self, '_data_loaded_from_current_state')
            )

            if need_to_load_data:
                logger.info(f"Loading existing results from {rerun_dir}")

                events_path, pmodel_path, smodel_path = _restesting._find_latest_files(rerun_dir)

                if not all([events_path, pmodel_path, smodel_path]):
                    logger.error("Could not find required files for resolution test")
                    return False

                # Load events and arrivals
                self.events = pd.read_hdf(events_path, key='events')
                self.arrivals = pd.read_hdf(events_path, key='arrivals')

                # Load velocity models using the same method as main loading
                _pwave_model = pykonal.fields.read_hdf(pmodel_path)
                _swave_model = pykonal.fields.read_hdf(smodel_path)

                # Convert to internal format (same as in load_velocity_models)
                self.step_size = _pwave_model.step_size
                self.pwave_model = _picklabel.ScalarField3D(coord_sys="spherical")
                self.swave_model = _picklabel.ScalarField3D(coord_sys="spherical")

                self.phases = self.cfg["algorithm"]["phase_order"]

                # Copy attributes
                for model, loaded_model in [(self.pwave_model, _pwave_model), 
                                           (self.swave_model, _swave_model)]:
                    model.min_coords = loaded_model.min_coords
                    model.node_intervals = loaded_model.node_intervals
                    model.npts = loaded_model.npts
                    model.values = loaded_model.values

                logger.info(f"Resolution test starting with {len(self.events)} events and {len(self.arrivals)} arrivals")

            else:
                logger.info("Using current inversion state for resolution test")

            # Parse resolution test parameters - fix the parameter name
            test_params = self.cfg["model"]["res_test_size_mag"]
            horiz_block_size_km = float(test_params[0])
            amplitude = float(test_params[1])

        else:
            horiz_block_size_km = None # yes, these need to be set for the rest of the workers
            amplitude = None

        need_to_load_data = COMM.bcast(need_to_load_data, root=ROOT_RANK)
        horiz_block_size_km = COMM.bcast(horiz_block_size_km, root=ROOT_RANK)
        amplitude = COMM.bcast(amplitude, root=ROOT_RANK)

        self.synchronize(attrs=["pwave_model", "swave_model", "step_size", "arrivals", "phases", "stations"])

        # thie is required as it preps station data. if we start saving station data, maybe could avoid
        if need_to_load_data:
            self.sanitize_data(for_res_test=True)

        # update events & arrivals (adds KDE weight)
        self.update_event_weights()
        self.synchronize(attrs=["events"])

        # run process
        for phase in self.phases:
            self._run_resolution_test_single_phase(phase, horiz_block_size_km, amplitude)
        return True

    @_utilities.log_errors(logger)
    def _run_resolution_test_single_phase(self, phase, horiz_block_size_km, amplitude):
        """
        Run resolution test per phase
        """

        if RANK == ROOT_RANK:
            logger.info(f"Running checkerboard test for {phase}")

            # Store original state
            original_arrivals = self.arrivals.copy()
            base_model = self.pwave_model if phase == 'P' else self.swave_model
            original_model = _restesting._copy_scalar_field(base_model)  # Make a deep copy

            # Create synthetic model and arrivals for both phases, regardless
            synthetic_model = _restesting._create_checkerboard_model(base_model,
                                                                     horiz_block_size_km,
                                                                     vertical_layers=self.cfg["model"]["res_test_layers"],
                                                                     amplitude=amplitude)
            logger.debug(f"Synthetic P&S models created with shape: {synthetic_model.values.shape}")

            # Replace our model with checkerboard
            if phase.upper() == 'P':
                self.pwave_model = synthetic_model
            else:
                self.swave_model = synthetic_model

        else:
            original_arrivals = None
            original_model = None
            synthetic_model = None

        # Sync synthetic model
        self.synchronize(attrs=[f"{phase.lower()}wave_model"])

        # Generate synth data and traveltimes with known locations
        self.compute_traveltime_lookup_tables(run_phases=[phase])
        self.update_arrival_residuals(run_phases=[phase])
        self.update_arrival_weights(phase)

        # Restore base model
        if RANK == ROOT_RANK:
            if phase == 'P':
                self.pwave_model = original_model
            else:
                self.swave_model = original_model

        # Re-sync to base model... also the arrivals again
        self.synchronize(attrs=[f"{phase.lower()}wave_model","arrivals"])

        # Run full inversion using same parameters as real inversion
        nreal = self.cfg["algorithm"]["nreal"]
        nvoronoi = self.cfg["algorithm"]["nvoronoi"]
        kvoronoi = int(nvoronoi * self.cfg["algorithm"]["kvoronoi"]/100)
        alpha = self.cfg["algorithm"]["paretos_alpha"]
        hvr = self.cfg["algorithm"]["hvr"]
        min_rays_per_cell = self.cfg["algorithm"]["min_rays_per_cell"]
        adaptive_weight = self.cfg["algorithm"].get("adaptive_data_weight", 0.0)
        adaptive_weight = min(adaptive_weight,1.0)

        # Reset stack and run multiple realizations
        self._reset_realization_stack(phase)

        for self.ireal in range(nreal):
            logger.info(f"{phase} RESOLUTION TEST realization {self.ireal+1}/{nreal}")

            # Use same sampling and stochastic variations as real inversion but don't QC the residuals-- they will be large!
            self._sample_events(do_remove_outliers=False)
            self._sample_arrivals(phase,do_remove_outliers=False)
            self._trace_rays(phase)

            # Add same stochastic variability to Voronoi cells
            mod_nvoronoi = int(nvoronoi * np.random.uniform(low=0.70, high=1.15)) # mostly dip lower
            mod_kvoronoi = min(kvoronoi, int(mod_nvoronoi * 0.7))

            self._generate_voronoi_cells(phase,mod_kvoronoi,mod_nvoronoi,alpha,adaptive_weight)
            self._compute_sensitivity_matrix(phase,hvr)
            self._update_projection_matrix(phase,hvr)
            self._compute_model_update(phase,min_rays=min_rays_per_cell)

        # Process stack using same method as real inversion
        self.update_model(phase)

        if RANK == ROOT_RANK:

            # now extract the final averaged model (not raw stack)
            recovered_model = _restesting._copy_scalar_field(self.pwave_model if phase == 'P' else self.swave_model)

            # analyze using original base model as reference
            metrics = _restesting._analyze_resolution(self, synthetic_model, recovered_model, phase, ref_model=original_model)

            # save results
            _restesting._save_results(
                self.argc.output_dir, synthetic_model, recovered_model,
                metrics, phase, horiz_block_size_km
            )

            # restore original arrivals
            self.arrivals = original_arrivals

        self.synchronize(attrs=["arrivals"])
        self.purge_raypaths()


# # # # # # # # # # # # # # # # # # # # # # # #

@_utilities.log_errors(logger)
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

    # if dataframe has only 1 item, it is converted to a Series
    # this ensures it remains a DataFrame (RCP)
    if not isinstance(dataframe,pd.DataFrame):
        dataframe = dataframe.to_frame().T

    # failsafe against weirdness or if stations have their start/end times set incorrectly
    # need to revisit first <=1 part, unclear if that ever happens normally
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


@_utilities.log_errors(logger)
def station_dict(dataframe):
    """
    Return a dictionary with network geometry suitable for passing to
    the EQLocator constructor.

    Returned dictionary has "station_id" keys, where "station_id" =
    f"{network}.{station}", and values are spherical coordinates of
    station locations.
    """

    if np.any(dataframe[["network", "station"]].duplicated()):
        raise (IOError("Multiple coordinates supplied for single station(s)"))

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

def remove_outliers(dataframe, tukey_k, column, max_resid=None):
    """
    Return DataFrame with outliers removed using Tukey fences.
    ALSO remove any arrival or event beyond maxresid (first)
    Note that "column" is always "residual" in our case
    """

    # toss max residuals for both arrivals and events
    if max_resid:
        dataframe = dataframe[
             (dataframe[column] <= max_resid)
            &(dataframe[column] >= -max_resid)]

    # don't Tukey the events
    if tukey_k and 'phase' not in dataframe.keys():
        q1, q3 = dataframe[column].quantile(q=[0.25, 0.75])
        iqr = q3 - q1
        vmin = q1 - tukey_k * iqr
        vmax = q3 + tukey_k * iqr
        dataframe = dataframe[
             (dataframe[column] >= vmin)
            &(dataframe[column] <= vmax)]

    return dataframe

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

    # optimized haversine
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

    # convert to radians
    phi1 = lat1 * _constants.DEG_TO_RAD
    phi2 = lat2 * _constants.DEG_TO_RAD

    # pre-compute trigonometric functions
    cos_phi1 = np.cos(phi1)
    cos_phi2 = np.cos(phi2)

    dlon = (lon2 - lon1) * _constants.DEG_TO_RAD
    dlat = (lat2 - lat1) * _constants.DEG_TO_RAD

    # use sine squared directly
    sin_dlat_2 = np.sin(0.5 * dlat)
    sin_dlon_2 = np.sin(0.5 * dlon)

    # optimized haversine
    a = sin_dlat_2 * sin_dlat_2 + cos_phi1 * cos_phi2 * sin_dlon_2 * sin_dlon_2
    a = np.minimum(a, 1.0)  # ensure a doesn't exceed 1 due to floating point errors

    return _constants.EARTH_RADIUS * 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


#no longer in use
def eq_angle(eq_distkm,eq_depth):
    """
    Returns the angle in degrees from station to event.
    primarily to reduce shallow events with crustal reflections 
    but still allow deep teleseismic events through
    """
    theta = np.arctan2(eq_distkm, eq_depth)
    return 90 - np.abs(np.degrees(theta))