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
from . import _constants
from . import _picklabel
from . import _utilities
from . import _restesting

# Get logger handle
logger = _utilities.get_logger(f"__main__.{__name__}")

# Define pykonal aliases
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
    A class providing core functionality PyVoroTomo i.e. the inversion process
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
        self._pwave_1d_stack = None
        self._swave_1d_stack = None
        self._pwave_variance = None
        self._swave_variance = None
        self._pwave_quality = None
        self._swave_quality = None
        self._pqual_realization_stack = None
        self._squal_realization_stack = None
        self._gradient_magnitude = None
        self._grid_coords = None
        self._prev_conda = None
        self._residuals = None
        self._residual_weights = None
        self._sensitivity_matrix = None
        self._stations = None
        self._step_size = None
        self._sampled_arrivals = None
        self._sampled_events = None
        self._voronoi_cells = None
        self._model_lat_center = 0


        if RANK == ROOT_RANK:
            scratch_dir = argc.scratch_dir
            self._scratch_dir_obj = tempfile.TemporaryDirectory(dir=scratch_dir)
            self._scratch_dir = self._scratch_dir_obj.name

            _tempfile = tempfile.TemporaryFile(dir=scratch_dir)
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

    @arrivals.setter
    def arrivals(self, value):
        self._arrivals = value

    @property
    def arrivals_history(self):
        return self._arrivals_history

    @arrivals_history.setter
    def arrivals_history(self, value):
        self._arrivals_history = value

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
    def niter(self):
        return self.cfg["algorithm"]["niter"]

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

    @pwave_quality.setter
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
    def residual_weights(self):
        return self._residual_weights

    @residual_weights.setter
    def residual_weights(self, value):
        self._residual_weights = value

    @property
    def gradient_magnitude(self):
        return self._gradient_magnitude

    @gradient_magnitude.setter
    def gradient_magnitude(self, value):
        self._gradient_magnitude = value

    @property
    def grid_coords(self):
        return self._grid_coords

    @grid_coords.setter
    def grid_coords(self, value):
        self._grid_coords = value

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

    @swave_quality.setter
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


    # # # # # # END PROPERTY INITS


    def _get_weight_blend(self):
        """
        Calculate solver weighting blend factor based on current iteration.

        Returns blend factor in [0, 1]:
            - 0.0 = no weighting (uniform weights)
            - 1.0 = full weighting

        Linearly interpolates from robust_weight_start to robust_weight_end
        over the course of iterations.
        """
        # Get config params with defaults (0 = disabled by default)
        start_blend = self.cfg["algorithm"].get("solver_weight_start", 0.3)
        end_blend = self.cfg["algorithm"].get("solver_weight_end", 0.9)

        if self.niter <= 1 or self.iiter < 1:
            return start_blend
        if end_blend <= 0:
            return 0

        # Linear ramp: 0 at iter 1, 1 at final iter
        progress = (self.iiter - 1) / (self.niter - 1)

        # Try SQRT scaling so it ramps up a bit faster at the start (nah, this is a bad idea)
        #progress = np.sqrt(np.clip(progress,0,1))

        blend = start_blend + progress * (end_blend - start_blend)

        return np.clip(blend, 0.0, 1.0)

    @_utilities.log_errors(logger)
    def _diagnose_damping_regime(self, weighted_sensitivity, weighted_residuals,
                                 x, damp_vector, nstation, nvoronoi,
                                 atol, btol, conlim, maxiter):
        """
        Tell the user whether damping is too high, too low, or about right.

        Strategy: re-solve the SAME augmented system at a near-zero damping
        level to estimate the misfit FLOOR -- the part of the residual that no
        amount of model freedom can remove (coherent noise + parameterization /
        ray-theory error). Compare the current misfit against that floor:

          - current misfit >> floor  -> damping is suppressing fittable signal
                                        (TOO HIGH)
          - current misfit ~= floor  -> damping isn't measurably hurting the fit;
                                        you're at/below the point where it matters
                                        (could go LOWER, no benefit to)
          - in between               -> regularizing without hurting the fit (OK)

        Also reports whether the problem is FLOOR-LIMITED: if even zero damping
        explains little of the data, the limit is the floor (data quality /
        parameterization), NOT damping -- so tuning damping is the wrong lever.

        Costs one extra LSMR solve. Call sparingly (e.g. realization 0 only).
        Returns a dict of the raw numbers; logging is done by the caller on root.
        """
        ncol = weighted_sensitivity.shape[1]

        # Near-zero-damping solve: scale the chosen per-cell vector down by 1e-3
        # so the SHAPE (and the sub-min_rays pins) are preserved but the level is
        # negligible. This isolates the floor without changing the parameterization.
        tiny = np.asarray(damp_vector, dtype=float) * 1e-3
        D0 = scipy.sparse.diags(np.concatenate([tiny, np.zeros(nstation)]), format='csr')
        aG = scipy.sparse.vstack([weighted_sensitivity, D0], format='csr')
        ad = np.concatenate([weighted_residuals, np.zeros(ncol)])
        x0 = scipy.sparse.linalg.lsmr(
            aG, ad, damp=0, atol=atol, btol=btol,
            conlim=conlim, maxiter=maxiter, show=False)[0]

        d_norm       = np.linalg.norm(weighted_residuals)
        misfit_now   = np.linalg.norm(weighted_sensitivity @ x  - weighted_residuals)
        misfit_floor = np.linalg.norm(weighted_sensitivity @ x0 - weighted_residuals)

        # fraction of current misfit that exists BECAUSE of damping (vs the floor)
        damp_excess = (misfit_now - misfit_floor) / (misfit_floor + 1e-30)
        var_red       = 1.0 - (misfit_now   / (d_norm + 1e-30)) ** 2
        var_red_floor = 1.0 - (misfit_floor / (d_norm + 1e-30)) ** 2

        return dict(d_norm=d_norm, misfit_now=misfit_now, misfit_floor=misfit_floor,
                    damp_excess=damp_excess, var_red=var_red, var_red_floor=var_red_floor)


    def _ascii_lcurve(self, xx, yy, corner_idx=None, width=100, height=30):
        """
        Render an ASCII L-curve to a list of strings suitable for logger.info.
        xx, yy are 1-D arrays (already log-transformed).
        corner_idx is the index of the detected corner, marked with '*'.
        """
        if len(xx) == 0:
            return ["(no points)"]
        # Normalize to plot grid
        x_min, x_max = xx.min(), xx.max()
        y_min, y_max = yy.min(), yy.max()
        x_range = max(x_max - x_min, 1e-12)
        y_range = max(y_max - y_min, 1e-12)
        grid = [[" "] * width for _ in range(height)]
        # Place each point. y axis is inverted (row 0 = top).
        for i, (x, y) in enumerate(zip(xx, yy)):
            col = int(round((x - x_min) / x_range * (width - 1)))
            row = int(round((1 - (y - y_min) / y_range) * (height - 1)))
            col = max(0, min(width - 1, col))
            row = max(0, min(height - 1, row))
            marker = "X" if i == corner_idx else "."
            grid[row][col] = marker
        # Frame + axis labels
        lines = []
        lines.append(f"  log||m|| = {y_max:.2f}")
        lines.append("    +" + "-" * width + "+")
        for row in grid:
            lines.append("    |" + "".join(row) + "|")
        lines.append("    +" + "-" * width + "+")
        lines.append(f"  log||m|| = {y_min:.2f}    "
                     f"log||r||: {x_min:.2f} .. {x_max:.2f}")
        return lines


    @_utilities.log_errors(logger)
    def _lcurve_corner_level(self, weighted_sensitivity, weighted_residuals,
                             base_damp, shape, nvoronoi, nstation,
                             atol, btol, conlim, maxiter, levels=None):
        """
        Choose the damping LEVEL from the L-curve sweep.

        Take the KNEE -- the onset of the misfit rise, i.e. the
        smallest level at which damping first costs more than `tol` of misfit
        over the floor. If misfit never rises (well-determined / floor-limited),
        fall back to the lowest level. `strength` is kept only as a diagnostic;
        it no longer drives the choice, because on plateau+gentle-rise curves the
        max-distance "corner" drifts into the over-damped arm.

        Returns (level, strength, has_corner, misfits, norms, levels).
        """
        if levels is None:
            levels = np.geomspace(1e-6, 1.0, 100)

        sigma_station  = self.cfg["algorithm"].get("sigma_station")
        lambda_station = 1.0 / (sigma_station + 1e-9)

        norms   = np.empty(len(levels))
        misfits = np.empty(len(levels))
        for i, lev in enumerate(levels):
            d  = lev * base_damp * shape
            vd = scipy.sparse.diags(d, format='csr')
            sd = scipy.sparse.diags(np.full(nstation, lambda_station), format='csr')
            fd = scipy.sparse.block_diag([vd, sd])
            aG = scipy.sparse.vstack([weighted_sensitivity, fd])
            ad = np.concatenate([weighted_residuals, np.zeros(nvoronoi + nstation)])
            xs = scipy.sparse.linalg.lsmr(aG, ad, damp=0, atol=atol, btol=btol,
                                          conlim=conlim, maxiter=maxiter, show=False)[0]
            norms[i]   = np.linalg.norm(xs[:nvoronoi])
            misfits[i] = np.linalg.norm(weighted_sensitivity @ xs - weighted_residuals)

        xx = np.log(misfits + 1e-30)
        yy = np.log(norms   + 1e-30)

        # --- corner-likeness GATE (not a locator): max perpendicular distance
        #     from the endpoint chord, normalized by the box diagonal. ~0.005 for
        #     a smooth ramp; >~0.2 only for a genuine localized elbow. ---
        x0, y0, x1, y1 = xx[0], yy[0], xx[-1], yy[-1]
        chord = np.hypot(x1 - x0, y1 - y0) + 1e-30
        diag  = np.hypot(np.ptp(xx), np.ptp(yy)) + 1e-30
        dist  = np.abs((y1 - y0) * xx - (x1 - x0) * yy + x1 * y0 - y1 * x0) / chord
        strength   = float(dist.max() / diag)
        corner_idx = int(np.argmax(dist))

        # --- onset-of-rise knee: stable, light damping (the usual outcome) ---
        mmin = misfits.min()
        tol  = self.cfg["algorithm"].get("lcurve_misfit_tol", 0.003) # probably don't need to add this param but TODO
        over = np.where(misfits > mmin * (1.0 + tol))[0]
        knee_idx = int(over[0]) if over.size else 0

        # --- decision ---
        x_span     = float(np.ptp(xx))
        corner_thr = self.cfg["algorithm"].get("lcurve_corner_strength", 0.15)
        if x_span < 0.02:
            used_idx, regime, has_corner = 0, "floor-limited (misfit flat)", False
        elif strength >= corner_thr:
            used_idx, regime, has_corner = corner_idx, \
                f"REAL corner (strength {strength:.3f})", True
        else:
            used_idx, regime, has_corner = knee_idx, \
                f"no corner (strength {strength:.3f} < {corner_thr}) -> onset knee", False
        level = float(levels[used_idx])

        if RANK == ROOT_RANK:
            logger.info(f"  L-curve: {regime}; level={level:.4g} (idx {used_idx}); "
                        f"knee idx {knee_idx}, max-dist idx {corner_idx}, "
                        f"strength {strength:.4f}")
            # print the ascii chart (fun!)
            for line in self._ascii_lcurve(xx, yy, corner_idx=used_idx):
                logger.info(line)
            # print all the chart values (change from debug to info to see ALL values)
            for i, lev in enumerate(levels):
                is_used = i == used_idx
                marker = " X" if is_used else "  "
                log = logger.info if is_used else logger.debug
                log(f"  {marker} lev={lev:.4g}  ||r||={misfits[i]:.4e} ||m||={norms[i]:.4e}")            
        return level, strength, has_corner, misfits, norms, levels


    @_utilities.log_errors(logger)
    @_utilities.root_only(RANK)
    def _compute_model_update(self, phase, min_rays=3, use_weights=True, compute_1d=False):
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
            raise (ValueError(f"Unrecognized phase ({phase}) supplied"))
        
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

        # Apply residual-based weighting
        solver_weight_method = self.cfg["algorithm"].get("solver_weight_method", "huber")
        solver_weight_tuning = self.cfg["algorithm"].get("solver_weight_tuning", -1)
        solver_weight_blend_factor = self._get_weight_blend()

        if use_weights and solver_weight_blend_factor > 0:
            # Compute robust weights
            raw_weights = _utilities.compute_residual_weights(
                self.residuals, 
                method=solver_weight_method, 
                tuning_param=solver_weight_tuning
            )

            # Blend with uniform weights
            weights = _utilities.blend_weights(raw_weights, solver_weight_blend_factor)
            self.residual_weights = weights

            # Log statistics
            min_weight = np.min(raw_weights)
            mean_weight = np.mean(raw_weights)
            std_weight = np.std(raw_weights)
            n_downweighted = np.sum(raw_weights < 0.999) #slightly less than 1.0 since there are small epsilon values at play
            logger.info(f" Arrival weighting: blend: {solver_weight_blend_factor:.2f}, mean(std): {mean_weight:.2f} ({std_weight:.2f}), min: {min_weight:.3f}, "
                f"% weighted (total): {n_downweighted/len(raw_weights)*100:.1f}% ({len(raw_weights)})")

            # Apply weights: W^(1/2) * G * x = W^(1/2) * d
            sqrt_weights = np.sqrt(weights)
            weight_matrix = scipy.sparse.diags(sqrt_weights)
            weighted_sensitivity = weight_matrix @ self.sensitivity_matrix
            weighted_residuals = sqrt_weights * self.residuals
        else:
            weighted_sensitivity = self.sensitivity_matrix
            weighted_residuals = self.residuals
            self.residual_weights = np.ones(len(self.residuals))


        # Perform automatic damping if damp < 0 (probably should be the default/only option)
        damp = self.cfg["algorithm"]["damp"]

        if damp <= -10: # coarse bulk damping 
            # Use scipy's auto-damping heuristic + iteration scaling.
            # ||G||_F is a reliable scale for the matrix; small fraction of that is
            # the canonical Tikhonov damping anchor.
            G_scale = scipy.sparse.linalg.norm(weighted_sensitivity, ord='fro')

            # Anchor: ||G||_F over the sqrt(min dim) is a defensible default.
            base_damp = G_scale / np.sqrt(min(weighted_sensitivity.shape))

            # Scale by current data misfit relative to initial -- want MORE damping
            # as the model converges (smaller updates per iteration).
            # 1st iteration is only 90%, but grows from there
            #if self.iiter > 0: # e.g. normal run
            #    iter_mult = 1.0 + 0.05 * (self.iiter-2)
            #else:
            #    iter_mult = 1.0 # res tests have iter == 0
            iter_mult = 1.0

            damp = base_damp * iter_mult

            if RANK == ROOT_RANK:
                logger.info(f"  auto damp: G_scale={G_scale:.2f} | base damp: {base_damp:.4f} * iter_mult: {iter_mult:.2f} --> effective damp: {damp:.4f}")

            result = scipy.sparse.linalg.lsmr(
                weighted_sensitivity, weighted_residuals, 
                damp=damp, atol=atol, btol=btol, conlim=conlim, maxiter=maxiter, show=False
            )
            x, istop, itn, normr, normar, norma, conda, normx = result

        elif damp < 0: # per-cell damping & L-curve corner calculation for the first 10 realizations
            df = pd.DataFrame({'col': sensitivity_coo.col, 'data': np.abs(sensitivity_coo.data)})
            target_rays = self.cfg["meshing"].get("target_rays_per_cell", 25) # 0 below min_rays, ramping to 1 by some "fully resolved" threshold
            norm_ray_count = np.clip(ray_counts / target_rays, 0, 1) # 0 below min_rays/cell, gradient to target_rays per cell
            cell_quality = norm_ray_count # should be from 0 to 1

            if RANK == ROOT_RANK:
                used_mask = ray_counts > 0
                used_quality = cell_quality[used_mask]
                percentiles = np.percentile(used_quality, [10, 25, 50, 75, 90])
                logger.info(f" Cell quality distribution  "
                            f"10%: {percentiles[0]:.2f}, 25%: {percentiles[1]:.2f}, "
                            f"50%: {percentiles[2]:.2f}, 75%: {percentiles[3]:.2f}, 90%: {percentiles[4]:.2f}")

            G_scale = scipy.sparse.linalg.norm(weighted_sensitivity, ord='fro')
            base_damp = G_scale / np.sqrt(min(weighted_sensitivity.shape))
            shape = 0.5 + 1.5 * (1 - cell_quality)        # per-cell multiplier?
            #shape = np.ones(nvoronoi) # just set the same? safe option

            # Self-appoint the overall LEVEL by L-curve corner. For the first
            # N_SEARCH realizations of each iteration we run the full corner
            # search; for the rest we reuse the median corner found so far.
            # base_damp and the per-cell shape are always rebuilt fresh; only the
            # dimensionless LEVEL multiplier is shared across realizations.
            total_cols = weighted_sensitivity.shape[1]
            nstation = total_cols - nvoronoi
            N_SEARCH = 9

            # Reset the accumulator at the start of each iteration (epoch)
            epoch_key = (self.iiter,phase)
            if getattr(self, "_level_epoch", None) != epoch_key:
                self._level_epoch = epoch_key
                self._level_samples = []

            if self.ireal < N_SEARCH or not self._level_samples:
                level, strength, has_corner, misfits, norms, levels = \
                    self._lcurve_corner_level(
                        weighted_sensitivity, weighted_residuals,
                        base_damp, shape, nvoronoi, nstation,
                        atol, btol, conlim, maxiter)
                self._level_samples.append(level)
                searched = True
            else:
                level = float(np.median(self._level_samples))
                searched = False
                has_corner = None
                strength = None

            damp = level * base_damp * shape

            if RANK == ROOT_RANK:
                if searched:
                    corner_txt = (f"L-curve CORNER (strength {strength:.3f})"
                                  if has_corner else
                                  f"no corner (strength {strength:.3f}<thr) -> guessing knee")
                    logger.info(f"  self-appointed level {level:.5f} x base_damp {base_damp:.3f} "
                                f"[{corner_txt}; epoch median {np.median(self._level_samples):.3f} "
                                f"over {len(self._level_samples)}]")
                else:
                    logger.info(f"  damping: reused epoch-median level {level:.5f} x base_damp {base_damp:.3f} "
                                f"(median over {len(self._level_samples)} searched realizations)")

            # Create damping matrix for Voronoi cells only
            voronoi_damping = scipy.sparse.diags(damp, format='csr')

            # Optionally add station terms in the solver
            sigma_station = self.cfg["algorithm"].get("sigma_station") # default is -1
            if sigma_station < 0: # station terms are free
                station_damping = scipy.sparse.diags(np.zeros(nstation), format='csr')
            else:
                # applying some constraints. small numbers = trust that stations correct
                lambda_station = 1.0 / (sigma_station + 1e-9)    # in slowness to be in the same units cells
                station_damping = scipy.sparse.diags(np.full(nstation, lambda_station), format='csr')

            # Combine into full damping matrix matching sensitivity matrix columns
            full_damping = scipy.sparse.block_diag([voronoi_damping, station_damping])

            # Augment the (weighted!) system: [G; D] * x = [d; 0]
            augmented_G = scipy.sparse.vstack([weighted_sensitivity, full_damping])
            augmented_d = np.concatenate([weighted_residuals, np.zeros(nvoronoi + nstation)])

            result = scipy.sparse.linalg.lsmr(
                augmented_G, augmented_d,
                damp=0, atol=atol, btol=btol, conlim=conlim, maxiter=maxiter, show=False
            )
            x, istop, itn, normr, normar, norma, conda, normx = result

            station_terms = x[nvoronoi:]
            logger.info(f"  station term magnitudes: mean|x|={np.mean(np.abs(station_terms)):.3f}, "
                        f"max|x|={np.max(np.abs(station_terms)):.3f}, "
                        f"||stations||={np.linalg.norm(station_terms):.3f}")

            # Capture the per-cell damping vector BEFORE collapsing it to a scalar,
            # so the diagnostic (and the 1D path) can use it.
            damp_vector = np.atleast_1d(np.asarray(damp, dtype=float)).copy()

            # Damping-regime diagnostic: re-solve at ~zero damping to estimate the
            # misfit floor, and report whether damping is too high, too low, or OK.
            # One extra solve; run on realization 0 only. Needs the full per-cell
            # vector, so it must come before the np.mean() collapse below.
            if self.ireal == 0 and len(damp_vector) == nvoronoi:
                reg = self._diagnose_damping_regime(
                    weighted_sensitivity, weighted_residuals, x, damp_vector,
                    nstation, nvoronoi, atol, btol, conlim, maxiter)
                if RANK == ROOT_RANK:
                    if reg["damp_excess"] > 0.25:
                        verdict = "TOO HIGH -- damping adds >10% misfit over the floor"
                    elif reg["damp_excess"] < 0.01:
                        verdict = "LOW side -- in the flat zone; damping barely affects fit"
                    else:
                        verdict = "OK -- regularizing without measurably hurting the fit"
                    logger.info(f"  damping regime: {verdict}")
                    logger.info(
                        f"    var.reduction {reg['var_red']:.1%} (floor {reg['var_red_floor']:.1%}) | "
                        f"misfit {reg['misfit_now']:.1f} vs floor {reg['misfit_floor']:.1f} | "
                        f"excess {reg['damp_excess']:.1%}")
                    if reg["var_red_floor"] < 0.5:
                        logger.warning(
                            f"    NOTE: even at ~zero damping only {reg['var_red_floor']:.0%} "
                            f"variance reduction -- FLOOR-limited (data/parameterization), "
                            f"not damping-limited; tuning damp won't help   ###")

            # Per-cell damping distribution over valid cells (uses the vector
            # captured above, before the scalar collapse below).
            #if RANK == ROOT_RANK and len(damp_vector) == nvoronoi and np.any(valid_cells):
            #    p = np.percentile(damp_vector[valid_cells], [10, 25, 50, 75, 90])
            #    logger.info(
            #        f"  per-cell damp (valid only)  10%:{p[0]:.3f} 25%:{p[1]:.3f} "
            #        f"50%:{p[2]:.3f} 75%:{p[3]:.3f} 90%:{p[4]:.3f}  | "
            #        f"{int(valid_cells.sum())}/{nvoronoi} valid")

            # Collapse to a representative scalar for the 1D path and logging.
            damp = np.mean(damp)

        else:
            result = scipy.sparse.linalg.lsmr(
                weighted_sensitivity, weighted_residuals, 
                damp=damp, atol=atol, btol=btol, conlim=conlim, maxiter=maxiter, show=False
            )
            x, istop, itn, normr, normar, norma, conda, normx = result


        # Set cells with insufficient rays to ZERO which nulls their contribution
        x[:nvoronoi][ray_counts < min_rays] = 0

        logger.info(f"  ||G||         = {norma:8.2f}  (sensitivity matrix mag)")
        logger.info(f"  ||Gm-d||      = {normr:8.2f}  (residual norm)")
        logger.info(f"  ||m||         = {normx:8.3f}  (solution norm)")
        logger.info(f"  ||G^-1||||G|| = {conda:8.2f}  (condition estimate)")
        logger.info(f"  med(std) G,r  = {np.median(self.sensitivity_matrix.data):.3f}({np.std(self.sensitivity_matrix.data):.3f}), {np.median(self.residuals.data):.3f}({np.std(self.residuals.data):.3f})")
        logger.info(f"  used or median damp   = {damp:.3f}")
        logger.info(f"  used/nvoronoi = {np.sum(ray_counts >= min_rays):.0f}/{nvoronoi:.0f}  ({itn} LSMR iterations)")

        delta_slowness = self.projection_matrix * x[:nvoronoi]
        delta_slowness = delta_slowness.reshape(model.npts)

        # Compute/log quality metrics
        ray_coverage = np.zeros(nvoronoi)
        ray_coverage[ray_counts >= min_rays] = ray_counts[ray_counts >= min_rays] / np.max(ray_counts)
        local_quality_coverage = self.projection_matrix * ray_coverage
        local_quality_coverage = local_quality_coverage.reshape(model.npts)

        """
        # Increases with decreasing residual norm
        quality_normr = 1.0 / (normr + 1e-8)
        # Increases with decreasing condition number
        quality_conda = 1.0 / (conda + 1e-8)
        # Raypath coverage ratio relative to number of cells
        quality_coverage = np.sum(ray_counts >= min_rays) / nvoronoi

        # Combine global factors into a single global quality score (you can adjust weights)
        global_quality = (quality_normr * 0.4 +
                          quality_conda * 0.4 +
                          quality_coverage * 0.2)
        """

        # Add to our solution stack
        if phase == "P":
            self.pwave_realization_stack[self.ireal] = delta_slowness
            #self.pqual_realization_stack[self.ireal] = global_quality # not sure this is working. seems to be the same as pwave?
        else:
            self.swave_realization_stack[self.ireal] = delta_slowness
            #self.squal_realization_stack[self.ireal] = global_quality
        
        # Also calculate a 1D model?
        if compute_1d:
            nz = model.npts[0]

            # Note that using the same 3D damp. Can either do a full recalc (TODO) or fudge assume fine
            damp_1d = damp

            # Lazy init of the 1D realization stack
            stack_attr = f"_{phase.lower()}wave_1d_stack"
            if getattr(self, stack_attr) is None:
                nreal = self.cfg["algorithm"]["nreal"]
                setattr(self, stack_attr, np.full((nreal, nz), np.nan))

            # Map each Voronoi cell to its nearest depth bin (radial node index)
            bin_indices = np.round(
                (self.voronoi_cells[:, 0] - model.min_coords[0])
                / model.node_intervals[0]
            ).astype(int)
            bin_indices = np.clip(bin_indices, 0, nz - 1)

            # Aggregation matrix: (nvoronoi, nz)
            # Columns of sensitivity_voronoi that share the same depth bin
            # are summed together to form the 1D sensitivity column.
            agg = scipy.sparse.coo_matrix(
                (np.ones(nvoronoi), (np.arange(nvoronoi), bin_indices)),
                shape=(nvoronoi, nz)
            ).tocsr()

            # 1D sensitivity matrix: (nray, nz)
            # Reuse the already-sliced sensitivity_voronoi matrix
            sensitivity_1d = sensitivity_voronoi @ agg

            # Apply the same residual weights 3D
            if use_weights and solver_weight_blend_factor > 0:
                weighted_sensitivity_1d = weight_matrix @ sensitivity_1d
            else:
                weighted_sensitivity_1d = sensitivity_1d

            # Solve — use scalar damp (mean of per-cell damp if auto, else as-is)
            result_1d = scipy.sparse.linalg.lsmr(
                weighted_sensitivity_1d, weighted_residuals,
                damp=damp_1d, atol=atol, btol=btol, conlim=conlim,
                maxiter=maxiter, show=False
            )
            x_1d, istop_1d, itn_1d, normr_1d, normar_1d, norma_1d, conda_1d, normx_1d = result_1d

            logger.info(f"  1D ||G||         = {norma_1d:8.2f}  (sensitivity matrix mag)")
            logger.info(f"  1D ||Gm-d||      = {normr_1d:8.2f}  (residual norm)")
            logger.info(f"  1D ||m||         = {normx_1d:8.3f}  (solution norm)")
            logger.info(f"  1D ||G^-1||||G|| = {conda_1d:8.2f}  (condition estimate)")
            logger.info(f"  1D est/used damp = {np.std(weighted_residuals)/conda_1d:.4e} / {damp_1d:.4e}")
            logger.info(f"  1D # iterations  = {itn_1d}")

            stack_1d = getattr(self, stack_attr)
            stack_1d[self.ireal] = x_1d

        return True


    def _raypath_turning_depth_km(self, raypath):
        """
        Return the depth (km) of the turning point (minimum rho) of a raypath.
        raypath: np.ndarray of shape (N, 3), columns = [rho, theta, phi]
        """
        min_rho = raypath[:, 0].min()
        return _constants.EARTH_RADIUS - min_rho


    @_utilities.log_errors(logger)
    def _compute_sensitivity_matrix(self, phase, hvr):
        """
        Compute the sensitivity matrix
        """

        logger.debug(f"Computing {phase}-wave sensitivity matrix")

        raypath_dir = self.raypath_dir

        index_keys = ["network", "station"]
        arrivals = self.sampled_arrivals.set_index(index_keys)

        arrivals = arrivals.sort_index()

        stationused = self.sampled_arrivals[index_keys]
        stationused = stationused.drop_duplicates().reset_index() # probably don't need to drop duplicates again TODO
        stationused['idx'] = range(len(stationused))
        stationused = stationused.set_index(index_keys)
        nstation = int(stationused['idx'].max() + 1)

        # raypath bottom filter
        raypath_filter_min,raypath_filter_max = self.cfg["algorithm"].get("raypath_bottom_mask", [-1,-1])
        num_raypath_bottom_filtered = 0

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

            # Diagnostic: ray coverage statistics
            sensitivity_voronoi = matrix.tocsr()[:, :nvoronoi]
            coo = sensitivity_voronoi.tocoo()

            # Count unique rays (rows) per cell (column)
            ray_cell_pairs = np.unique(np.column_stack([coo.row, coo.col]), axis=0)
            unique_rays_per_cell = np.bincount(ray_cell_pairs[:, 1], minlength=nvoronoi)

            cells_with_rays = unique_rays_per_cell > 0
            n_cells_with_rays = cells_with_rays.sum()
            n_empty_cells = nvoronoi - n_cells_with_rays
            avg_rays_per_sampled_cell = unique_rays_per_cell[cells_with_rays].mean() if n_cells_with_rays > 0 else 0

            logger.info(
                f" Ray coverage: {n_cells_with_rays}/{nvoronoi} cells ({n_cells_with_rays/nvoronoi*100:.1f}%), "
                f"avg rays-per-sampled-cell: {avg_rays_per_sampled_cell:.1f}"
            )

        else:

            nvoronoi = len(self.voronoi_cells)
            step_size = self.step_size
            events = self.events.set_index("event_id")
            events["idx"] = range(len(events))

            # Build once, reuse for all raypaths this realization
            if phase == 'P':
                min_coords = self.pwave_model.min_coords
                max_coords = self.pwave_model.max_coords
            else:
                min_coords = self.swave_model.min_coords
                max_coords = self.swave_model.max_coords
            center = (min_coords + max_coords) / 2
            _cells = center + (self.voronoi_cells - center) / [1, hvr, hvr]
            _voronoi_tree = cKDTree(sph2xyz(_cells))

            column_idxs_list = []
            nsegments_list = []
            nonzero_values_list = []
            residuals_list = []

            while True:
                item = self._request_dispatch()

                if item is None:
                    logger.debug("Sentinel received. Gathering sensitivity matrix.")

                    column_idxs    = np.concatenate(column_idxs_list)    if column_idxs_list    else np.array([], dtype=_constants.DTYPE_INT)
                    nsegments      = np.array(nsegments_list,             dtype=_constants.DTYPE_INT)
                    nonzero_values = np.concatenate(nonzero_values_list)  if nonzero_values_list else np.array([], dtype=_constants.DTYPE_REAL)
                    residuals      = np.array(residuals_list,             dtype=_constants.DTYPE_REAL)

                    COMM.gather(column_idxs,    root=ROOT_RANK)
                    COMM.gather(nsegments,      root=ROOT_RANK)
                    COMM.gather(nonzero_values, root=ROOT_RANK)
                    COMM.gather(residuals,      root=ROOT_RANK)
                    break

                network, station = item
                _arrivals = arrivals.loc[[(network, station)]]
                _arrivals = _arrivals.set_index("event_id")
                station_idxs = stationused['idx'].loc[[(network, station)]] + nvoronoi

                filename = f"{network}.{station}.{phase}.h5"
                path = os.path.join(raypath_dir, filename)

                with h5py.File(path, mode="r") as raypath_file:
                    for event_id, arrival in _arrivals.iterrows():
                        event = events.loc[event_id]
                        idx = int(event["idx"])

                        raypath = raypath_file[phase][:, idx]
                        raypath = np.stack(raypath).T

                        # Avoid adding any raypaths bottoming between raypath_filter_min and max
                        #  we should avoid including event origins within this range also
                        if raypath_filter_min < raypath_filter_max:
                            if not (raypath_filter_min < event["depth"] < raypath_filter_max):
                                raypath_bottom = self._raypath_turning_depth_km(raypath)
                                if raypath_filter_min < raypath_bottom < raypath_filter_max:
                                    num_raypath_bottom_filtered += 1
                                    continue

                        if len(raypath) < 1:
                            logger.warning("raypath is 0??")

                        _column_idxs, counts = self._projected_ray_idxs(raypath, hvr, _voronoi_tree, center) # now passing pre-computed voronoi
                        _column_idxs = np.append(_column_idxs, station_idxs)

                        column_idxs_list.append(_column_idxs)
                        nsegments_list.append(len(_column_idxs))
                        nonzero_values_list.append(np.append(counts * step_size, 1))
                        residuals_list.append(arrival["residual"])

        total_filtered = COMM.reduce(num_raypath_bottom_filtered, op=MPI.SUM, root=ROOT_RANK)
        if RANK == ROOT_RANK and total_filtered > 0:
            pct = 100 * total_filtered / len(arrivals)
            logger.info(f" {total_filtered} ({pct:.1f}%) raypaths removed which bottomed between {raypath_filter_min} to {raypath_filter_max}km")

        COMM.barrier()
        return True


    @_utilities.log_errors(logger)
    def _dispatch(self, ids, sentinel=None):
        """
        Dispatch ids to hungry workers, then dispatch sentinels
        """

        logger.debug("_dispatch called with %d items" % len(list(ids)))

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
        Fast estimate of average horizontal cell width using KDTree
        """

        cells_geo = np.array([sph2geo(cell) for cell in voronoi_cells])
        lats = cells_geo[:, 0]
        lons = cells_geo[:, 1]

        # Project rough local cartesian (km)
        mean_lat = np.mean(lats)
        x_km = (lons - np.mean(lons)) * 111 * np.cos(np.radians(mean_lat))
        y_km = (lats - np.mean(lats)) * 111

        points_2d = np.column_stack([x_km, y_km])

        # Find nearest neighbor for each point (k=2 to skip self)
        # TODO: could only select a random half, for speed, but I suppose it's already pretty fast
        tree = cKDTree(points_2d)
        distances, indices = tree.query(points_2d, k=2)
        nearest_distances = distances[:, 1]

        # Convert to cell width estimate
        mean_spacing = np.mean(nearest_distances)

        return mean_spacing * 2 # in practice they are quite a bit wider

    # not in use
    def _build_station_coverage_mask(self, candidate_cells_sph):
        """
        Return a boolean mask (True = keep) for candidate Voronoi cells that fall
        within ~max_dist km of at least one active station, measured as horizontal
        surface distance only (depth is ignored).

        Args:
        candidate_cells_sph : np.ndarray, shape (N, 3)
            Candidate cell positions in spherical (rho, theta, phi) coords.

        Returns:
        mask : np.ndarray of bool, shape (N,)
        """
        scale = 1.15 # extend 15% just because

        if scale <= 0 or self.stations is None or len(self.stations) == 0:
            return np.ones(len(candidate_cells_sph), dtype=bool)

        coverage_km  = self.cfg["algorithm"]["max_dist"] * scale
        coverage_rad = coverage_km / _constants.EARTH_RADIUS

        # Station positions on unit sphere — surface only
        sta = self.stations
        sta_geo = np.column_stack([
            sta["latitude"].values,
            sta["longitude"].values,
            np.zeros(len(sta))
        ])
        sta_sph = np.array([geo2sph(row) for row in sta_geo])
        sta_xyz = sph2xyz(sta_sph)   # unit-sphere Cartesian

        # Candidate cells projected to unit sphere (strip rho)
        cell_theta_phi = candidate_cells_sph.copy()
        cell_theta_phi[:, 0] = _constants.EARTH_RADIUS  # set to surface radius
        cell_xyz = sph2xyz(cell_theta_phi)

        # Chord-distance threshold: chord = 2*sin(arc/2)
        chord_threshold = 2.0 * np.sin(coverage_rad / 2.0)

        tree = cKDTree(sta_xyz)
        dist, _ = tree.query(cell_xyz, k=1)

        return dist <= chord_threshold


    @_utilities.log_errors(logger)
    def _generate_voronoi_cells(self, phase):
        """
        Generate Voronoi cells via top-down density-aware refinement.
     
        Algorithm:
            1. Generate a coarse uniform-random initial mesh, sized so that
               average cell width ~= max_cell_width_km. Initial seeds are
               placed with bias toward high-density / high-gradient regions
               via the existing density_to_gradient_weight blend.
            2. Iteratively refine: for each cell, count rays captured within
               a sphere of radius (nearest-neighbor-distance / 2). If too many,
               split the cell into 4 children placed within the parent's
               neighborhood. Repeat until no cell needs splitting.
            3. Refinement stops on cells that hit min_cell_width_km — their
               ray counts are accepted regardless.
            4. Sparse-region cells naturally never get split, so they remain
               at the initial coarse size — no extra work in those regions.

        This is much faster than bottom-up greedy placement on large datasets:
        work scales with final cell count (~thousands), not with point count
        (~millions).

        Reads (cfg["algorithm"]):
            target_rays_per_cell        target unique rays per cell
            density_to_gradient_weight  data:gradient blend (default 0.5)
            cell_size_jitter            +/- jitter on per-realization target (default 0.2)
            ray_subsample_fraction      fraction of rays kept per realization (default 0.8)
            min_cell_width_km           refinement floor (km). 0 = no floor.
            max_cell_width_km           initial mesh scale (km). 0 = uses bbox/10.
            enable_backfill             keep backfill in empty regions (default True)
            max_refine_passes           safety cap on refinement iterations (default 20)
            hvr                         horizontal/vertical ratio (default 1.0).
                                        >1 produces horizontally-elongated, vertically-thin
                                        cells. min/max_cell_width_km describe horizontal
                                        extent; vertical extent ~= width/hvr.
                                        Matches the transform used in _projected_ray_idxs.

        """
        if RANK == ROOT_RANK:
     
            cfg               = self.cfg["meshing"]
            target_rpc        = cfg["target_rays_per_cell"]
            dg_weight         = cfg["density_to_gradient_weight"]
            size_jitter       = 0.2 # hardwired at 20%
            ray_subsample     = 0.8 # hardwired, re-select only 80% of rays (todo parameterize?)
            min_cell_width_km = cfg["min_cell_width_km"]
            max_cell_width_km = cfg["max_cell_width_km"]
            enable_backfill   = cfg["enable_backfill"]
            max_passes        = 20 # hardwired at 20 / more than enough
            hvr               = cfg["hvr"]
     
            dg_weight     = float(np.clip(dg_weight, 0.0, 1.0))
            ray_subsample = float(np.clip(ray_subsample, 0.25, 1.0))
     
            target_rpc_real = max(
                5,
                int(target_rpc * np.random.uniform(1.0 - size_jitter,
                                                   1.0 + size_jitter))
            )
     
            if phase == "P":
                model = self.pwave_model
            elif phase == "S":
                model = self.swave_model
            else:
                raise ValueError(f"Unrecognized phase ({phase})")
     
            min_coords = model.min_coords
            max_coords = model.max_coords
            delta      = max_coords - min_coords

            # ----------------------------------------------------------------------
            # Anisotropic coordinate transform (follows _projected_ray_idxs)
            # ----------------------------------------------------------------------
            # hvr > 1 compresses theta/phi (horizontal) axes during distance
            # computations. After the transform, vertical separations look larger
            # relative to horizontal ones, so:
            #   - The kd-tree "nearest neighbor" prefers vertical alignment
            #   - Refinement splits more aggressively in the vertical direction
            #   - Final cells become horizontally elongated, vertically thin
            # The user-facing min/max_cell_width_km parameters are interpreted in
            # transformed-space km, which corresponds to *horizontal* km in the
            # raw model (vertical cell extents will be ~width/hvr).

            center = (min_coords + max_coords) / 2.0
            scale = np.array([1.0/hvr, 1.0, 1.0]) # this is inverted relative to projected_ray_idxs.. we are only increasing vertical (rho) density
     
            def _to_xyz(coords_sph):
                """Apply hvr scaling, then convert spherical → cartesian (km)."""
                if hvr != 1:
                    transformed = center + (coords_sph - center) / scale
                    return sph2xyz(transformed)
                else:
                    return sph2xyz(coords_sph)
     
            def _from_xyz(coords_xyz):
                """Inverse: cartesian → spherical, then undo hvr scaling."""
                if hvr != 1:
                    transformed = xyz2sph(coords_xyz)
                    return center + (transformed - center) * scale
                else:
                    return xyz2sph(coords_xyz)

            # 1. Get raypath cloud + ray IDs
            points, ray_ids = self._sample_raypaths(phase)
            if len(points) == 0:
                logger.warning("No raypath points — falling back to random cells")
                self.voronoi_cells = np.random.rand(100, 3) * delta + min_coords
                self.synchronize(attrs=["voronoi_cells"])
                return True

            in_bounds = np.all((points >= min_coords) & (points <= max_coords), axis=1)
            points  = points[in_bounds]
            ray_ids = ray_ids[in_bounds]

            if ray_subsample < 1.0:
                unique_rays = np.unique(ray_ids)
                n_keep = max(10, int(len(unique_rays) * ray_subsample)) # stay above 10 at least..
                keep_rays = np.random.choice(unique_rays, n_keep, replace=False)
                keep_mask = np.isin(ray_ids, keep_rays)
                points  = points[keep_mask]
                ray_ids = ray_ids[keep_mask]

            n_points = len(points)
            n_unique_rays = len(np.unique(ray_ids))
            if n_unique_rays == 0:
                logger.warning("No rays after subsampling — falling back")
                self.voronoi_cells = np.random.rand(100, 3) * delta + min_coords
                self.synchronize(attrs=["voronoi_cells"])
                return True

            points_xyz = _to_xyz(points)
            point_tree = cKDTree(points_xyz)
            bbox_diag = np.linalg.norm(points_xyz.max(axis=0) - points_xyz.min(axis=0))

            # 2. Density field for biasing initial mesh
            density_interp = None
            try:
                data_density_3d, _ = self._estimate_data_density(phase, 1.0)
                grad_density_3d    = self.gradient_magnitude
                if data_density_3d is not None and grad_density_3d is not None:
                    eff_density = (dg_weight * data_density_3d
                                   + (1.0 - dg_weight) * grad_density_3d)
                    floor = max(eff_density.max() * 0.01, 1e-6)
                    eff_density = np.maximum(eff_density, floor)
                    density_interp = scipy.interpolate.RegularGridInterpolator(
                        self.grid_coords, eff_density,
                        bounds_error=False, fill_value=floor
                    )
            except Exception as e:
                logger.warning(f"Density field unavailable: {e}")

            # 3. Initial coarse mesh
            # Sizing: use max_cell_width_km as the initial cell scale. The
            # refinement loop with Voronoi assignment will subdivide where data
            # density warrants. Starting coarse is correct — sparse regions stay
            # coarse, dense regions split down.
            if max_cell_width_km > 0:
                target_init_width = max_cell_width_km
            else:
                target_init_width = bbox_diag / 10.0

            #bbox_volume = np.prod(points_xyz.max(axis=0) - points_xyz.min(axis=0))
            #cell_volume = (4.0 / 3.0) * np.pi * (target_init_width / 2.0) ** 3
            #n_initial = int(bbox_volume / max(cell_volume, 1e-9))

            # This approximates a better estimate given model dimensions?
            corners = _to_xyz(np.array([min_coords, max_coords]))
            model_volume = np.prod(np.abs(corners[1] - corners[0]))
            n_initial = int(model_volume / target_init_width ** 3)

            n_initial = int(n_initial * np.random.uniform(0.8, 1.2)) # a touch of chaos
            n_initial = max(16, n_initial) # 16 is bottom of the barrel. Code should come up with something sensible...
            if n_initial > 20000:
                logger.warning("limiting the initial number of cells to 20000 (!!!)")
            n_initial = min(n_initial, 20000)

            # Sample initial seed positions in spherical coords. If density_interp
            # exists, use rejection sampling biased by it; else uniform.
            seeds_sph = self._sample_initial_seeds(
                n_initial, min_coords, max_coords, delta,
                density_interp, oversample=4
            )
            n_initial = len(seeds_sph)

            # 4. Refinement loop: split cells with too many rays
            n_splits_total = 0
            for refine_pass in range(max_passes):
                seeds_xyz = _to_xyz(seeds_sph)
                seed_tree = cKDTree(seeds_xyz)

                # NN distance per seed = approximate cell diameter
                nn_dists, _ = seed_tree.query(seeds_xyz, k=2)
                nn_dists = nn_dists[:, 1]

                # Voronoi-style assignment: each raypath point's "owner" cell is
                # its nearest seed. This is what the actual Voronoi tessellation
                # downstream will do, so splitting decisions match real cell
                # behavior. Avoids the capture-sphere overlap problem where the
                # same ray gets counted in multiple sparse cells' spheres.
                _, nearest_cell_per_point = seed_tree.query(points_xyz, k=1)

                # Per-cell unique ray count
                n_cells = len(seeds_xyz)
                cell_ray_counts = np.zeros(n_cells, dtype=int)
                # Group point indices by their owning cell
                sort_idx = np.argsort(nearest_cell_per_point)
                sorted_owners = nearest_cell_per_point[sort_idx]
                sorted_rays   = ray_ids[sort_idx]
                # Find segment boundaries (where owner changes)
                change_points = np.concatenate([
                    [0], np.where(np.diff(sorted_owners) != 0)[0] + 1, [len(sorted_owners)]
                ])
                for k in range(len(change_points) - 1):
                    a, b = change_points[k], change_points[k + 1]
                    if a == b:
                        continue
                    cell_id = sorted_owners[a]
                    cell_ray_counts[cell_id] = len(np.unique(sorted_rays[a:b]))

                # Decide which cells to split
                cells_to_split = []
                for i in range(n_cells):
                    n_rays_here = cell_ray_counts[i]
                    r_cell = nn_dists[i]
                    # Split if over target AND splitting would not produce too-small cells.
                    # Children will be roughly r_cell / 2 wide after a split into 4.
                    if (n_rays_here > target_rpc_real and
                        (min_cell_width_km == 0 or r_cell / 1.587 >= min_cell_width_km)): # n.b. 4^(1/3) = 1.587
                        cells_to_split.append(i)

                if not cells_to_split: # i.e. converged
                    break

                np.random.shuffle(cells_to_split) # a touch of chaos

                # Perform splits: replace each parent with N=4 children placed
                # randomly within the parent's neighborhood.
                new_seeds_sph = []
                keep_mask = np.ones(len(seeds_sph), dtype=bool)
                for i in cells_to_split:
                    keep_mask[i] = False
                    parent_xyz = seeds_xyz[i]
                    r_split = nn_dists[i] * 0.5
                    n_children = np.random.choice([3,4,5])
                    offsets = np.random.normal(scale=r_split / 2.0, size=(n_children, 3))
                    children_xyz = parent_xyz + offsets
                    children_sph = _from_xyz(children_xyz)
                    children_sph = np.clip(children_sph, min_coords, max_coords)
                    new_seeds_sph.append(children_sph)

                n_splits_total += len(cells_to_split)
                seeds_sph = np.vstack([seeds_sph[keep_mask]] + new_seeds_sph)

            else:
                logger.warning(f"  Refinement hit max_passes ({max_passes}) without "
                               f"converging — mesh may have under-resolved cells")


            # 5. Cull cells with zero assigned rays (Voronoi-style)
            # Initial-mesh cells that landed in regions with no rays will own
            # zero raypath points under nearest-seed assignment. Drop them.
            n_pre_cull = len(seeds_sph)
            seeds_xyz_now = _to_xyz(seeds_sph)
            seed_tree_now = cKDTree(seeds_xyz_now)
            _, owners = seed_tree_now.query(points_xyz, k=1)
            owned_cells = np.unique(owners)
            keep = np.zeros(len(seeds_sph), dtype=bool)
            keep[owned_cells] = True
            seeds_sph = seeds_sph[keep]
            n_culled = n_pre_cull - len(seeds_sph)


            # 6. Backfill model-volume gaps (optional, geometric)
            """ OLD
            n_data_cells = len(seeds_sph)
            n_filler = 0
            if enable_backfill and n_data_cells > 0:
                seeds_xyz_arr = _to_xyz(seeds_sph)
                seed_tree = cKDTree(seeds_xyz_arr)
                if len(seeds_xyz_arr) >= 2:
                    nn2, _ = seed_tree.query(seeds_xyz_arr, k=2)
                    median_nn = np.median(nn2[:, 1])
                else:
                    median_nn = bbox_diag * 0.1
                backfill_spacing = median_nn * 1.5

                n_candidates = 12 ** 3 * 2
                cands_sph = np.random.uniform(min_coords, max_coords,
                                              size=(n_candidates, 3))

                cands_xyz = _to_xyz(cands_sph)
                d_to_nearest, _ = seed_tree.query(cands_xyz, k=1)
                far_mask = d_to_nearest > backfill_spacing
                filler_sph = cands_sph[far_mask]
                n_filler = len(filler_sph)
                if n_filler > 0:
                    seeds_sph = np.vstack([seeds_sph, filler_sph])
            """

            # 6. Optionally backfill model-volume gaps at the COARSE scale (max_cell_width_km)
            n_data_cells = len(seeds_sph)
            n_filler = 0
            if enable_backfill and n_data_cells > 0:
                seeds_xyz_arr = _to_xyz(seeds_sph)
                seed_tree = cKDTree(seeds_xyz_arr)

                # Coarse spacing = one max-width cell (already in transformed-xyz km here,
                # same units as the kd-tree distances and as target_init_width above).
                backfill_spacing = max_cell_width_km if max_cell_width_km > 0 else bbox_diag * 0.1

                # Candidate count ~ a coarse tiling of the model volume at that spacing.
                # Nearly all candidates pass the far-from-data test (data footprint is
                # tiny), so this is effectively the final filler count. Cap for safety.
                corners = _to_xyz(np.array([min_coords, max_coords]))
                model_volume = np.prod(np.abs(corners[1] - corners[0]))
                n_candidates = int(np.clip(model_volume / backfill_spacing**3 * 2, 32, 20000))

                cands_sph = np.random.uniform(min_coords, max_coords, size=(n_candidates, 3))
                cands_xyz = _to_xyz(cands_sph)
                d_to_nearest, _ = seed_tree.query(cands_xyz, k=1)
                far_mask = d_to_nearest > backfill_spacing
                filler_sph = cands_sph[far_mask]
                n_filler = len(filler_sph)
                if n_filler > 0:
                    seeds_sph = np.vstack([seeds_sph, filler_sph])

            self.voronoi_cells = seeds_sph


            # 7. Diagnostics
            # Cell widths reported on data-driven cells only — backfill cells
            # sit on a regular grid that would otherwise dominate the statistics
            # and hide the data-adaptive cell-size variation.
            seeds_data = seeds_sph[:n_data_cells]
            if len(seeds_data) >= 2:
                seeds_xyz_data = _to_xyz(seeds_data)
                ctree = cKDTree(seeds_xyz_data)
                nn_dists_data, _ = ctree.query(seeds_xyz_data, k=2)
                cell_widths_km = nn_dists_data[:, 1] * 2.0
                wp10, wp25, wp50, wp75, wp90 = np.percentile(
                    cell_widths_km, [10, 25, 50, 75, 90]
                )
                width_mean = float(np.mean(cell_widths_km))
            else:
                wp10 = wp25 = wp50 = wp75 = wp90 = width_mean = 0.0

            logger.info(
                f" Adaptive mesh: {len(seeds_sph)} cells "
                f"({n_data_cells} data-driven, {n_filler} backfill); "
                f"started with {n_initial}, performed {n_splits_total} splits "
                f"in {refine_pass + 1} passes, then culled {n_culled} empty cells"
            )
            logger.info(
                f" Cell width (km, {hvr}x vert compression, data-driven only): "
                f"10%: {wp10:.1f} 25%: {wp25:.1f} 75%: {wp75:.1f} 90%: {wp90:.1f}, "
                f"mean|median = {width_mean:.1f}|{wp50:.1f}"
            )

        self.synchronize(attrs=["voronoi_cells"])
        return True


    def _sample_initial_seeds(self, n, min_coords, max_coords, delta,
                              density_interp, oversample=4):
        """
        Generate n initial seed positions in spherical model coords.

        If density_interp is provided, uses rejection sampling biased by it
        (more seeds where data density / gradient magnitude is higher).
        Otherwise uniform random.
        """
        if density_interp is None:
            return np.random.rand(n, 3) * delta + min_coords

        # Generate oversample candidates uniform in model space, accept with
        # probability density(candidate) / density.max()
        candidates = np.random.rand(oversample * n, 3) * delta + min_coords
        weights = density_interp(candidates)
        wmax = max(weights.max(), 1e-9)
        probs = weights / wmax
        accept = np.random.rand(len(candidates)) < probs * np.random.uniform(0.9, 1.1) # add a bit of variability to the acceptance thresh
        chosen = candidates[accept]

        if len(chosen) >= n:
            idx = np.random.choice(len(chosen), n, replace=False)
            return chosen[idx]
        # If we got fewer than n (low overall density), top up with uniform
        deficit = n - len(chosen)
        extra = np.random.rand(deficit, 3) * delta + min_coords
        return np.vstack([chosen, extra])


    @_utilities.root_only(RANK)
    def _diagnose_mesh(self, phase):
        """
        Log post-meshing diagnostics
        """
        if self.sensitivity_matrix is None or self.voronoi_cells is None:
            return

        nvoronoi = len(self.voronoi_cells)
        target_rpc = self.cfg["meshing"].get("target_rays_per_cell", 25)
        min_rays = self.cfg["meshing"].get("min_rays_per_cell", 10)

        sens = self.sensitivity_matrix.tocsr()[:, :nvoronoi].tocoo()
        if sens.nnz == 0:
            logger.warning("  diagnose_mesh: sensitivity matrix is empty")
            return

        pairs = np.unique(np.column_stack([sens.row, sens.col]), axis=0)
        rays_per_cell = np.bincount(pairs[:, 1], minlength=nvoronoi)

        # Compute quality stats over informed cells (rays > 0) only;
        # report total mesh size separately for context.
        informed_mask = rays_per_cell > 0
        n_informed = int(informed_mask.sum())
        n_empty    = nvoronoi - n_informed
        informed_rpc = rays_per_cell[informed_mask]

        if n_informed == 0:
            logger.warning(f"  Mesh diagnostic: no informed cells (mesh has {nvoronoi} cells, all empty)")
            return

        n_under     = int(np.sum(informed_rpc < min_rays))
        n_well      = int(np.sum(informed_rpc >= target_rpc * 0.5))

        p10, p25, p50, p75, p90 = np.percentile(informed_rpc, [10, 25, 50, 75, 90])

        logger.info(f"  Mesh diagnostic ({phase}, target {target_rpc} rays/cell):")
        logger.info(f"    {n_informed} populated cells: "
                    f"{n_under} ({100*n_under/n_informed:.1f}%) under min_rays ({min_rays}), "
                    f"{n_well} ({100*n_well/n_informed:.1f}%) well-resolved")
        logger.info(f"    rays-per-cell 10%: {p10:.0f}, 25%: {p25:.0f}, 50%: {p50:.0f}, 75%: {p75:.0f}, 90%: {p90:.0f}")


    def _estimate_data_density(self, phase, adaptive_weight):
        """
        Estimate data density based on arrival counts in model coordinate system

        Parameters:
        -----------
        phase : str
            Phase type
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

        if phase == "P":
            model = self.pwave_model
        elif phase == "S":
            model = self.swave_model

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
                uniform_density = np.ones(model.npts)
                edges = [
                    np.linspace(model.min_coords[i], model.max_coords[i], model.npts[i] + 1)
                    for i in range(3)
                ]
                return uniform_density, edges
            else:
                return np.ones(np.prod(model.npts)) / np.prod(model.npts)

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
                uniform_density = np.ones(model.npts)
                edges = [
                    np.linspace(model.min_coords[i], model.max_coords[i], model.npts[i] + 1)
                    for i in range(3)
                ]
                return uniform_density, edges
            else:
                return np.ones(np.prod(model.npts)) / np.prod(model.npts)

        midpoints = midpoints[valid_points]
        min_coords = model.min_coords
        max_coords = model.max_coords

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
                uniform_density = np.ones(model.npts)
                edges = [
                    np.linspace(model.min_coords[i], model.max_coords[i], model.npts[i] + 1)
                    for i in range(3)
                ]
                return uniform_density, edges
            else:
                return np.ones(np.prod(model.npts)) / np.prod(model.npts)

        midpoints = midpoints[bounded_mask]

        # Create histogram bins directly from model grid
        edges = [
            np.linspace(model.min_coords[i], model.max_coords[i], model.npts[i] + 1)
            for i in range(3)
        ]

        # Create 3D histogram
        try:
            hist, edges = np.histogramdd(midpoints, bins=edges)

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
                uniform_density = np.ones(model.npts)
                edges = [
                    np.linspace(model.min_coords[i], model.max_coords[i], model.npts[i] + 1)
                    for i in range(3)
                ]
                return uniform_density, edges
            else:
                return np.ones(np.prod(model.npts)) / np.prod(model.npts)


    def _estimate_velocity_gradient_density(self, phase):
        """
        Estimate 3D velocity gradient magnitude to guide adaptive meshing.
        Returns normalized gradient field (0-1) on model grid.

        Called in iterate once per iteration, per phase

        Prioritize horizontal gradients!
        """
        if RANK == ROOT_RANK:

            logger.debug("Calculating velocity gradient magnitudes")

            if phase == "P":
                model = self.pwave_model
            elif phase == "S":
                model = self.swave_model
            else:
                raise ValueError(f"Unrecognized phase ({phase})")

            # Get gradient from pykonal (per axis; shape: dz, N1, N2, 3)
            # TODO we may want to smooth this out a bit later
            gradient = model.gradient.values

            # Compute gradient magnitude
            grad_magnitude = np.sqrt(
                #gradient[..., 0]**2 + gradient[..., 1]**2 + gradient[..., 2]**2
                gradient[..., 1]**2 + gradient[..., 2]**2 # just horizontal probably best
            )

            # Normalize 0-1
            grad_max = grad_magnitude.max()
            if grad_max > 0:
                grad_magnitude = grad_magnitude / grad_max
            else:
                grad_magnitude = np.zeros_like(grad_magnitude)

            # Optional: emphasize strong gradients
            grad_magnitude = grad_magnitude ** 0.5

            # Create 1D coordinate arrays for RegularGridInterpolator
            # (monotonic already via pykonal)
            # Needed for both data_interpolator AND grid_interpolator
            grid_coords = [
                np.linspace(
                    model.min_coords[i],
                    model.max_coords[i],
                    model.npts[i],
                    dtype=np.float64
                )
                for i in range(3)
            ]

            self.gradient_magnitude = grad_magnitude
            self.grid_coords = grid_coords

        self.synchronize(attrs=["gradient_magnitude","grid_coords"])

        return


    @_utilities.log_errors(logger)
    def _sample_raypaths(self, phase):
        """
        Get raypath points from stored HDF5 files.

        Returns
        -------
        points : (N, 3) float array
            All raypath sample points concatenated.
        ray_ids : (N,) int array
            Ray ID for each point. Points with the same ray_id belong to the
            same arrival/ray. Ray IDs are unique within a realization but have
            no meaning across realizations.
        """
        point_chunks = []
        ray_id_chunks = []
        next_ray_id = 0
     
        logger.debug("Model depth range: %.2f to %.2f" % (
            6371 - self.pwave_model.min_coords[0],
            6371 - self.pwave_model.max_coords[0]
        ))

        arrivals = self.sampled_arrivals.set_index(["network", "station"]).sort_index()
        index = arrivals.index.unique()
        events = self.events.set_index("event_id")
        events["idx"] = np.arange(len(events))

        for network, station in index:
            filename = f"{network}.{station}.{phase}.h5"
            path = os.path.join(self.raypath_dir, filename)
            with h5py.File(path, mode="r") as raypath_file:
                event_ids = arrivals.loc[[(network, station)], "event_id"]
                idxs = events.loc[event_ids, "idx"]
                idxs = np.sort(idxs).astype(int)
                raypoints = raypath_file[phase][:, idxs]
                # raypoints: shape (3, n_events), object array; each cell is a
                # 1D float array of coordinate samples along that ray.

                if raypoints.ndim > 1:
                    n_events = raypoints.shape[1]
                    # Per-event ray length = length of the rho-coordinate array
                    # (all 3 components have the same length per ray)
                    for j in range(n_events):
                        rho   = np.asarray(raypoints[0, j], dtype=float)
                        theta = np.asarray(raypoints[1, j], dtype=float)
                        phi   = np.asarray(raypoints[2, j], dtype=float)
                        n_pts = len(rho)
                        if n_pts == 0:
                            continue
                        one_ray = np.column_stack([rho, theta, phi])  # (n_pts, 3)
                        point_chunks.append(one_ray)
                        ray_id_chunks.append(np.full(n_pts, next_ray_id, dtype=np.int32))
                        next_ray_id += 1
                else:
                    # Single-event station: raypoints is shape (3,), each cell a 1D array
                    rho   = np.asarray(raypoints[0], dtype=float)
                    theta = np.asarray(raypoints[1], dtype=float)
                    phi   = np.asarray(raypoints[2], dtype=float)
                    n_pts = len(rho)
                    if n_pts > 0:
                        one_ray = np.column_stack([rho, theta, phi])
                        point_chunks.append(one_ray)
                        ray_id_chunks.append(np.full(n_pts, next_ray_id, dtype=np.int32))
                        next_ray_id += 1

        if not point_chunks:
            return np.empty((0, 3), dtype=float), np.empty(0, dtype=np.int32)

        points  = np.vstack(point_chunks).astype(float, copy=False)
        ray_ids = np.concatenate(ray_id_chunks)
        logger.debug(f"_sample_raypaths: {len(points)} points across "
                     f"{next_ray_id} unique rays "
                     f"(avg {len(points)/max(next_ray_id,1):.1f} pts/ray)")
        return points, ray_ids


    def _sample_raypaths_OLD(self, phase):
        """
        Get raypath points from stored HDF5 files
        """
        points = np.empty((0, 3))

        logger.debug("Model depth range: %.2f to %.2f" % (6371-self.pwave_model.min_coords[0],6371-self.pwave_model.max_coords[0]))

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
                    raypoints = np.stack(raypoints).reshape(-1, 1)  # force (n_points, 1) so 1-event catalogs can be used
                raypoints = raypoints.T
                points = np.vstack([points, raypoints])

        return points


    @_utilities.log_errors(logger)
    def _projected_ray_idxs(self, raypath, hvr, voronoi_tree=None, center=None):
        """
        Return the cell IDs (column IDs) of each segment of the given
        raypath and the length of each segment in counts.
        """
        if voronoi_tree is None:
            min_coords = self.pwave_model.min_coords # technically this is only every called from compute_sensitivty_matrix so P/S already caught. otherwise TODO
            max_coords = self.pwave_model.max_coords
            center = (min_coords + max_coords) / 2

            voronoi_cells = self.voronoi_cells
            voronoi_cells = center + (voronoi_cells - center) / [1, hvr, hvr] # n.b. dividing here is correct. hvr > 1 makes wider cells
            voronoi_cells = sph2xyz(voronoi_cells)
            tree = cKDTree(voronoi_cells)
        else:
            tree = voronoi_tree

        raypath = center + (raypath - center) / [1, hvr, hvr]
        raypath = sph2xyz(raypath)

        _, column_idxs = tree.query(raypath)
        column_idxs, counts = np.unique(column_idxs, return_counts=True)

        logger.debug("Ray query results: %d points, counts range: %d-%d", len(column_idxs), counts.min(), counts.max())

        return (column_idxs, counts)


    @_utilities.log_errors(logger)
    def _request_dispatch(self):
        """
        Request, receive, and return item from dispatcher
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


    def _stack_statistic(self, stack_ma):
        """
        The stack statistic that defines the final model update
        (trimmed mean or median per algorithm.stack_trim_percent and
        algorithm.stack_type), computed over a masked realization stack.
        Shared by update_model() and the convergence check so that
        "converged" means converged in the quantity that actually
        matters.
        """
        trim_fraction = self.cfg["algorithm"].get("stack_trim_percent", 0)
        trim_fraction = np.clip(float(trim_fraction / 100), 0, 0.485)

        if trim_fraction > 0:
            n = stack_ma.shape[0]
            n_trim = int(n * trim_fraction)
            if n_trim > 0 and n > 2 * n_trim:
                stack_ma = np.ma.sort(stack_ma, axis=0)[n_trim:-n_trim]

        stack_type = self.cfg["algorithm"].get("stack_type")
        if stack_type.lower() == "median":
            return np.ma.median(stack_ma, axis=0)
        else:
            return np.ma.mean(stack_ma, axis=0)


    def _stack_converged(self, phase, nreal):
        """
        Adaptive early stopping: decide (on ROOT, broadcast to all ranks)
        whether adding further realizations would change the final stack
        by less than algorithm.stack_convergence_tol.

        Metric: RMS(S_n - S_prev) / RMS(S_n) over valid nodes, where S is
        the running _stack_statistic and S_prev is the statistic at the
        previous check (check_every realizations earlier). Must stay
        below tol for `patience` consecutive checks. Returns True to
        break the realization loop; MUST be called by ALL ranks.
        """
        cfg = self.cfg["algorithm"]
        tol = cfg.get("stack_convergence_tol")

        decision = False
        if RANK == ROOT_RANK and tol > 0:
            n_done = self.ireal + 1
            min_nreal = max(int(cfg.get("stack_convergence_min_nreal")), 2)
            check_every = max(int(cfg.get("stack_convergence_check_every")), 1)
            patience = max(int(cfg.get("stack_convergence_patience")), 1)

            if n_done >= min_nreal and (n_done - min_nreal) % check_every == 0:
                dataset = self._f5_workspace[f"{phase.lower()}wave_stack"]
                raw = dataset[:n_done]
                stack_ma = np.ma.masked_invalid(raw)
                current = self._stack_statistic(stack_ma)

                state = self._stack_conv_state.setdefault(
                    phase, {"prev": None, "passes": 0}
                )
                if state["prev"] is not None:
                    diff = current - state["prev"]
                    denom = np.sqrt(np.ma.mean(current ** 2)) + 1e-30
                    metric = float(np.sqrt(np.ma.mean(diff ** 2)) / denom)
                    state["passes"] = (
                        state["passes"] + 1 if metric < tol else 0
                    )
                    logger.info(
                        f"  stack convergence [{phase}]: rel change = "
                        f"{metric:.4f} over last {check_every} realization(s) "
                        f"(tol {tol:.4f}, pass {state['passes']}/{patience}, "
                        f"n={n_done}/{nreal})   ###"
                    )
                    if state["passes"] >= patience:
                        decision = True
                        logger.info(
                            f"  stack converged for {phase} after {n_done} "
                            f"of {nreal} realizations; skipping the "
                            f"remaining {nreal - n_done}   ###"
                        )
                state["prev"] = current

        decision = COMM.bcast(decision, root=ROOT_RANK)
        return decision


    @_utilities.log_errors(logger)
    @_utilities.root_only(RANK)
    def _reset_realization_stack(self, phase):
        """ Reset the realization stack by deleting and recreating """
        phase = phase.lower()

        if not hasattr(self, "_stack_conv_state"):
            self._stack_conv_state = {}
        self._stack_conv_state.pop(phase, None)
        self._stack_conv_state.pop(phase.upper(), None)

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

            # Filter by distance for sampled_arrivals
            max_dist   = self.cfg["algorithm"]["max_dist"]
            min_dist   = self.cfg["algorithm"]["min_dist"]

            arrivals = self.arrivals[
                (self.arrivals["phase"] == phase) & 
                (self.arrivals["slant_dist"] >= min_dist) & 
                (self.arrivals["delta"] <= max_dist)
            ]

            event_ids = self.sampled_events["event_id"]
            arrivals = arrivals[arrivals["event_id"].isin(event_ids)].copy()

            if self.cfg["algorithm"]["narrival_percent"] > 0:
                narrival = int(len(arrivals) * self.cfg["algorithm"]["narrival_percent"]/100) + 1
            else:
                narrival = self.cfg["algorithm"]["narrival"]

            if do_remove_outliers:
                tukey_k = self.cfg["algorithm"]["outlier_removal_factor"]
                max_arr_resid = self.cfg["algorithm"]["max_arrival_residual"]
                n0 = len(arrivals)
                arrivals = _utilities.remove_outliers(arrivals, tukey_k, "residual", max_arr_resid)
                if len(arrivals) < 0.75*n0:
                    logger.warning(f"FYI: remove_outliers removed {(n0-len(arrivals))/n0*100:.1f}% of your arrivals!   ###")

            if not useall and narrival < len(arrivals):
                if 'weight' not in arrivals.columns:
                    arrivals['weight'] = 1.0
                    logger.warning("arrivals['weight'] column missing in sample_arrivals??   ###")
                always_include = self.cfg["model"].get("always_include_truepicks", True)
                if (always_include and "truepick" in arrivals.columns and arrivals["truepick"].any()):
                    # ground-truth picks constrain EVERY realization: sample
                    # the remainder, then union the truepicks back in
                    true_arrivals = arrivals[arrivals["truepick"]]
                    rest = arrivals[~arrivals["truepick"]]
                    n_rest = max(narrival - len(true_arrivals), 0)
                    if n_rest > 0 and n_rest < len(rest):
                        rest = rest.sample(n=n_rest, weights="weight")
                    arrivals = pd.concat([true_arrivals, rest])
                    logger.debug(
                        f"sampled_arrivals: {len(true_arrivals)} truepicks "
                        f"force-included + {len(rest)} sampled   ###"
                    )
                else:
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
                nevent = int(len(self.events) * self.cfg["algorithm"]["nevent_percent"]/100) + 1
            else:
                nevent = self.cfg["algorithm"]["nevent"]

            if do_remove_outliers:
                max_evt_resid = self.cfg["algorithm"]["max_event_residual"]
                n0 = len(self.events)
                events = _utilities.remove_outliers(self.events, None, "residual", max_evt_resid)
                if len(events) < 0.75*n0:
                    logger.warn(f"FYI: remove_outliers removed {(n0-len(self.events))/n0*100:.1f}% of your events!   ###")
            else:
                events = self.events

            if not useall and nevent < len(events):
                if 'weight' not in events.columns:
                    events['weight'] = 1.0
                    logger.warning("events['weight'] column missing in sample_events??   ###")
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
            n_events = len(events)

            # Precompute spherical coords for all events once instead
            coords_cache = {}
            for event_id, event in events.iterrows():
                coords = event[["latitude", "longitude", "depth"]]
                coords_cache[event_id] = geo2sph(coords).astype(dtype=_constants.DTYPE_REAL)

            # Pre-compute event_id to idx mapping
            idx_cache = dict(zip(events.index, events["idx"]))

            # Precompute station -> event_ids mapping once
            station_event_ids = {
                key: grp["event_id"].values
                for key, grp in arrivals.groupby(level=[0, 1])
            }

            _path = self.traveltime_inventory_path
            with TraveltimeInventory(_path, mode="r") as traveltime_inventory:
                while True:
                    item = self._request_dispatch()
                    if item is None: break

                    network, station = item
                    handle = "/".join([network, station, phase])

                    filename = ".".join([network, station, phase])
                    path = os.path.join(raypath_dir, filename + ".h5")
                    raypath_file = h5py.File(path, mode="a")

                    if phase not in raypath_file:
                        dtype = h5py.vlen_dtype(_constants.DTYPE_REAL)
                        dataset = raypath_file.create_dataset(phase,(3, n_events,), dtype=dtype)
                    else:
                        dataset = raypath_file[phase]

                    event_ids = station_event_ids[(network, station)]

                    # Establish a bool here per event rather than load all the arrivals
                    if "traced" not in raypath_file:
                        initial_traced = np.array([
                            np.stack(dataset[:, i]).size != 0
                            for i in range(n_events)
                        ])
                        raypath_file.create_dataset("traced", data=initial_traced)
                    traced = raypath_file["traced"][:]

                    # Early exit: if every event for this station is already traced, skip
                    needed_indices = np.fromiter(map(idx_cache.__getitem__, event_ids), dtype=int, count=len(event_ids))
                    if traced[needed_indices].all():
                        raypath_file.close()
                        continue
                    else:
                        traveltime = traveltime_inventory.read(handle)
                        traced0 = traced.copy() # n.b. this is just a small bool array

                    for event_id in event_ids:

                        idx = idx_cache[event_id]
                        if traced[idx]: continue

                        coords = coords_cache[event_id]

                        # trace_ray does not handle bad events very well.. skipping them should be OK?
                        try:
                            raypath = traveltime.trace_ray(coords)
                            dataset[:, idx] = raypath.T
                            traced[idx] = True

                            # Check for empty paths
                            if len(raypath) < 1:
                                logger.warning(f"Empty raypath for event_id {event_id}, {network}, {station}, {phase}")
                                continue

                            # Ensure consistent shape
                            if raypath.ndim != 2 or raypath.shape[1] != 3:
                                logger.warning(f"Invalid raypath shape {raypath.shape} for event_id {event_id}")
                                continue

                        except Exception as e:
                            logger.warning(f"traveltime issue with event_id {event_id} {network}.{station}.{phase}"
                                           f": {e} setting to a safe value   ###")
                            try:
                                max_val = np.max(traveltime.values[np.isfinite(traveltime.values)])
                                traveltime.values[~np.isfinite(traveltime.values)] = 1.01 * max_val
                                raypath = traveltime.trace_ray(coords)
                                dataset[:, idx] = raypath.T
                                traced[idx] = True
                            except Exception as e:
                                logger.warning(f"...couldn't fix traveltime: {e}")
                                continue

                    if not np.array_equal(traced,traced0):
                        raypath_file["traced"][:] = traced
                    raypath_file.close()

        COMM.barrier()
        return True


    @_utilities.log_errors(logger)
    def track_residual_improvement(self, safe_improvement=-0.03, safe_residual=None):
        """
        Track arrival and event residuals per iteration,
        remove if above safe_residual AND not improving. Meant to just remove clearly bad stuff.

        safe_improvement is a fraction (e.g. -0.05 = -5%).
          Arrivals/Events that improve by at least this much are safe regardless.

        safe_residual in seconds, applies for both event/arrival

        If safe_residual not set (default), assume: mean with floor of 0.25 seconds
            (e.g. residuals under this don't _need_ to improve)
        """

        if RANK == ROOT_RANK:
            logger.info(f"Tracking residuals for iteration {self.iiter}...   ###")

            # Ease into it..
            if self.iiter <= 1:
                n_std = 3.0
            elif self.iiter <= 2:
                n_std = 2.8
            elif self.iiter <= 3:
                n_std = 2.6
            elif self.iiter <= 4:
                n_std = 2.4
            elif self.iiter <= 5:
                n_std = 2.2
            else:
                n_std = 2.0

            # soften a bit for events?
            n_std_event = n_std + 0.2

            current_arrival_ids = set(self.arrivals['arrival_id'])
            current_event_ids = set(self.events['event_id'])

            # Update only for arrivals/events that still exist
            history_arrival_mask = self.arrival_history['arrival_id'].isin(current_arrival_ids)
            self.arrival_history = self.arrival_history[history_arrival_mask].reset_index(drop=True)

            history_event_mask = self.event_history['event_id'].isin(current_event_ids)
            self.event_history = self.event_history[history_event_mask].reset_index(drop=True)

            # Add residuals for current iteration / create a mapping for efficient lookup
            arrival_residual_map = dict(zip(self.arrivals['arrival_id'], self.arrivals['residual']))
            event_residual_map = dict(zip(self.events['event_id'], self.events['residual']))

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


            ##### Arrival improvements
            prev_arrival_residuals = self.arrival_history[prev_col].abs() # we want the abs value for these !
            curr_arrival_residuals = self.arrival_history[curr_col].abs()

            # if safe_residual is None, then set to mean
            if not safe_residual:
                mean_residual = np.mean(curr_arrival_residuals)
                #std_residual = np.std(curr_arrival_residuals)
                safe_arr_residual = max(0.3, mean_residual)
            else:
                safe_arr_residual = safe_residual

            # Only check improvement for arrivals with residuals above safe_arr_residual
            significant_arrival_mask = prev_arrival_residuals > safe_arr_residual
            arrival_improvement = (prev_arrival_residuals - curr_arrival_residuals) / (prev_arrival_residuals + 1e-6)

            significant_improvements = arrival_improvement[significant_arrival_mask]
            if len(significant_improvements) > 0:
                mean_improvement = np.mean(significant_improvements)
                std_improvement = np.std(significant_improvements)
                improvement_threshold = mean_improvement - n_std * std_improvement
            else:
                improvement_threshold = -np.inf

            outlier_mask = (
                significant_arrival_mask &
                (arrival_improvement < improvement_threshold) &
                (arrival_improvement < safe_improvement)
            )
            arrivals_to_remove = self.arrival_history.loc[outlier_mask, 'arrival_id'].values

            # Calculate mean improvement using mean of residuals
            mean_prev_arrival = self.arrival_history[prev_col].abs().mean()
            mean_curr_arrival = self.arrival_history[curr_col].abs().mean()
            mean_arrival_improvement = ((mean_prev_arrival - mean_curr_arrival) / mean_prev_arrival) * 100 if mean_prev_arrival > 0 else 0
            logger.info(f"Mean arrival residual reduction: {mean_arrival_improvement:.2f}% "
                   f"({mean_prev_arrival:.4f} -> {mean_curr_arrival:.4f})   ###")


            ##### Event improvements
            prev_event_residuals = self.event_history[prev_col].abs()
            curr_event_residuals = self.event_history[curr_col].abs()

            # if safe_residual is None, then set to 1 std below mean
            if not safe_residual:
                mean_residual = np.mean(curr_event_residuals)
                #std_residual = np.std(curr_event_residuals)
                safe_evt_residual = max(0.2, mean_residual)
            else:
                safe_evt_residual = safe_residual

            # Only check improvement for events with residuals above safe_evt_residual
            significant_event_mask = prev_event_residuals > safe_evt_residual
            event_improvement = (prev_event_residuals - curr_event_residuals) / (prev_event_residuals + 1e-6)

            significant_event_improvements = event_improvement[significant_event_mask]
            if len(significant_event_improvements) > 0:
                mean_event_improvement = np.mean(significant_event_improvements)
                std_event_improvement = np.std(significant_event_improvements)
                event_improvement_threshold = mean_event_improvement - n_std_event * std_event_improvement
            else:
                event_improvement_threshold = -np.inf

            event_outlier_mask = (
                significant_event_mask &
                (event_improvement < event_improvement_threshold) &
                (event_improvement < safe_improvement)
            )
            events_to_remove = self.event_history.loc[event_outlier_mask, 'event_id'].values

            # Calculate mean improvement using mean of residuals
            mean_prev_event = self.event_history[prev_col].abs().mean()
            mean_curr_event = self.event_history[curr_col].abs().mean()
            mean_event_improvement = ((mean_prev_event - mean_curr_event) / mean_prev_event) * 100 if mean_prev_event > 0 else 0
            logger.info(f"Mean event residual reduction: {mean_event_improvement:.2f}% "
                   f"({mean_prev_event:.4f} -> {mean_curr_event:.4f})   ###")

            # Remove non-improving events FIRST (and their associated arrivals)
            # HOWEVER! Don't remove any on the first pass. But DO remove bad arrivals on the first pass.
            # Events may be good but have enough bad picks to affect this.
            # If removing arrivals puts events below minpicks thresh they will be booted in resanitize_data
            if self.iiter > 1 and len(events_to_remove) > 0:
                # Sanity check
                if len(events_to_remove) > 0.3 * len(self.events):
                    logger.warning(f"Removed {len(events_to_remove)/(len(self.events)+.1)*100:.2f}% of total events!   ###")

                # Print the dropped source_ids for later review
                dropped_events = self.events[self.events['event_id'].isin(events_to_remove)]
                logger.info(f"Removing {len(dropped_events)} events with source_ids: {dropped_events['source_id'].tolist()}   ###")
                dropped_events[['source_id', 'residual']].to_csv(
                    self.cfg["model"]["output_dir"]+'/dropped_events.txt', 
                    mode='a', header=False, index=False, sep=' ')

                # Remove events
                self.events = self.events[~self.events['event_id'].isin(events_to_remove)].reset_index(drop=True)

                # Remove all arrivals associated with removed events
                arrivals_before = len(self.arrivals)
                self.arrivals = self.arrivals[~self.arrivals['event_id'].isin(events_to_remove)].reset_index(drop=True)
                arrivals_removed_by_event = arrivals_before - len(self.arrivals)

                # Update arrival history to remove arrivals from deleted events
                arrival_event_mask = ~self.arrival_history['event_id'].isin(events_to_remove)
                self.arrival_history = self.arrival_history[arrival_event_mask].reset_index(drop=True)

                logger.info(f"Removed {len(events_to_remove)} non-improving events "
                            f"(threshold: {event_improvement_threshold*100:.1f}%, n_std: {n_std_event:.1f}) "
                            f"and {arrivals_removed_by_event} associated arrivals. {len(self.events)} events remain.   ###")

            # Remove individual non-improving arrivals (only those not already removed)
            # We may need to add a caveat for single-arrival events (e.g. teleseisms) TODO
            if len(arrivals_to_remove) > 0:
                # Sanity check
                remaining_arrival_ids = set(self.arrivals['arrival_id'])
                arrivals_to_remove = [aid for aid in arrivals_to_remove if aid in remaining_arrival_ids]

                # Filter out arrival_ids that were already removed with their events
                if len(arrivals_to_remove) > 0:
                    if len(arrivals_to_remove) > 0.3 * len(self.arrivals):
                        logger.warning(f"Removed {len(arrivals_to_remove)/(len(self.arrivals)+.1)*100:.2f}% of total arrivals!   ###")

                    self.arrivals = self.arrivals[~self.arrivals['arrival_id'].isin(arrivals_to_remove)].reset_index(drop=True)

                    # Update arrival history
                    arrival_mask_still_valid = self.arrival_history['arrival_id'].isin(self.arrivals['arrival_id'])
                    self.arrival_history = self.arrival_history[arrival_mask_still_valid].reset_index(drop=True)
                    logger.info(f"Removed {len(arrivals_to_remove)} additional non-improving arrivals "
                                f"(threshold: {improvement_threshold*100:.1f}%, n_std: {n_std:.1f})   ###")

        self.synchronize(attrs=["arrivals","arrival_history","events","event_history"])
        return True

    # new! track individual picks
    @_utilities.log_errors(logger)
    def analyze_pick_residuals(self):
        """
        Identify individual (event, station, phase) picks whose residuals are
        persistently large AND consistent in sign across iterations, relative
        to other picks at the same station/phase.

        Writes <output_dir>/NN.bad_picks.csv with one row per flagged pick
        and optionally removes them from self.arrivals.

        Must be called AFTER track_residual_improvement() so the current
        iteration's signed residual has been appended to arrival_history.

        Config keys (under cfg["analyze"], all optional):
          pick_start_iter         : int,   default 3
          pick_min_iters_present  : int,   default 3
          pick_drop               : bool,  default False
          pick_median_threshold   : float, default 0.75   (seconds)
          pick_mad_max            : float, default 0.15  (seconds)
          pick_station_excess     : float, default 3.5
          pick_max_drop_fraction  : float, default 0.02
        """
        cfg = self.cfg["analyze"]
        start_iter        = cfg.get("pick_start_iter") # default 3
        min_iters_present = cfg.get("pick_min_iters_present") # default 3
        do_drop           = cfg.get("pick_drop") # default False
        med_thresh        = cfg.get("pick_median_threshold") # default 0.5
        mad_max           = cfg.get("pick_mad_max") # default 0.5
        station_excess    = cfg.get("pick_station_excess") # default 2.5
        max_drop_frac     = cfg.get("pick_max_drop_fraction") # default 0.10 (10%)
        pick_scale_k      = cfg.get("pick_scale_k") # default 3.5

        iter_cols = [c for c in self.arrival_history.columns if c.startswith("iter_")]
        usable_iter_cols = [c for c in iter_cols if int(c.split("_")[1]) >= start_iter]

        # Only starts tracking after back_pick_start_iter, and only starts counting after min_iters_persent (e.g. 6th iteration?)
        if len(usable_iter_cols) < min_iters_present:
            if RANK == ROOT_RANK:
                logger.info(f"Analyze_pick_residuals: Not enough usable iterations yet ({len(usable_iter_cols)} < {min_iters_present}); skipping.   ###")
            return True

        # All analysis runs on ROOT_RANK; results broadcast at the end if we drop.
        if RANK == ROOT_RANK:
            logger.info(f"Analyzing per-pick residuals at iter {self.iiter}...   ###")

            hist = self.arrival_history.copy()
            vals = hist[usable_iter_cols].to_numpy(dtype=float)
            n_present = np.sum(~np.isnan(vals), axis=1)

            # ---------------------------------------------------------------
            # Robust bilinear demeaning of the residual matrix.
            #
            # The raw post-relocation residual of pick i at iteration k is
            # approximately:
            #     r_ik = e_jk (event common-mode)  +  s_m (station static)
            #          + p_i (true pick error)     +  noise
            # Events contaminated by bad picks relocate to absorb them, which
            # puts a persistent, consistent-sign common-mode offset e_jk on
            # ALL of that event's picks. Flagging on the raw residual then
            # (a) flags the event's GOOD picks (false positives) and
            # (b) hides the bad pick, whose offset was partially absorbed.
            # Removing the per-event median and per-station-phase median
            # (two alternating robust passes) isolates the pick term p_i.
            # ---------------------------------------------------------------
            demean = bool(cfg.get("pick_demean", True)) ## TODO ADD TO CFG
            if demean:
                dm = pd.DataFrame(vals, copy=True)
                ev_key = hist["event_id"].values
                st_key = (hist["network"].astype(str) + "." +
                          hist["station"].astype(str) + "." +
                          hist["phase"].astype(str)).values
                for _ in range(2):
                    dm = dm - dm.groupby(ev_key).transform("median")
                    dm = dm - dm.groupby(st_key).transform("median")
                vals_dm = dm.to_numpy(dtype=float)
            else:
                vals_dm = vals

            # Per-pick robust statistics over the usable iteration window
            # (computed on the demeaned matrix)
            pick_median = np.nanmedian(vals_dm, axis=1)
            # MAD scaled to ~sigma-equivalent for a normal distribution
            pick_mad = 1.4826 * np.nanmedian(
                np.abs(vals_dm - pick_median[:, None]), axis=1
            )
            # Raw (un-demeaned) median kept for the diagnostic CSV
            pick_median_raw = np.nanmedian(vals, axis=1)

            event_time_map = dict(zip(self.events["event_id"], self.events["time"]))
            pick_time_map  = dict(zip(self.arrivals["arrival_id"], self.arrivals["time"]))

            stats = pd.DataFrame({
                "arrival_id":   hist["arrival_id"].values,
                "event_id":     hist["event_id"].values,
                "event_time":   hist["event_id"].map(event_time_map).values,
                "pick_time":    hist["arrival_id"].map(pick_time_map).values, 
                "network":      hist["network"].values,
                "station":      hist["station"].values,
                "phase":        hist["phase"].values,
                "n_iter":       n_present,
                "median_resid": pick_median,
                "median_resid_raw": pick_median_raw,
                "mad_resid":    pick_mad,
            })

            # Per-station-phase median of |median_resid|. This is our local
            # baseline; we only flag picks that stand out FROM THEIR OWN STATION.
            # (After demeaning this is mostly a residual safety net; station
            # statics have already been removed.)
            stats["abs_median"] = stats["median_resid"].abs()
            station_baseline = (
                stats.groupby(["network", "station", "phase"])["abs_median"].transform("median")
            )
            stats["station_baseline"] = station_baseline

            # Adaptive detection floor: the population MAD of the demeaned
            # pick medians estimates the effective pick-noise scale of THIS
            # dataset. The user threshold acts as a minimum, but if the data
            # are noisier than the configured threshold we raise the floor
            # instead of flooding the flags with noise-level picks.
            pop_med  = np.nanmedian(stats["median_resid"])
            noise_scale = 1.4826 * np.nanmedian(
                np.abs(stats["median_resid"] - pop_med)
            )
            eff_thresh = max(med_thresh, pick_scale_k * noise_scale)

            # Flag logic (all four must hold):
            #  - present in enough iterations
            #  - large signed median (consistent miss in one direction)
            #  - stable: scatter small relative to BOTH an absolute cap and
            #    the offset itself. A 2 s bad pick on an event whose
            #    relocation wanders (mad ~0.5 s) is still obviously bad; a
            #    hard mad<=mad_max gate would veto exactly those true
            #    positives while passing common-mode-stabilized good picks.
            #  - notably worse than the station's typical pick
            eligible = stats["n_iter"] >= min_iters_present
            large    = stats["abs_median"] >= eff_thresh
            tight    = stats["mad_resid"] <= np.maximum(
                mad_max, 0.5 * stats["abs_median"]
            )
            excess   = stats["abs_median"] >= station_excess * (station_baseline + 1e-6)

            stats["flag_bad_pick"] = eligible & large & tight & excess
            logger.info(
                f"analyze_pick_residuals: noise_scale={noise_scale:.3f}s, "
                f"effective |median| threshold={eff_thresh:.3f}s   ###"
            )
            # Keep for event diagnostics
            self._pick_stats = stats

            n_flagged = int(stats["flag_bad_pick"].sum())
            n_total   = len(stats)
            frac      = n_flagged / max(n_total, 1)

            logger.info(
                f"Bad-pick flags: {n_flagged}/{n_total} ({100*frac:.2f}%)  "
                f"thresholds: |med|>={med_thresh}, mad<={mad_max}, "
                f"station_excess>={station_excess}x   ###"
            )

            # Always write the diagnostic CSV (whether or not we drop)
            flagged = stats[stats["flag_bad_pick"]].sort_values(
                "abs_median", ascending=False
            )
            out_path = os.path.join(
                self.cfg["model"]["output_dir"],
                f"{self.iiter:02d}.bad_picks.csv"
            )
            cols_out = ["arrival_id", "event_id", "event_time", "pick_time",
                        "network", "station", "phase",
                        "n_iter", "median_resid", "median_resid_raw",
                        "mad_resid", "station_baseline"]

            out = flagged[cols_out].copy()

            # Format epoch-second timestamps as ISO 8601 (UTC, millisecond precision)
            out["event_time"] = pd.to_datetime(out["event_time"], unit="s", utc=True, errors="coerce")
            out["event_time"] = out["event_time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z"
            out["event_time"] = out["event_time"].fillna("")

            out["pick_time"] = pd.to_datetime(out["pick_time"], unit="s", utc=True, errors="coerce")
            out["pick_time"] = out["pick_time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z"
            out["pick_time"] = out["pick_time"].fillna("")

            # Round float residual columns only (don't apply .round() to whole frame)
            for c in ["median_resid", "median_resid_raw", "mad_resid", "station_baseline"]:
                out[c] = out[c].round(4)
            out.to_csv(out_path, index=False)
            logger.info(f"Wrote bad-pick diagnostics to {out_path}   ###")

            # Safety: refuse to drop too many in one shot
            if do_drop:
                if frac > max_drop_frac:
                    logger.warning(
                        f"Would drop {100*frac:.2f}% of arrivals "
                        f"(> {100*max_drop_frac:.2f}% cap). "
                        f"NOT dropping. Inspect {out_path} and tighten thresholds "
                        f"or raise pick_max_drop_fraction.   ###"
                    )
                elif n_flagged > 0:
                    ids_to_drop = flagged["arrival_id"].values
                    if "truepick" in self.arrivals.columns:
                        protected_ids = self.arrivals.loc[
                            self.arrivals["truepick"].fillna(False).astype(bool),
                            "arrival_id"
                        ].values
                        n_protected = np.isin(ids_to_drop, protected_ids).sum()
                        if n_protected > 0:
                            logger.info(
                                f"analyze_pick_residuals: {n_protected} "
                                f"flagged picks are truepicks; NOT dropping them   ###"
                            )
                            ids_to_drop = ids_to_drop[
                                ~np.isin(ids_to_drop, protected_ids)
                            ]
                    before = len(self.arrivals)
                    self.arrivals = self.arrivals[
                        ~self.arrivals["arrival_id"].isin(ids_to_drop)
                    ].reset_index(drop=True)
                    self.arrival_history = self.arrival_history[
                        ~self.arrival_history["arrival_id"].isin(ids_to_drop)
                    ].reset_index(drop=True)
                    logger.info(
                        f"Dropped {before - len(self.arrivals)} bad-pick arrivals (of {before}).   ###"
                    )

        self.synchronize(attrs=["arrivals", "arrival_history"])
        return True


    @_utilities.log_errors(logger)
    @_utilities.root_only(RANK)
    def save_event_diagnostics(self):
        """
        Save a per-event diagnostic table flagging events that may need review.
        Outputs a CSV sorted by number of flags (most suspicious first).
        """
        logger.info("Saving event diagnostics...   ###")

        threshold_residual = self.cfg["analyze"].get("event_residual_threshold", 0.6)
        threshold_weight   = self.cfg["analyze"].get("event_weight_threshold", 0.01)
        threshold_std      = self.cfg["analyze"].get("event_std_threshold", 0.5)

        events   = self.events.copy()
        arrivals = self.arrivals.copy()

        # Robust per-event statistics. Mean/std are dominated by a single
        # gross pick: one 4 s outlier among 20 arrivals adds 0.2 s to the
        # mean |residual| and blows up the std, flagging an otherwise
        # well-located event. Median/MAD answer "is the EVENT bad?", while
        # the pick-level analysis answers "does the event contain bad picks?".
        def _mad(x):
            x = np.asarray(x, dtype=float)
            m = np.nanmedian(x)
            return 1.4826 * np.nanmedian(np.abs(x - m))

        event_stats = arrivals.groupby('event_id').agg(
            n_arrivals      = ('arrival_id', 'count'),
            mean_residual   = ('residual', 'mean'),
            median_residual = ('residual', 'median'),
            abs_residual    = ('residual', lambda x: x.abs().median()),
            std_residual    = ('residual', 'std'),
            mad_residual    = ('residual', _mad),
            mean_weight     = ('weight', 'mean'),
        ).reset_index()

        events = events.merge(event_stats, on='event_id', how='left')

        # Fold in pick-level flags if analyze_pick_residuals has run:
        # contaminated-but-well-located events get their own flag instead of
        # polluting the location-quality flags.
        pick_stats = getattr(self, "_pick_stats", None)
        if pick_stats is not None:
            n_flagged_picks = (
                pick_stats.groupby("event_id")["flag_bad_pick"].sum()
                .rename("n_flagged_picks")
            )
            events = events.merge(n_flagged_picks, on="event_id", how="left")
            events["n_flagged_picks"] = events["n_flagged_picks"].fillna(0).astype(int)
        else:
            events["n_flagged_picks"] = 0

        events['flag_high_residual'] = events['abs_residual'] > threshold_residual
        events['flag_low_weight']    = events['mean_weight']  < threshold_weight
        events['flag_few_arrivals']  = events['n_arrivals']   < self.cfg["algorithm"]["min_narrival"]
        events['flag_high_std']      = events['mad_residual'] > threshold_std
        events['flag_bad_picks']     = events['n_flagged_picks'] > 0
        events['n_flags'] = events[['flag_high_residual', 'flag_low_weight',
                                     'flag_few_arrivals',  'flag_high_std',
                                     'flag_bad_picks']].sum(axis=1)

        events = events.round(5) # round sig figs to something sane
        # put event_id as column 1
        cols = ['event_id'] + [c for c in events.columns if c != 'event_id']
        events = events[cols]

        path = os.path.join(self.cfg["model"]["output_dir"], "event_diagnostics.csv")
        events.sort_values('n_flags', ascending=False).to_csv(path, index=False)
        logger.info(f"Saved event diagnostics to {path}   ###")

        return True


    @_utilities.log_errors(logger)
    @_utilities.root_only(RANK)
    def save_station_diagnostics(self):
        """
        Save a per-station diagnostic table flagging stations that may need review.
        """
        logger.info("Saving station diagnostics...   ###")

        threshold_residual = self.cfg["analyze"].get("station_residual_threshold", 0.5)
        threshold_std      = self.cfg["analyze"].get("station_std_threshold", 0.5)

        arrivals = self.arrivals.copy()

        def _mad(x):
            x = np.asarray(x, dtype=float)
            m = np.nanmedian(x)
            return 1.4826 * np.nanmedian(np.abs(x - m))

        station_stats = arrivals.groupby(['network', 'station', 'phase']).agg(
            n_arrivals      = ('arrival_id', 'count'),
            mean_residual   = ('residual', 'mean'),
            median_residual = ('residual', 'median'),
            abs_residual    = ('residual', lambda x: x.abs().median()),
            std_residual    = ('residual', 'std'),
            mad_residual    = ('residual', _mad),
            mean_weight     = ('weight', 'mean'),
        ).reset_index()

        station_stats['flag_high_residual'] = station_stats['abs_residual'] > threshold_residual
        station_stats['flag_high_std']      = station_stats['mad_residual'] > threshold_std
        station_stats['flag_high_mean']     = station_stats['median_residual'].abs() > threshold_residual  # systematic bias
        station_stats['n_flags'] = station_stats[['flag_high_residual',
                                                   'flag_high_std',
                                                   'flag_high_mean']].sum(axis=1)

        station_stats = station_stats.round(5)  # round sig figs to something sane
        station_stats = station_stats.sort_values('n_flags', ascending=False)

        path = os.path.join(self.cfg["model"]["output_dir"], "station_diagnostics.csv")
        station_stats.to_csv(path, index=False)
        logger.info(f"Saved station diagnostics to {path}   ###")

        return True


    @_utilities.log_errors(logger)
    def update_event_weights(self,npts=16):
        """
        Update events weights using KDE for homogeneous raypath sampling

        Args:
            npts: Number of points for KDE grid evaluation (16 is fine)
        """
        logger.info("Updating event KDE weights for raypath sampling   ###")

        if RANK == ROOT_RANK:
            events = self.events
            kde_columns = ["latitude", "longitude", "depth"]
            ndim = len(kde_columns)
            data = events[kde_columns].values

            # IQR normalization
            data_iqr = iqr(data, axis=0)
            data_median = np.median(data, axis=0)

            # Handle zero-variance dimensions
            mask_zero_iqr = data_iqr == 0
            if np.any(mask_zero_iqr):
                logger.warning(f"Zero variance detected in dimensions: {np.array(kde_columns)[mask_zero_iqr]}   ###")
                data_iqr[mask_zero_iqr] = np.median(data_iqr[~mask_zero_iqr])

            # Normalize
            data_normalized = (data - data_median) / data_iqr

            # Compute robust bandwidth using Scott's rule (which seems to be a bit larger than Silverman)
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
                    fill_value=np.max(values)
                )

                # Compute densities at data points
                densities = interpolator(data_normalized)

                if self.iiter < 2:
                    # Strong inverse weighting
                    weights = 1.0 / densities
                elif self.iiter < 4:
                    # Transition to exp-based weighting
                    weights = 1.0 / np.exp(densities)
                else:
                    # Taper from density-weighted to uniform over remaining iterations
                    progress = (self.iiter - 4) / (self.niter - 4)  # 0 at iter 4, 1 at final iter
                    progress = min(progress, 1.0)
                    density_weight = 1.0 / np.exp(densities)
                    uniform_weight = np.ones_like(densities)
                    weights = (1 - progress) * density_weight + progress * uniform_weight

                # Clip to 95th percentile to prevent outlier events dominating
                weight_cap = np.percentile(weights[np.isfinite(weights)], 95)

                # Monitor how weights are being capped
                uncapped_max = weights[np.isfinite(weights)].max()
                cap_ratio = uncapped_max / weight_cap  # how much the highest weight exceeds the cap
                if cap_ratio > 5.0:
                    logger.info(f"  95th percentile weight cap={weight_cap:.2f}, max_uncapped={uncapped_max:.2f} ({cap_ratio:.1f}x cap)   ###")

                # Cap & Normalise
                weights = np.minimum(weights, weight_cap)

                # Ground-truth picks always carry the maximum weight so
                # they are preferentially drawn in every realization /// NOT IN EVENTS KDE!!
                #if "truepick" in arrivals.columns:
                #    is_true = arrivals["truepick"].fillna(False).astype(bool).values
                #    if is_true.any():
                #        weights[is_true] = weight_cap
                weight_sum = weights.sum()
                if weight_sum > 0:
                    weights = weights / weight_sum
                else:
                    logger.warning(f"  KDE event weights sum to zero for (!), using uniform weights   ###")
                    weights = np.ones(len(weights)) / len(weights)

                # Set any problem infinite or NaN values to 0
                bad_values = ~np.isfinite(weights)
                if np.any(bad_values):
                    logger.warning(f"Found {np.sum(bad_values)} infinite/NaN in KDE event weights, setting to 0   ###")
                weights[bad_values] = 0

                # Set em!
                events["weight"] = weights
                self.events = events

                logger.info(f"  (event bandwidth = {bandwidth:.2f})   ###")

                # >>> COMPLETELY SEPARATE RESIDUAL ANALYSIS but may as well warn about it here (TODO organize sanely?)

                # Check for NaN values and warn
                residuals = events['residual']
                nan_count = residuals.isna().sum()
                if nan_count > 0:
                    logger.warning(f"Found {nan_count} events with NaN residuals   ###")

                # Calculate/print residual statistics
                valid_residuals = residuals.dropna()
                if len(valid_residuals) > 0:
                    logger.info(
                        f"mean event residual (s): {valid_residuals.mean():.4f} "
                        f"({valid_residuals.std():.4f} std)   ###"
                    )
                else:
                    logger.warning("No valid event residuals found - all are NaN!!!   ###")

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
                # Get parameters
                max_arr_resid = self.cfg["algorithm"]["max_arrival_residual"]

                arrivals = self.arrivals[self.arrivals["phase"] == phase]

                # Merge event data with renamed columns
                events = self.events.rename(columns={
                    "latitude": "event_latitude",
                    "longitude": "event_longitude",
                    "depth": "event_depth"
                })
                merge_columns = ["event_latitude", "event_longitude", "event_depth", "event_id"]
                arrivals = arrivals.merge(events[merge_columns], on="event_id")

                # Merge station data with renamed columns
                stations = self.stations.rename(columns={
                    "latitude": "station_latitude",
                    "longitude": "station_longitude"
                })
                merge_columns = ["station_latitude", "station_longitude", "network", "station"]
                merge_keys = ["network", "station"]
                arrivals = arrivals.merge(stations[merge_columns], on=merge_keys)

                # Compute ray geometry
                dlat = arrivals["event_latitude"] - arrivals["station_latitude"]
                dlon = arrivals["event_longitude"] - arrivals["station_longitude"]
                arrivals["azimuth"] = np.arctan2(dlat, dlon)

                # Now done in re-sanitize_data also but whatever (TODO sanity check)
                # Note that using km is more sound for KDE especially near the poles
                arrivals["delta"] = _utilities.dist_km(
                    arrivals["event_latitude"],
                    arrivals["event_longitude"],
                    arrivals["station_latitude"],
                    arrivals["station_longitude"])

                # Prepare data for KDE
                kde_columns = ["event_latitude", "event_longitude", "event_depth", "azimuth", "delta"]
                ndim = len(kde_columns)
                data = arrivals[kde_columns].values

                # IQR normalization
                data_iqr = iqr(data, axis=0)
                data_median = np.median(data, axis=0)

                # Handle zero-variance dimensions
                mask_zero_iqr = data_iqr == 0
                if np.any(mask_zero_iqr):
                    logger.warning(f"Zero variance detected in dimensions: {np.array(kde_columns)[mask_zero_iqr]}   ###")
                    data_iqr[mask_zero_iqr] = np.median(data_iqr[~mask_zero_iqr])

                data_normalized = (data - data_median) / data_iqr

                # Compute Scott's bandwidth
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
                    fill_value=np.max(values)
                )
                densities = interpolator(data_normalized)

                # For arrivals, just use the same weight scheme the whole time
                weights = 1 / np.exp(densities)

                # Clip to 95th percentile to prevent outlier arrivals dominating
                weight_cap = np.percentile(weights[np.isfinite(weights)], 95)

                # Monitor how weights are being capped
                uncapped_max = weights[np.isfinite(weights)].max()
                cap_ratio = uncapped_max / weight_cap  # how much the highest weight exceeds the cap
                if cap_ratio > 2.0:
                    logger.info(f"  95th percentile weight cap={weight_cap:.2f}, max_uncapped={uncapped_max:.2f} ({cap_ratio:.1f}x cap)   ###")

                # Cap & Normalise
                weights = np.minimum(weights, weight_cap)

                # Ground-truth picks always carry the maximum weight so
                # they are preferentially drawn in every realization
                if "truepick" in arrivals.columns:
                    is_true = arrivals["truepick"].fillna(False).astype(bool).values
                    if is_true.any():
                        weights[is_true] = weight_cap

                weight_sum = weights.sum()
                if weight_sum > 0:
                    weights = weights / weight_sum
                else:
                    logger.warning(f"  KDE arrival weights sum to zero for {phase} (!), using uniform weights   ###")
                    weights = np.ones(len(weights)) / len(weights)

                # Set any infinite or NaN values to 0
                bad_values = ~np.isfinite(weights)
                if np.any(bad_values):
                    logger.warning(f"Found {np.sum(bad_values)} infinite/NaN in KDE arrival weights, setting to 0   ###")
                weights[bad_values] = 0

                # Set em!
                arrivals["weight"] = weights

                # Update self.arrivals with new weights
                index_columns = ["network", "station", "event_id", "phase"]
                arrivals = arrivals.set_index(index_columns)
                _arrivals = self.arrivals.set_index(index_columns).sort_index()
                _arrivals.loc[arrivals.index, "weight"] = arrivals["weight"]
                self.arrivals = _arrivals.reset_index()

                # Log statistics
                valid_arrivals = arrivals[abs(arrivals['residual']) <= max_arr_resid]['residual']
                logger.info(f"  ({phase} arrival bandwidth = {bandwidth:.2f})   ###")
                logger.info(
                    f"mean {phase} arrival residual (s)     : "
                    f"{valid_arrivals.mean():.4f} ({valid_arrivals.std():.4f} std)   ###"
                )
                logger.info(
                    f"   mean abs arrival residual (s): "
                    f"{valid_arrivals.abs().mean():.4f} ({valid_arrivals.abs().std():.4f} std)   ###"
                )

            except Exception as e:
                logger.error(f"Failed to update arrival weights: {str(e)}")
                return False

        self.synchronize(attrs=["arrivals"])
        return True


    @_utilities.log_errors(logger)
    def _update_projection_matrix(self, phase, hvr):
        """
        Update the projection matrix using the current Voronoi cells

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

            # Transform voronoi cells
            voronoi_cells = self.voronoi_cells
            voronoi_cells = center + (voronoi_cells - center) / [1, hvr, hvr]
            voronoi_cells = sph2xyz(voronoi_cells)
            tree = cKDTree(voronoi_cells)

            # Transform nodes
            nodes = model.nodes
            nodes = center + (nodes - center) / [1, hvr, hvr]
            nodes = nodes.reshape(-1, 3)
            nodes = sph2xyz(nodes)

            # Get mapping from nodes to voronoi cells
            _, column_ids = tree.query(nodes)
            nnodes = np.prod(model.nodes.shape[:-1])
            row_ids = np.arange(nnodes)
            values = np.ones(nnodes,)

            # Create base projection matrix
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

        # Sometimes makes sense to just calculate the specific phase tables
        run_phases = run_phases or self.phases

        # Compute a global max traveltime from the data, +8% margin
        # Pass to solver to truncate FMM beyond what we'll ever need.
        if RANK == ROOT_RANK:
            event_time = dict(zip(self.events["event_id"], self.events["time"]))
            tt_observed = (
                self.arrivals["time"] - self.arrivals["event_id"].map(event_time)
            )
            max_tt_data = float(tt_observed.quantile(0.999)) # 99.9% just in case one wild pick in there
            max_tt = max_tt_data* 1.08 # 8% higher
            logger.info(
                f"  max observed traveltime: {max_tt_data:.2f}s, solver tt cap at {max_tt:.1f}   ###"
            )
        else:
            max_tt = None
        max_tt = COMM.bcast(max_tt, root=ROOT_RANK)

        # Build the (station, phase) work list from what the post-QC catalog
        # actually needs, not every station in the geometry file.
        arrivals_by_phase = {
            p: self.arrivals[self.arrivals["phase"] == p] for p in run_phases
        }
        needed_by_phase = {
            p: set(zip(a["network"], a["station"]))
            for p, a in arrivals_by_phase.items()
        }
        all_geometry_ids = set(zip(self.stations["network"], self.stations["station"]))
        needed_any_phase = set().union(*needed_by_phase.values())
        needed_by_phase = COMM.bcast(needed_by_phase, root=ROOT_RANK)

        if RANK == ROOT_RANK:
            skipped = all_geometry_ids - needed_any_phase
            if skipped:
                logger.info(
                    f"  skipping tt creation of {len(skipped)}/{len(all_geometry_ids)} stations "
                    f"with no post-QC arrivals in any phase   ###"
                )
            logger.info(f"  Building traveltimes here: {traveltime_dir}")
            os.makedirs(traveltime_dir, exist_ok=True)
            ids = sorted(needed_any_phase)
            self._dispatch(ids)
        else:
            geometry = self.stations
            geometry = geometry.set_index(["network", "station"])
            while True:
                item = self._request_dispatch()
                if item is None: break

                network, station = item
                keys = ["latitude", "longitude", "depth"]
                coords = geometry.loc[(network, station), keys]
                coords = geo2sph(coords)

                for phase in run_phases:
                    if (network, station) not in needed_by_phase[phase]: continue # don't compute tt's if not needed!
                    handle = f"{phase.lower()}wave_model"
                    model = getattr(self, handle)
                    solver = PointSourceSolver(coord_sys="spherical")
                    solver.vv.min_coords = model.min_coords
                    solver.vv.node_intervals = model.node_intervals
                    solver.vv.npts = model.npts
                    solver.vv.values = model.values
                    solver.src_loc = coords
                    solver.solve(max_traveltime=max_tt) # capping traveltime to some % above the max to save CPU

                    # Reset undefined to some max value in case some actual glitches show up
                    if np.any(~np.isfinite(solver.tt.values)):
                        logger.info(f"inf values found in solver.tt for {network}.{station}.{phase} ...no problem, setting these out of reach")
                        try:
                            max_val = np.max(solver.tt.values[np.isfinite(solver.tt.values)])
                            solver.tt.values[~np.isfinite(solver.tt.values)] = 1.01 * max_val
                        except Exception as e:
                            n_finite = int(np.sum(np.isfinite(solver.tt.values)))
                            if n_finite < 2:
                                logger.warning(
                                    f"could not fix {network}.{station}.{phase}: only "
                                    f"{n_finite} finite values in tt array ({e!r})"
                                )
                            else:
                                logger.warning(
                                    f"mystery issue with {network}.{station}.{phase} "
                                    f"(possible elevation/coord issue): {e!r}"
                                )
                            continue

                    path = os.path.join(traveltime_dir,f"{network}.{station}.{phase}.h5")
                    solver.tt.to_hdf(path)

        COMM.barrier()

        if RANK == ROOT_RANK:
            _path = self.traveltime_inventory_path
            if os.path.isfile(_path):
                os.remove(_path)

            # Distance-limited, compressed storage (new pykonal): each
            # station's grid is cropped to tt_crop_km. Auto mode derives
            # the crop from algorithm.max_dist (the tomography arrival
            # cutoff) with a 10% margin so no usable ray can graze the
            # crop edge. 
            # mask defaults to False here:
            # ray tracing descends the traveltime gradient and must not meet NaN node
            crop_km = self.cfg["algorithm"].get("tt_crop_km", -1)
            if crop_km is None or crop_km < 0:
                max_dist = self.cfg["algorithm"]["max_dist"] # km
                crop_km = max_dist * 1.10
            crop_km = None if crop_km == 0 else float(crop_km)
            crop_mask = False # bool(self.cfg["algorithm"].get("tt_crop_mask", False))

            _station_coords = _utilities.station_dict(self.stations)

            with TraveltimeInventory(_path, mode="w") as tt_inventory:
                pattern = os.path.join(traveltime_dir, "*.h5")
                paths = glob.glob(pattern)
                paths = sorted(paths)
                tt_inventory.merge(
                    paths,
                    station_coords=_station_coords,
                    max_dist=crop_km,
                    mask=crop_mask,
                    compress=True
                )
            if crop_km is not None:
                size_mb = os.path.getsize(_path) / 1e6
                logger.info(
                    f"traveltime inventory: {size_mb:.0f} MB with "
                    f"{crop_km:.0f} km distance limit (mask={crop_mask})   ###"
                )

            shutil.rmtree(self.traveltime_dir)

        COMM.barrier()
        return True


    @_utilities.log_errors(logger)
    def iterate(self):
        """
        Execute one iteration the entire inversion procedure including
        updating velocity models, event locations, and arrival residuals.
        """

        nreal = self.cfg["algorithm"]["nreal"]

        hvr = self.cfg["meshing"]["hvr"]
        min_rays_per_cell = self.cfg["meshing"]["min_rays_per_cell"]
        adaptive_weight = self.cfg["meshing"].get("adaptive_data_weight")
        adaptive_weight = max(0,min(adaptive_weight,1.0))
        density_to_gradient_weight = self.cfg["meshing"]["density_to_gradient_weight"]
        density_to_gradient_weight = max(0,min(density_to_gradient_weight,1.0))

        # ONLY doing the resolution test? (via -t flag)
        if self.argc.test_only:
            return self.run_resolution_test()

        self.iiter += 1

        try:
            phase_order = self.cfg["algorithm"]["phase_order"]
        except:
            phase_order =  ['P', 'S']

        logger.info(f"Iteration #{self.iiter} (/{self.niter}) ###")

        if self.cfg["argc"]["relocate_first"] == "False":
            self.sanitize_data()
        else:
            self.resanitize_data()

        if self.iiter == 1:
            self.save_stations() # save the first filtered station for sharing/etc
        
        # Determine if we will also compute a 1D velocity model on the last iteration
        do_compute_1d = False
        if self.iiter == self.niter:
            do_compute_1d = self.cfg["model"].get("output_1d_model", True)

        self.check_event_bounds()

        for phase in phase_order:
            logger.info(f" >>> Starting {phase}-wave iteration {self.iiter} of {self.niter} <<<")
            if do_compute_1d:
                logger.info("  * Also calculating 1D model on this iteration")

            self._reset_realization_stack(phase)
            self._estimate_velocity_gradient_density(phase) # new! define vel gradients for adaptive meshing

            self._prev_conda = None # hold the previous iteration's damping estimate for the following

            for self.ireal in range(nreal):
                logger.info(f"{phase} Realization # {self.ireal+1}/{nreal} | Iteration # {self.iiter}/{self.niter}")
                self._sample_events()
                self._sample_arrivals(phase)
                t0 = time.perf_counter()
                self._trace_rays(phase)
                trace_rays_time = time.perf_counter() - t0

                t0 = time.perf_counter()
                self._generate_voronoi_cells(phase)
                gen_voronoi_time = time.perf_counter() - t0
                if trace_rays_time > 120:
                    logger.info(f" Time elapsed: trace_rays {trace_rays_time/60:0.1f} min, generate_voronoi_cells {gen_voronoi_time:0.1f} s")
                else: 
                    logger.info(f" Time elapsed: trace_rays {trace_rays_time:0.1f} s, generate_voronoi_cells {gen_voronoi_time:0.1f} s") 
                self._compute_sensitivity_matrix(phase,hvr)
                self._diagnose_mesh(phase)
                self._update_projection_matrix(phase,hvr)
                self._compute_model_update(phase,min_rays=min_rays_per_cell,compute_1d=do_compute_1d)

                # Adaptive early stopping (all ranks; decision broadcast)
                if self._stack_converged(phase, nreal):
                    break

            self.update_model(phase,compute_1d=do_compute_1d)

            if not self.argc.test_only:
                self.save_model(phase, tag=None)

        self.compute_traveltime_lookup_tables() # without an argument, computes both phases
        self.relocate_events() # also calls update_arrival_residuals, update_event_weights, and update_arrival_weights
        self.track_residual_improvement() # track improvement of residuals and boot any gremlins TODO: can also be used to prematurely stop iterations?

        if self.iiter >= self.cfg["analyze"].get("pick_start_iter", 3):
            self.analyze_pick_residuals() # try to flag individual pick issues and write out to csv

        if self.iiter <= 3:
            self.check_event_migration() # implement a check to see if EQs have migrated a great deal (located very poorly to begin with!)

        self.purge_raypaths()
        self.resanitize_data()
        self.save_events() # n.b. the first 00.events.h5 is the initial relocated (-r) model, may be faster to start from this in the future

        # At the end of the final iteration, save event & station diagnostics to help identify problem areas
        if self.iiter == self.niter:
            self.save_event_diagnostics()
            self.save_station_diagnostics()


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
                    logger.info(f" {filter_name:>10s} bounds filter: {dropped_count:5d} events dropped ({after_count:5d} remaining)   ###")

            dropped_event_ids = set(events['event_id']) - set(filtered['event_id'])

            events = filtered[events.columns]
            self.events = events

            if len(dropped_event_ids) > 0:
                dn = len(dropped_event_ids)
                # also have to toss any arrivals referencing these dropped events
                arrivals = self.arrivals
                arrivals = arrivals[~arrivals['event_id'].isin(dropped_event_ids)]
                self.arrivals = arrivals
                logger.info(f"   ...Dropped {dn} total events which are out of bounds. {n0-dn} remain.   ###")

        self.synchronize(attrs=['events','arrivals'])
        return True


    @_utilities.log_errors(logger)
    def check_event_migration(self):
        """
        Remove events that have been runaway migrated beyond some tolerance
        """
        if RANK == ROOT_RANK:
            logger.info("Removing runaway event migrations.   ###")

            events = self.events
            n0 = len(events)
            events0 = self.events0 # e.g. input catalog
            merged = pd.merge(events0, events, on='event_id', suffixes=('_0', ''))

            # Calculate absolute differences
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

            # Apply filters and log results
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
                logger.info(f" {filter_name:>10s} migration filter: {dropped_count:5d} events dropped ({after_count:5d} remaining)   ###")

            dropped_event_ids = set(events['event_id']) - set(filtered['event_id'])

            events = filtered[events.columns]
            self.events = events

            if len(dropped_event_ids) > 0:
                dn = len(dropped_event_ids)
                # Also have to toss any arrivals referencing these dropped events (now done elsewhere)
                arrivals = self.arrivals
                arrivals = arrivals[~arrivals['event_id'].isin(dropped_event_ids)]
                self.arrivals = arrivals
                logger.info(f"   ...Dropped {dn} events which have migrated too far from original position. {n0-dn} remain.   ###")
                for ele in dropped_event_ids:
                    logger.debug(f"dropped event: %6d" % ele)

        self.synchronize(attrs=['events','arrivals'])
        return len(self.events) > 0


    @_utilities.log_errors(logger)
    def load_cfg(self):
        """
        Parse and store configuration-file parameters.

        ROOT_RANK parses configuration file and broadcasts contents to all other processes.
        """
        if RANK == ROOT_RANK:
            logger.info("Loading configuration parameters")

            # Parse configuration file parameters
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

            data = _dataio.parse_event_data(self.cfg)
            self.events, self.arrivals = data

            self.arrival_history = pd.DataFrame({
                'arrival_id': self.arrivals['arrival_id'],
                'event_id': self.arrivals['event_id'],
                'network': self.arrivals['network'],
                'station': self.arrivals['station'],
                'phase': self.arrivals['phase'],
                f'iter_{self.iiter}': self.arrivals['residual']
            })

            self.event_history = pd.DataFrame({
                'event_id': self.events['event_id'],
                f'iter_{self.iiter}': self.events['residual']
            })

            # Register the available phase types also just in case.
            # Note this isn't necessarily the same as phase_order in the cfg!
            phases = self.arrivals["phase"]
            phases = phases.unique()
            self.phases = sorted(phases)

        self.synchronize(attrs=["events", "arrivals", "event_history", "arrival_history", "phases"])

        return True


    @_utilities.log_errors(logger)
    def load_network_geometry(self):
        """
        Parse and return station data (or "network geometry") from file.

        ROOT_RANK parses file and broadcasts contents to all other processes.
        """
        if RANK == ROOT_RANK:
            logger.info("Loading station data")

            stations = _dataio.parse_network_geometry(self.cfg)
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
        logger.debug(f"Purging raypath directory: {self.raypath_dir}")

        shutil.rmtree(self.raypath_dir)
        os.makedirs(self.raypath_dir)
        return True


    @_utilities.log_errors(logger)
    def relocate_events(self, weightsonly=False):
        """
        Umbrella function to relocate via DE (or some other future method?)
        Also updates arrival residuals/weights, and event weights
        """
        if not weightsonly:
            self._relocate_events_de()

        # After relocating events, update arrival residuals also
        self.update_arrival_residuals()

        # Then update arrival and event KDE weights
        phase_order = self.cfg["algorithm"]["phase_order"]
        for phase in phase_order:
            self.update_arrival_weights(phase) #this re-computes delta, but we shouldn't need to do that

        self.update_event_weights()

        return True


    @_utilities.log_errors(logger)
    def _relocate_events_de(self):
        """
        Relocate all events via Differential Evolution (L1)
        or Equal Differential Time (EDT) and update the "events" attribute
        """

        if RANK == ROOT_RANK:
            ids = self.events["event_id"]
            method = self.cfg["relocation"]["method"]
            logger.info("Relocating %d events via PyKonal-%s" % (len(ids),method.upper()) )
            self._dispatch(sorted(ids))

            logger.debug("Dispatch complete. Gathering events.")
            # Gather and concatenate events from all workers.
            events = COMM.gather(None, root=ROOT_RANK)
            events = pd.concat(events, ignore_index=True)

            self.events = events

            if len(self.events) == 0:
                logger.error("Events DataFrame is empty! (did you over-filter?)")
                return False

        else:
            # Define columns to output
            columns = ["latitude","longitude","depth","time","residual","event_id","source_id"]

            # Initialize EQLocator object
            _path = self.traveltime_inventory_path
            _station_dict = _utilities.station_dict(self.stations)

            with pykonal.locate.EQLocator(_path) as locator:

                cfg         = self.cfg["relocation"]
                method      = cfg["method"]
                depth_min   = cfg["depth_min"]
                dlat        = cfg["dlat"]
                dlon        = cfg["dlon"]
                dz          = cfg["ddepth"]
                dt          = cfg["dtime"]
                tt_error    = cfg["tt_error"]
                pick_uncert = cfg["pick_uncert"]
                truepick_error = cfg["truepick_error"]

                # register station coordinates -> azimuthal gap etc. via
                # locator.quality() become available; default pick error
                # feeds the EDT pair variances
                locator.add_stations(_station_dict)
                locator.default_pick_error = pick_uncert

                # Convert configuration-file parameters from geographic to spherical coordinates
                rho_max = _constants.EARTH_RADIUS - depth_min
                dtheta = np.radians(dlat)
                dphi = np.radians(dlon) * np.cos(np.radians(self._model_lat_center)) # better scaled
                delta = np.array([dz, dtheta, dphi, dt],dtype=_constants.DTYPE_REAL)
                # slightly nonzero dlat and dlon for the quasi teleseisms.. within error anyway
                #   if zero, pykonal can have a tantrum,
                #     if too small then ||G|| zero. 1e-4 seems to work OK
                delta_tele = np.array([.1,.0001,.0001,dt],dtype=_constants.DTYPE_REAL)

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

                    # Extract the initial event location and convert to spherical coordinates
                    _columns = ["latitude", "longitude", "depth", "time"]
                    initial = events.loc[event_id, _columns].values.astype(_constants.DTYPE_REAL)

                    # Grab the source id (str) here also
                    source_id = events.loc[event_id,"source_id"]

                    initial[:3] = geo2sph(initial[:3])

                    locator.clear_arrivals()

                    # Update EQLocator with arrivals for this event
                    _arrivals = _utilities.arrival_dict(self.arrivals, event_id)
                    locator.add_arrivals(_arrivals)

                    # Per-pick uncertainties: truepicks get truepick_error
                    # (near-zero -> they dominate the EDT likelihood)
                    _pick_errors = _utilities.pick_error_dict(
                        self.arrivals, event_id, pick_uncert, truepick_error
                    )
                    locator.add_pick_errors(_pick_errors)

                    # Relocate. EDT needs >= 2 arrivals (pairwise
                    # differential times); single-arrival "teleseisms"
                    # fall back to the L1 objective.
                    try:
                        if len(_arrivals) == 1:
                            # Only let the teleseisms shift via time dimension since 1D (seems to be sensitive to the value. 1e-4 works)
                            loc = locator.locate(initial, delta_tele, alpha=tt_error, method="l1")
                            logger.debug("locating TELESEISM")
                        else:
                            loc = locator.locate(initial, delta, alpha=tt_error, method=method)

                    except Exception as e:
                        logger.warning(f"Location failed for event {event_id}: {str(e)}")
                        raise

                    # Cap at surface AND adjust OT? hmm..
                    if 1 == 2:
                        mdl = self.pwave_model # may be an issue if running S-only TODO
                        rho_top = float(mdl.max_coords[0])          # surface
                        rho_bot = float(mdl.min_coords[0])          # model bottom
                        eps = 1e-3 * float(mdl.node_intervals[0])   # ~metres
                        rho_clamped = min(max(loc[0], rho_bot + eps), rho_top - eps)
                        if rho_clamped != loc[0]:
                            loc[0] = rho_clamped
                            try:
                                t0 = locator.origin_time(loc[:3])   # re-derive origin time at pinned depth
                                if t0 == t0:
                                    loc[3] = t0
                            except Exception:
                                pass
                    else: # or just cap depth to surface and move on as before (didn't seem to be a problem)
                        loc[0] = min(loc[0], rho_max)
                    
                    # Get residual RMS, reformat, append to relocated_events dataframe
                    rms = locator.rms(loc)
                    loc[:3] = sph2geo(loc[:3])

                    event = pd.DataFrame({
                        "latitude": [loc[0]],
                        "longitude": [loc[1]],
                        "depth": [loc[2]],
                        "time": [loc[3]],
                        "residual": [rms],
                        "event_id": [event_id],
                        "source_id": [source_id]
                    })

                    relocated_events = pd.concat([relocated_events, event], ignore_index=True)

        self.synchronize(attrs=["events"])

        return True


    @_utilities.log_errors(logger)
    def sanitize_data(self, for_res_test=False):
        """
        Clean up stations, events, and arrivals. Also adds necessary extra keys etc.
        """

        if RANK == ROOT_RANK:
            logger.info("Sanitizing data")

            # Drop events where residual is NaN
            n0 = len(self.events)
            self.events = self.events.dropna(subset='residual')
            dn = n0 - len(self.events)
            if dn > 0:
                logger.info(f"Dropped {dn} NaNs from events (tends to happen if contains an arrival with NaN residual)   ###")

            # Drop events where weight is 0 (probably shouldn't be happening here)
            if 'weight' in self.events.columns:
                n0 = len(self.events)
                self.events = self.events[self.events['weight'] > 0]
                dn = n0 - len(self.events)
                if dn > 0:
                    logger.info(f"Dropped {dn} events with zero weights. {n0-dn} remain.")

            # Drop arrivals where residual is NaN
            n0 = len(self.arrivals)
            self.arrivals = self.arrivals.dropna(subset='residual')
            dn = n0 - len(self.arrivals)
            if dn > 0:
                logger.info(f"Dropped {dn} NaNs from arrivals (arrivals get NaN residuals if too near model boundary)   ###")

            # Drop arrivals where weight is 0 (probably shouldn't be happening here)
            if 'weight' in self.arrivals.columns:
                n0 = len(self.arrivals)
                self.arrivals = self.arrivals[self.arrivals['weight'] > 0]
                dn = n0 - len(self.arrivals)
                if dn > 0:
                    logger.info(f"Dropped {dn} arrivals with zero weights. {n0-dn} remain.   ###")

            # Drop duplicate arrivals
            keys = ["network", "station", "phase", "event_id"]
            n0 = len(self.arrivals)
            self.arrivals = self.arrivals.drop_duplicates(keys)
            dn = n0 - len(self.arrivals)
            if dn > 0:
                logger.info(f"Dropped {dn} duplicate arrivals {n0-dn} remain.   ###")

            # Drop duplicate stations
            keys = ["network", "station"]
            n0 = len(self.stations)
            self.stations = self.stations.drop_duplicates(keys)
            dn = n0 - len(self.stations)
            if dn > 0:
                logger.info(f"Dropped {dn} duplicate stations {n0-dn} remain.   ###")

            # Drop duplicate events
            keys = ["latitude", "longitude", "depth", "time"]
            n0 = len(self.events)
            self.events = self.events.drop_duplicates(keys)
            dn = n0 - len(self.events)
            if dn > 0:
                logger.info(f"Dropped {dn} duplicate events {n0-dn} remain.   ###")

            # Drop events outside of the velocity model
            velmodel = self.pwave_model
            minlat,maxlon,mindepth = sph2geo(velmodel.max_coords) # a little confusing but this is correct
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
                logger.info(f"Dropped {dn} events outside of velocity model. {n0-dn} remain.   ###")

            # Drop stations outside of velocity model
            stations = self.stations
            n0 = len(self.stations)
            idx_keep = stations[ (minlon <= stations['longitude'])
                             & (stations['longitude']<= maxlon)
                             & (minlat <= stations['latitude'])
                             & (stations['latitude']<= maxlat)].index
            self.stations = stations.loc[idx_keep]
            dn = n0 - len(self.stations)
            if dn > 0:
                logger.info(f"Dropped {dn} stations outside of velocity model. {n0-dn} remain.   ###")

            # Drop stations outside of map_filter (if it exists)
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
                    logger.info(f"Dropped {dn} stations outside of map_filter. {n0-dn} remain.   ###")

            # Drop arrivals linked to those dropped stations
            n0 = len(self.arrivals)
            stations_set = set(zip(self.stations['network'], self.stations['station']))
            arrival_mask = self.arrivals.apply(lambda x: (x['network'], x['station']) in stations_set, axis=1)
            self.arrivals = self.arrivals[arrival_mask]
            dn = n0 - len(self.arrivals)
            if dn > 0:
                logger.info(f"Dropped {dn} arrivals with stations outside velocity model. {n0-dn} remain.   ###")

            if not for_res_test:
                # Drop events without minimum number of arrivals
                min_narrival = self.cfg["algorithm"]["min_narrival"]
                n0 = len(self.events)
                counts = self.arrivals["event_id"].value_counts()
                counts = counts[(counts >= min_narrival) | (counts == 1)] # allow singular teleseisms-as-synthetic-events to remain
                event_ids = counts.index
                self.events = self.events[self.events["event_id"].isin(event_ids)]
                dn = n0 - len(self.events)
                if dn > 0:
                    logger.info(f"Dropped {dn} events with < {min_narrival} arrivals. {n0-dn} remain.   ###")

                # Drop arrivals without events
                n0 = len(self.arrivals)
                bool_idx = self.arrivals["event_id"].isin(self.events["event_id"])
                self.arrivals = self.arrivals[bool_idx]
                dn = n0 - len(self.arrivals)
                if dn > 0:
                    logger.info(f"Dropped {dn} arrivals without associated events. {n0-dn} remain.   ###")

            # Drop events/arrivals out of desired lateral range
            arrivals   = self.arrivals
            max_dist   = self.cfg["algorithm"]["max_dist"] # km, only 2D
            min_dist   = self.cfg["algorithm"]["min_dist"] # this is the 3D distance effectively
            max_depth   = self.cfg["algorithm"]["max_depth"]
            min_depth   = self.cfg["algorithm"]["min_depth"]

            # Merge event data (why is this needed?)
            events = self.events.rename(
                columns={
                    "latitude": "event_latitude",
                    "longitude": "event_longitude",
                    "depth": "event_depth"
                }
            )

            # Merge event data into arrivals
            merge_columns = [
                "event_latitude",
                "event_longitude",
                "event_depth",
                "event_id"
            ]
            arrivals = arrivals.merge(events[merge_columns], on="event_id", how='left')

            # Merge station data
            stations = self.stations.rename(
                columns={
                    "latitude": "station_latitude",
                    "longitude": "station_longitude"
                }
            )

            # Merge station data into arrivals
            merge_columns = [
                "station_latitude",
                "station_longitude",
                "network",
                "station"
            ]
            merge_keys = ["network", "station"]
            arrivals = arrivals.merge(stations[merge_columns], on=merge_keys, how='left')

            # Set arrivals distance (for use later, in sample_arrivals. do this once per iteration.)
            epicentral = np.asarray(
                _utilities.dist_km(arrivals["event_latitude"], arrivals["event_longitude"],
                                   arrivals["station_latitude"], arrivals["station_longitude"]),
                dtype=float)
            depth = np.asarray(arrivals["event_depth"], dtype=float)

            self.arrivals["delta"]      = epicentral
            self.arrivals["slant_dist"] = np.sqrt(epicentral**2 + depth**2)

            # Remove any events outside of epicentral limit
            n0 = len(self.arrivals)
            keep_dist = (
                (self.arrivals["slant_dist"] >= min_dist) &
                (self.arrivals["delta"]      <= max_dist)
            )
            self.arrivals = self.arrivals[keep_dist].reset_index(drop=True)
            dn = n0 - len(self.arrivals)
            if dn > 0:
                logger.info(
                    f"Distance filter: dropped {dn} arrivals outside "
                    f"[min_dist={min_dist} km (slant), max_dist={max_dist} km "
                    f"(epicentral)]. {len(self.arrivals)} remain.   ###"
                )

            # Only select events within desired depth bounds
            idx_keep = self.events[
                ((self.events['depth'] >= min_depth) & (self.events['depth'] <= max_depth))
            ].index

            n0 = len(self.events)
            self.events = self.events.loc[idx_keep]
            dn = n0 - len(self.events)
            if dn > 0:
                logger.info(f"Dropped {dn} events outside of requested depth range. {n0-dn} remain.   ###")


            # Drop events without arrivals
            n0 = len(self.events)
            bool_idx = self.events["event_id"].isin(self.arrivals["event_id"])
            self.events = self.events[bool_idx]
            self.events0 = self.events.copy() # to track as the code progresses (only ROOT)

            dn = n0 - len(self.events)
            if dn > 0:
                logger.info(f"Dropped {dn} events without associated arrivals. {n0-dn} remain.   ###")

            # Drop stations without arrivals.. then replant self.stations
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
                logger.info(f"Dropped {dn} stations without associated arrivals. {n0-dn} remain.   ###")

            # Drop arrivals without stations.. then replant arrivals
            n0 = len(self.arrivals)
            stations = self.stations.set_index(["network", "station"])
            idx_keep = stations.index.unique()
            arrivals = self.arrivals.set_index(["network", "station"])
            arrivals = arrivals.loc[idx_keep]
            arrivals = arrivals.reset_index()
            self.arrivals = arrivals

            dn = n0 - len(self.arrivals)
            if dn > 0:
                logger.info(f"Dropped {dn} arrivals without associated stations. {n0-dn} remain.   ###")

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
        """ RE-Sanitize data as we iterate """

        if RANK == ROOT_RANK:
            logger.info("RE-sanitizing data")

            # Drop events where residual is NaN
            n0 = len(self.events)
            self.events = self.events.dropna(subset='residual')
            dn = n0 - len(self.events)
            if dn > 0:
                logger.info(f"Dropped {dn} NaNs from events (shouldn't be happening...)   ###")

            # Drop events where weight is 0 (usually a bug in the source catalog)
            if 'weight' in self.events.columns:
                n0 = len(self.events)
                self.events = self.events[self.events['weight'] > 0]
                dn = n0 - len(self.events)
                if dn > 0:
                    logger.info(f"Dropped {dn} events with zero weights. {n0-dn} remain. (shouldn't happen!)   ###")

            # Drop arrivals where residual is NaN (!)
            n0 = len(self.arrivals)
            self.arrivals = self.arrivals.dropna(subset='residual')
            dn = n0 - len(self.arrivals)
            if dn > 0:
                logger.info(f"Dropped {dn} NaNs from arrivals (shouldn't be happening...)   ###")

            # Drop arrivals where weight is 0 but only if running normally, otherwise will drop non-used phases
            if ('weight' in self.arrivals.columns 
            and 'P' in self.cfg["algorithm"]["phase_order"] 
            and 'S' in self.cfg["algorithm"]["phase_order"]):
                n0 = len(self.arrivals)
                self.arrivals = self.arrivals[self.arrivals['weight'] > 0]
                dn = n0 - len(self.arrivals)
                if dn > 0:
                    logger.info(f"Dropped {dn} arrivals with zero weights. {n0-dn} remain. (shouldn't happen!)   ###")

            # Drop events or arrivals with bad residuals (note that by doing so, events may then fall below the min arrivals filter)
            if do_remove_outliers:
                max_evt_resid = self.cfg["algorithm"]["max_event_residual"]
                max_arr_resid = self.cfg["algorithm"]["max_arrival_residual"]
                n0 = len(self.arrivals)
                self.arrivals = _utilities.remove_outliers(self.arrivals,None,"residual", max_arr_resid)
                if len(self.arrivals) < n0:
                    dn = n0 - len(self.arrivals)
                    logger.info(f"Dropped {dn} arrivals with residual > {max_arr_resid}. {n0-dn} remain.   ###")
                n0 = len(self.events)
                self.events = _utilities.remove_outliers(self.events,None,"residual",max_evt_resid)
                if len(self.events) < n0:
                    dn = n0 - len(self.events)
                    logger.info(f"Dropped {dn} events with residual > {max_evt_resid}. {n0-dn} remain.   ###")

                n0 = len(self.arrivals)
                bool_idx = self.arrivals["event_id"].isin(self.events["event_id"])
                self.arrivals = self.arrivals[bool_idx]
                dn = n0 - len(self.arrivals)
                if dn > 0:
                    logger.info(f"Dropped {dn} arrivals linked to outlier-removed events. {n0-dn} remain.   ###")

            # Drop events without minimum number of arrivals
            # Important to do this here after track_residual_improvement!
            min_narrival = self.cfg["algorithm"]["min_narrival"]
            n0 = len(self.events)
            counts = self.arrivals["event_id"].value_counts()
            counts = counts[(counts >= min_narrival) | (counts == 1)] # also let singular (e.g. teleseisms) pass
            event_ids = counts.index
            self.events = self.events[self.events["event_id"].isin(event_ids)]
            dn = n0 - len(self.events)
            if dn > 0:
                logger.info(f"Dropped {dn} events with < {min_narrival} arrivals. {n0-dn} remain.   ###")

            # Drop arrivals without events
            n0 = len(self.arrivals)
            bool_idx = self.arrivals["event_id"].isin(self.events["event_id"])
            self.arrivals = self.arrivals[bool_idx]
            dn = n0 - len(self.arrivals)
            if dn > 0:
                logger.info(f"Dropped {dn} arrivals without associated events. {n0-dn} remain.   ###")

            # Drop events without arrivals 
            n0 = len(self.events)
            bool_idx = self.events["event_id"].isin(self.arrivals["event_id"])
            self.events = self.events[bool_idx]
            dn = n0 - len(self.events)
            if dn > 0:
                logger.info(f"Dropped {dn} events without associated arrivals. {n0-dn} remain.   ###")

            if len(self.stations) == 0:
                logger.error("All stations were dropped!!") # we aren't really testing stations for RE-sanitize but maybe we should be
            if len(self.events) == 0:
                logger.error("All events were dropped!!")
            if len(self.arrivals) == 0:
                logger.error("All arrivals were dropped!!")


            # We have to do this here now also if we're going to track slant_dist
            events = self.events.rename(columns={
                "latitude": "event_latitude",
                "longitude": "event_longitude",
                "depth": "event_depth",
            })
            a = self.arrivals.merge(
                events[["event_latitude", "event_longitude", "event_depth", "event_id"]],
                on="event_id", how="left",
            )
            stations = self.stations.rename(columns={
                "latitude": "station_latitude",
                "longitude": "station_longitude",
            })
            a = a.merge(
                stations[["station_latitude", "station_longitude", "network", "station"]],
                on=["network", "station"], how="left",
            )

            epicentral = np.asarray(
                _utilities.dist_km(a["event_latitude"], a["event_longitude"],
                                   a["station_latitude"], a["station_longitude"]),
                dtype=float)
            depth = np.asarray(a["event_depth"], dtype=float)

            self.arrivals["delta"] = epicentral
            self.arrivals["slant_dist"] = np.sqrt(epicentral**2 + depth**2)

            # And re-drop events (recently migrated?) outside of the epicentral/slant dist
            _max_dist = self.cfg["algorithm"]["max_dist"]
            _min_dist = self.cfg["algorithm"]["min_dist"]
            n0 = len(self.arrivals)
            self.arrivals = self.arrivals[
                (self.arrivals["slant_dist"] >= _min_dist) &
                (self.arrivals["delta"]      <= _max_dist)
            ].reset_index(drop=True)
            dn = n0 - len(self.arrivals)
            if dn > 0:
                logger.info(
                    f"Distance filter: dropped {dn} arrivals outside "
                    f"[min_dist={_min_dist} km (slant), max_dist={_max_dist} km "
                    f"(epicentral)]. {len(self.arrivals)} remain.   ###"
                )


        self.synchronize(attrs=["stations","events","arrivals"])

        return True


    @_utilities.log_errors(logger)
    @_utilities.root_only(RANK)
    def save_events(self):
        """ Save the current "events", and "arrivals" to and HDF5 file using pandas.HDFStore """
        logger.info(f"Saving event data from iteration #{self.iiter}")

        path = os.path.join(self.cfg['model']['output_dir'], f"{self.iiter:02d}")

        events       = self.events
        EVENT_DTYPES = _constants.EVENT_DTYPES
        for column in EVENT_DTYPES:
            events[column] = events[column].astype(EVENT_DTYPES[column])

        arrivals       = self.arrivals
        ARRIVAL_DTYPES = _constants.ARRIVAL_DTYPES
        for column in ARRIVAL_DTYPES:
            arrivals[column] = arrivals[column].astype(ARRIVAL_DTYPES[column])

        events.to_hdf(f"{path}.events.h5", key="events", complevel=5, complib="zlib")
        arrivals.to_hdf(f"{path}.events.h5", key="arrivals", complevel=5, complib="zlib")

        return True


    @_utilities.log_errors(logger)
    @_utilities.root_only(RANK)
    def save_stations(self):
        """
        Save the current stations to and HDF5 file using
        pandas.HDFStore. Usually just the first!
        """
        logger.info(f"Saving filtered station data from iteration #{self.iiter}")

        path = os.path.join(self.cfg['model']['output_dir'], f"{self.iiter:02d}")

        self.stations.to_hdf(f"{path}.stations.h5", key="stations", complevel=5, complib="zlib")

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
        path = os.path.join(self.cfg['model']['output_dir'], f"{self.iiter:02d}")

        # Save velocity
        handle = f"{phase}wave_model"
        label = f"{handle}.{tag}" if tag is not None else handle
        model = getattr(self, handle)
        model.to_hdf(path + f".{label}.h5")

        if self.iiter == 0:
            return True

        # Save variance
        handle = f"{phase}wave_variance"
        model = getattr(self, handle)
        label = f"{handle}.{tag}" if tag is not None else handle
        model.to_hdf(path + f".{label}.h5")

        # Iteration "quality" (work in progress / not in use)
        #handle = f"{phase}wave_quality"
        #model = getattr(self, handle)
        #label = f"{handle}.{tag}" if tag is not None else handle
        #model.to_hdf(path + f".{label}.h5")

        if self.argc.output_realizations is True:
            handle = f"{phase}wave_realization_stack"
            label = f"{handle}.{tag}" if tag is not None else handle
            stack = getattr(self, handle)
            with h5py.File(path + f".{label}.h5", mode="w") as f5:
                f5.create_dataset(f"{phase}wave_stack",data=stack[:])

            # Also do the quality stack (new! not working? stupid?)
            #handle = f"{phase}qual_realization_stack"
            #label = f"{handle}.{tag}" if tag is not None else handle
            #stack = getattr(self, handle)
            #with h5py.File(path + f".{label}.h5", mode="w") as f5:
            #    f5.create_dataset(f"{phase}qual_stack",data=stack[:])

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
            "gradient_magnitude",
            "grid_coords",
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

        if RANK == ROOT_RANK:
            n0 = len(self.arrivals)
            bool_idx = self.arrivals["event_id"].isin(self.events["event_id"])
            self.arrivals = self.arrivals[bool_idx]
            dn = n0 - len(self.arrivals)
            if dn > 0:
                logger.warning(f"Dropped {dn} orphaned arrivals in "
                    f"update_arrival_residuals. {n0-dn} remain.   ###")

        self.synchronize(attrs=["arrivals"])

        arrivals = self.arrivals.set_index(["network", "station", "phase"])
        logger.info("Updating %d arrival residuals" % len(arrivals))
        arrivals = arrivals.sort_index()

        if RANK == ROOT_RANK:
            ids = arrivals.index.unique()
            self._dispatch(ids)
            logger.debug("Dispatch complete. Gathering arrivals.")
            arrivals = COMM.gather(None, root=ROOT_RANK)
            arrivals = pd.concat(arrivals, ignore_index=True)

            # Sometimes NaNs sneak in as residuals,
            #   usually if the source or station is very nearby
            #   or eclipses model boundary?
            nan_mask = arrivals['residual'].isna()
            if nan_mask.any():
                bad = arrivals[nan_mask]
                logger.info(
                    f"Dropped {nan_mask.sum()} arrivals with NaN residuals. "
                    f"event_ids: {bad['event_id'].unique()}, "
                    f"arrival_ids: {bad['arrival_id'].unique()}   ###"
                )
            arrivals = arrivals.dropna(subset=['residual'])

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

                    if phase.upper() in run_phases:

                        handle = "/".join([network, station, phase])
                        if handle != last_handle:

                            traveltime = traveltime_inventory.read(handle)
                            last_handle = handle

                        _arrivals = arrivals.loc[(network, station, phase)]
                        _events = events.loc[_arrivals["event_id"].values]
                        arrival_times = _arrivals["time"].values
                        delta = _arrivals["delta"].values
                        origin_times = _events["time"].values
                        coords = _events[["latitude", "longitude", "depth"]].values
                        coords = geo2sph(coords)
                        residuals = arrival_times - (origin_times + traveltime.resample(coords))
                        truepick = (
                            _arrivals["truepick"].values
                            if "truepick" in _arrivals.columns
                            else np.zeros(len(_arrivals), dtype=bool)
                        )
                        _arrivals = dict(
                            network=network,
                            station=station,
                            phase=phase,
                            event_id=_arrivals["event_id"].values,
                            arrival_id=_arrivals["arrival_id"].values,
                            time=arrival_times,
                            residual=residuals,
                            delta=delta,
                            slant_dist=_arrivals["slant_dist"].values,
                            truepick=truepick
                        )
                        _arrivals = pd.DataFrame(_arrivals)
                        updated_arrivals = pd.concat([updated_arrivals, _arrivals])

        self.synchronize(attrs=["arrivals"])

        return True


    @_utilities.log_errors(logger)
    def update_model(self, phase, compute_1d=False):
        """
        Perform stack statistics to update our model
        """
        logger.info(f"Running update_model for {phase}   ###")
        phase = phase.lower()

        if RANK == ROOT_RANK:
            # Get slowness and quality stacks
            stack = getattr(self, f"{phase}wave_realization_stack")
            quality_stack = getattr(self, f"{phase}qual_realization_stack") # not really used (yet? ever?)

            # Restrict to realizations actually computed (rows are NaN-
            # filled at allocation; with adaptive early stopping the stack
            # may hold fewer than algorithm.nreal). Masking also protects
            # per-node NaNs within computed realizations.
            raw = stack[...]
            filled = ~np.all(
                np.isnan(raw.reshape(raw.shape[0], -1)), axis=1
            )
            n_filled = int(filled.sum())
            if n_filled < raw.shape[0]:
                logger.info(
                    f"update_model: stacking {n_filled} of "
                    f"{raw.shape[0]} allocated realizations   ###"
                )
            stack = np.ma.masked_invalid(raw[filled])

            # Get our values. median is better than mean here, and allows for sharper features to be resolved
            variance = np.ma.var(stack,axis=0)

            # Deriving the dominant stack value is a bit tricky.
            # Robust way is median, but we lose amplitude variation unless very small mesh.
            # A mean is great, but suffers if data is poor (fat, skewed, or multimodal distribution).
            # Over time a "clipped mean" seems to perform best for both synthetic and real data,
            #  but let users experiment themselves
            # Original v1 code was a pure median

            trim_fraction = self.cfg["algorithm"].get("stack_trim_percent", 0)
            trim_fraction = np.clip(float(trim_fraction/100),0,0.485) # cap at 48.5%!

            if 1 == 1: # tried & tested mean/median (shared with the
                # adaptive-stopping convergence metric via _stack_statistic)
                delta_slowness = self._stack_statistic(stack)
            else:
                # exotic MODAL method.. not great tbh
                #keep_fraction = 1 - 2*trim_fraction
                delta_slowness = _utilities.stack_modal(stack,trim=trim_fraction)

            # Grab the model we are updating (which should be in velocity)
            model = getattr(self, f"{phase}wave_model")

            # Hold onto the original copy to restore certain very low velocity (oceans mostly?) areas
            orig_model = model.values.copy()
            watermask = model.values <= 0.2 # assume 0.2 km/s is water / TODO parameterize?
            wateridx = np.where(watermask)

            if compute_1d:
                ref_slowness_1d = np.nanmean(1.0 / model.values, axis=(1, 2))  # (nz,)

            # Implement a bit of impedence in the early steps
            # so this limits model updates to step_frac of what they would have been
            if self.iiter <= 1:
                step_frac = 0.80
            elif self.iiter <= 2 and self.niter > 2:
                step_frac = 0.90
            elif self.iiter <= 3 and self.niter > 3:
                step_frac = 0.95
            else:
                step_frac = 1

            # Update model in slowness, then convert back to velocity. limit change to step_frac
            values = np.power(model.values, -1) + delta_slowness * step_frac
            velocities = np.power(values, -1)
            model.values = velocities
            # Restore water velocity
            model.values[watermask] = orig_model[wateridx]


            # Update variance also (work back to get this in variance in VELOCITY)
            model = getattr(self, f"{phase}wave_variance")
            #model.values = variance # n.b. this is variance of SLOWNESS

            ## But what if we want to convert variance from slowness to velocity?
            # Var(1/s) ~= Var(s) / s^4 = Var(s) * v^4
            slowness_values = np.power(velocities, -1)
            velocity_variance = variance * np.power(velocities, 4)

            # Store variance as velocity variance (in (km/s)^2 -- take SQRT when interpreting!)
            model.values = velocity_variance

            # Keep track of mean? median? so we can monitor throughout the iterations
            self._max_variance_km_s = np.mean( np.sqrt(velocity_variance) )
            logger.info(f"Mean {phase.upper()} velocity variance (km/s): {self._max_variance_km_s:0.6f}   ###")

            # Also calculate a 1D version using same method as 3D? It is fast, and fun as well,
            if compute_1d:
                stack_1d   = getattr(self, f"_{phase}wave_1d_stack")
                stack_1d_ma = np.ma.masked_invalid(stack_1d)
                stack_type = self.cfg["algorithm"].get("stack_type") 

                # Same trim as 3D stack
                if trim_fraction > 0:
                    nreal_1d = stack_1d_ma.shape[0]
                    n_trim   = int(nreal_1d * trim_fraction)
                    if n_trim > 0 and nreal_1d > 2 * n_trim:
                        sorted_1d  = np.ma.sort(stack_1d_ma, axis=0)
                        trimmed_1d = sorted_1d[n_trim:-n_trim, :]
                    else:
                        trimmed_1d = stack_1d_ma
                else:
                    trimmed_1d = stack_1d_ma

                if stack_type.lower() == "median":
                    delta_slowness_1d = np.ma.median(trimmed_1d, axis=0).filled(np.nan)
                else:
                    delta_slowness_1d = np.ma.mean(trimmed_1d, axis=0).filled(np.nan)

                delta_slowness_std = np.ma.std(trimmed_1d, axis=0).filled(np.nan)
                #n_contributing     = (~np.ma.getmaskarray(trimmed_1d)).sum(axis=0) # this is always the same/not too useful (TODO)

                # Convert delta-slowness to absolute velocity using the
                # updated 3D model laterally averaged at each depth node.
                # model.values is (nz, ntheta, nphi) — mean over lateral dims.
                abs_slowness_1d = ref_slowness_1d + delta_slowness_1d
                velocity_1d     = np.where(abs_slowness_1d > 0, 1.0 / abs_slowness_1d, np.nan)

                # Uncertainty: propagate slowness std to velocity std
                # Var(1/s) ≈ Var(s) * v^4  =>  std(v) ≈ std(s) * v^2
                velocity_std_1d = delta_slowness_std * np.where(
                    np.isfinite(velocity_1d), velocity_1d ** 2, np.nan
                )

                # Depth axis: same convention as 3D grid
                nz        = model.npts[0]
                rho_nodes = model.min_coords[0] + np.arange(nz) * model.node_intervals[0]
                depth_km  = _constants.EARTH_RADIUS - rho_nodes  # positive downward

                df_1d = pd.DataFrame({
                    "depth_km":          np.around(depth_km,3),
                    "velocity_km_s":     np.around(velocity_1d,3),
                    "velocity_std_km_s": np.around(velocity_std_1d,5),
                    #"n_realizations":    n_contributing,
                })
                df_1d = df_1d.sort_values("depth_km").reset_index(drop=True) # shallow first

                path_1d = os.path.join(
                    self.cfg["model"]["output_dir"],
                    f"{self.iiter:02d}.{phase}wave_model.1d.csv"
                )
                df_1d.to_csv(path_1d, index=False)
                logger.info(f"Saved 1D {phase.upper()}-wave model to {path_1d}   ###")

                setattr(self, f"_{phase}wave_1d_stack", None) # free up memory although should be trivial

        self.synchronize(attrs=[f"{phase}wave_model"])
        return True


    @_utilities.log_errors(logger)
    def run_resolution_test(self):
        """Execute resolution test"""

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

                # Not really needed if ONLY running resolution test but code complains elsewhere
                if 'arrival_id' not in self.arrivals.keys():
                    self.arrivals['arrival_id'] = range(len(self.arrivals))

                if 'source_id' not in self.events.keys():
                    self.events['source_id'] = "event_" + self.events['event_id'].astype(str).str.zfill(6)

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
                    model.min_coords = loaded_model.min_coords.astype(np.float64)
                    model.node_intervals = loaded_model.node_intervals
                    model.npts = loaded_model.npts
                    model.values = loaded_model.values.astype(np.float64)

                logger.info(f"Resolution test starting with {len(self.events)} events and {len(self.arrivals)} arrivals   ###")

            else:
                logger.info("Using current inversion state for resolution test   ###")

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

        # This is required as it preps station data. If we start saving station data, maybe could avoid
        if need_to_load_data:
            self.sanitize_data(for_res_test=True)

        # Update events & arrivals (adds KDE weight)
        self.update_event_weights()
        self.synchronize(attrs=["events"])

        # Run process
        for phase in self.cfg["algorithm"]["phase_order"]:
            self._run_resolution_test_single_phase(phase, horiz_block_size_km, amplitude)
        return True

    @_utilities.log_errors(logger)
    def _run_resolution_test_single_phase(self, phase, horiz_block_size_km, amplitude):
        """Run resolution test per phase"""

        if RANK == ROOT_RANK:
            logger.info(f"Running checkerboard test for {phase}   ###")

            # Store original state
            original_arrivals = self.arrivals.copy()
            base_model = self.pwave_model if phase == 'P' else self.swave_model
            original_model = _restesting._copy_scalar_field(base_model)  # Make a deep copy

            # Create synthetic model and arrivals for both phases, regardless
            synthetic_model = _restesting._create_checkerboard_model(base_model,
                                                                     horiz_block_size_km,
                                                                     vertical_layers=self.cfg["model"]["res_test_layers"],
                                                                     amplitude=amplitude)
            logger.debug(f"Synthetic {phase} model created with shape: {synthetic_model.values.shape}   ###")

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

        # Generate traveltimes and residuals on the synth data
        self.compute_traveltime_lookup_tables(run_phases=[phase])
        self.update_arrival_residuals(run_phases=[phase])
        self.update_arrival_weights(phase)

        # Flip residual polarity, restore original model
        if RANK == ROOT_RANK:
            # since times_checkerboard - times_original ~= - residual,
            # just flip it back to so the output model matches the input 
            self.arrivals['residual'] = -self.arrivals['residual']

            if phase == 'P':
                self.pwave_model = original_model
            else:
                self.swave_model = original_model

        # Re-sync to original model... also sync the arrivals again
        self.synchronize(attrs=[f"{phase.lower()}wave_model", "arrivals"])

        # Run full inversion using same parameters as real inversion
        nreal = self.cfg["algorithm"]["nreal"]

        hvr = self.cfg["meshing"]["hvr"]
        min_rays_per_cell = self.cfg["meshing"]["min_rays_per_cell"]
        adaptive_weight = self.cfg["meshing"]["adaptive_data_weight"]
        adaptive_weight = min(adaptive_weight,1.0)
        density_to_gradient_weight = self.cfg["meshing"]["density_to_gradient_weight"]
        density_to_gradient_weight = max(0,min(density_to_gradient_weight,1.0))        

        # Reset stack, define vel gradients, and run multiple realizations
        self._reset_realization_stack(phase)
        self._estimate_velocity_gradient_density(phase)

        for self.ireal in range(nreal):
            logger.info(f"{phase} RESOLUTION TEST realization {self.ireal+1}/{nreal}")

            # Use same sampling and stochastic variations as real inversion but don't QC the residuals-- they will be large!
            self._sample_events(do_remove_outliers=False)
            self._sample_arrivals(phase,do_remove_outliers=False)
            self._trace_rays(phase)

            self._generate_voronoi_cells(phase)
            self._compute_sensitivity_matrix(phase,hvr)
            self._update_projection_matrix(phase,hvr)
            self._compute_model_update(phase,min_rays=min_rays_per_cell,use_weights=False) # don't apply weights for the resolution test

        # Process stack using same method as real inversion
        self.update_model(phase)

        if RANK == ROOT_RANK:

            # Now extract the final averaged model (not raw stack)
            recovered_model = _restesting._copy_scalar_field(self.pwave_model if phase == 'P' else self.swave_model)

            # Analyze using original base model as reference
            metrics = _restesting._analyze_resolution(self, synthetic_model, recovered_model, phase, ref_model=original_model)

            # Save
            _restesting._save_results(
                self.cfg["model"]["output_dir"], synthetic_model, recovered_model,
                metrics, phase, horiz_block_size_km
            )

            # Restore original arrivals
            self.arrivals = original_arrivals

        self.synchronize(attrs=["arrivals"])
        self.purge_raypaths()


# # # end of InversionIterator
