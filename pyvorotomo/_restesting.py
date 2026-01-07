import os
import json
import glob
import pykonal
import numpy as np
import pandas as pd

from . import _constants
from . import _picklabel
from . import _utilities

# Get logger handle.
logger = _utilities.get_logger(f"__main__.{__name__}")


from scipy.ndimage import convolve1d
def _create_checkerboard_model(base_model, horiz_block_size_km,
    vertical_layers = [10,25,50,80,150], amplitude=0.08):
    """
    Create smooth checkerboard using cosine smoothing in horizontal,
    Hann windows (length 3*dz) in vertical. Flips vertical polarity
    at user defined levels (force users to decide!)
    """

    # Hardwire vertical smoothing
    dz = base_model.node_intervals[0] 
    vertical_smooth_km = 3 * dz  # turns OFF at 1 * dz

    logger.info(f"Creating checkerboard with {horiz_block_size_km}km horizontal,"
     f" {vertical_layers}km vertical blocks, {vertical_smooth_km:.1f}km vertical smoothing, {amplitude} amplitude   ###")

    min_coords = base_model.min_coords
    max_coords = base_model.max_coords
    checkerboard_model = _copy_scalar_field(base_model)
    checkerboard_values = base_model.values.copy()

    # Convert to wavelengths (full cycle = 2 blocks)
    angular_wavelength = 2 * horiz_block_size_km / _constants.EARTH_RADIUS
    nz, ny, nx = base_model.npts
    min_rho = min_coords[0]
    max_rho = min_coords[0] + (nz - 1) * base_model.node_intervals[0]
    max_depth = _constants.EARTH_RADIUS - min_rho
    min_depth = _constants.EARTH_RADIUS - max_rho

    sorted_layers = sorted(vertical_layers)

    # Create the vertical sign pattern (1D array)
    depth_array = _constants.EARTH_RADIUS - (min_coords[0] + np.arange(nz) * base_model.node_intervals[0])
    vert_sign_array = np.ones(nz)
    for i, depth in enumerate(depth_array):
        n_layers_crossed = sum(1 for layer_depth in sorted_layers if depth > layer_depth)
        vert_sign_array[i] = (-1) ** n_layers_crossed

    # Apply vertical smoothing using convolution
    n_smooth_cells = int(vertical_smooth_km / dz)
    if n_smooth_cells > 1:
        # Create smoothing kernel (Hann window)
        kernel_size = 2 * n_smooth_cells + 1
        kernel = np.hanning(kernel_size)
        kernel = kernel / kernel.sum()

        # & Smoooooth
        vert_sign_array = convolve1d(vert_sign_array, kernel, mode='nearest')

    # Now build full CB model
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                theta = min_coords[1] + iy * base_model.node_intervals[1]
                phi = min_coords[2] + ix * base_model.node_intervals[2]

                # Smooth horizontal components
                theta_component = np.cos(2 * np.pi * (theta - min_coords[1]) / angular_wavelength)
                phi_component = np.cos(2 * np.pi * (phi - min_coords[2]) / angular_wavelength)

                # Use pre-smoothed vertical component
                vert_component = vert_sign_array[iz]

                perturbation = vert_component * theta_component * phi_component
                checkerboard_values[iz, iy, ix] *= (1.0 + amplitude * perturbation)

    checkerboard_model.values = checkerboard_values
    return checkerboard_model


def _extract_recovered_model(iterator, phase):
    """ Extract recovered model from inversion results """
    
    if phase == 'P':
        final_model = iterator.pwave_model
    else:
        final_model = iterator.swave_model
    
    # Use the final processed model (already averaged by update_model())
    recovered_model = _copy_scalar_field(final_model)
    
    # Create Voronoi coverage mask to identify well-constrained areas
    voronoi_mask = _create_voronoi_coverage_mask(iterator, final_model, phase)
    
    # Apply masking - set poorly constrained areas to NaN
    masked_values = recovered_model.values.copy()
    masked_values[~voronoi_mask] = np.nan
    recovered_model.values = masked_values
    
    # Log diagnostics
    n_valid = np.sum(voronoi_mask)
    n_total = np.prod(final_model.values.shape)
    
    logger.info(f"Voronoi mask covers {n_valid}/{n_total} nodes ({n_valid/n_total*100:.1f}%)   ###")
    
    if n_valid > 0:
        valid_values = masked_values[voronoi_mask]
        logger.info(f"Masked velocity range: {valid_values.min():.2f} to {valid_values.max():.2f}")
    else:
        logger.warning("No valid nodes after Voronoi masking")
    
    logger.info(f"Final model velocity range: {final_model.values.min():.2f} to {final_model.values.max():.2f}   ###")
    
    return recovered_model

# this needs work
def _analyze_resolution(iterator, input_model, recovered_model, phase, ref_model):
    """Analyze checkerboard resolution test results with more appropriate metrics."""

    # Calculate perturbations
    input_pert = (input_model.values / ref_model.values) - 1.0
    recovered_pert = (recovered_model.values / ref_model.values) - 1.0

    # get the ray coverage mask - cells that actually have rays
    coverage_mask = _create_voronoi_coverage_mask(iterator, input_model, phase)
    # NaNs too
    nan_mask = ~(np.isnan(recovered_pert) | np.isnan(input_pert))
    coverage_mask = coverage_mask & nan_mask

    # exclude extreme boundaries
    shape = input_pert.shape
    margin = 1  # Minimal margin since coverage mask is primary
    boundary_mask = np.ones(shape, dtype=bool)
    boundary_mask[:margin, :, :] = False
    boundary_mask[-margin:, :, :] = False
    boundary_mask[:, :margin, :] = False
    boundary_mask[:, -margin:, :] = False
    boundary_mask[:, :, :margin] = False
    boundary_mask[:, :, -margin:] = False

    # Also require some minimum absolute perturbation
    signal_threshold = 0.01
    significant_input = np.abs(input_pert) > signal_threshold

    # Combine all masks - most importantly the coverage mask
    analysis_mask = coverage_mask & boundary_mask & significant_input

    # Also track cells that should have signal but got masked by coverage
    no_coverage_but_perturbed = significant_input & (~coverage_mask)
    n_no_coverage = np.sum(no_coverage_but_perturbed)
    if n_no_coverage > 0:
        logger.warning(f"{n_no_coverage} cells have checkerboard perturbations but no ray coverage")

    masked_input = input_pert[analysis_mask]
    masked_recovered = recovered_pert[analysis_mask]

    if len(masked_input) == 0:
        logger.warning("No valid regions found for resolution analysis")
        return {
            'phase': phase,
            'correlation': 0,
            'amplitude_ratio': 0,
            'rms_input': 0,
            'rms_recovered': 0,
            'well_resolved_fraction': 0,
            'total_nodes': len(input_pert.flatten()),
            'voronoi_nodes': 0
            }
    
    # Pattern correlation (handles spatial shifts better)
    correlation = np.corrcoef(masked_input.flatten(), masked_recovered.flatten())[0,1]
    
    # RMS amplitude recovery
    rms_input = np.sqrt(np.mean(masked_input**2))
    rms_recovered = np.sqrt(np.mean(masked_recovered**2))
    amplitude_ratio = rms_recovered / rms_input if rms_input > 0 else 0

    # Sign recovery (polarity test)
    correct_polarity = np.sign(masked_recovered) == np.sign(masked_input)
    polarity_recovery = np.sum(correct_polarity) / len(correct_polarity)

    # Relaxed amplitude recovery test
    recovery_ratio = np.abs(masked_recovered) / (np.abs(masked_input) + 1e-10)
    well_recovered_relaxed = (
        (recovery_ratio > 0.15) &  # Only 15% amplitude recovery required
        (np.sign(masked_recovered) == np.sign(masked_input))
    )
    well_resolved_fraction = np.sum(well_recovered_relaxed) / len(well_recovered_relaxed)

    # Variance reduction (how much pattern variance is explained)
    input_var = np.var(masked_input)
    residual_var = np.var(masked_input - masked_recovered)
    variance_reduction = 1 - (residual_var / input_var) if input_var > 0 else 0

    # Add coverage statistics
    n_coverage = np.sum(coverage_mask)
    n_analysis = np.sum(analysis_mask)
    coverage_fraction = n_coverage / len(input_pert.flatten())

    metrics = {
        'phase': phase,
        'correlation': float(correlation),
        'amplitude_ratio': float(amplitude_ratio),
        'polarity_recovery': float(polarity_recovery),
        'well_resolved_fraction': float(well_resolved_fraction),
        'variance_reduction': float(variance_reduction),
        'rms_input': float(rms_input),
        'rms_recovered': float(rms_recovered),
        'total_nodes': int(len(input_pert.flatten())),
        'coverage_nodes': int(n_coverage),
        'analysis_nodes': int(n_analysis),
        'coverage_fraction': float(coverage_fraction)
    }

    logger.info(f"Checkerboard resolution results for {phase}:   ###")
    logger.info(f"  Ray coverage: {n_coverage}/{len(input_pert.flatten())} nodes ({coverage_fraction:.1%})   ###")
    logger.info(f"  Analysis coverage: {n_analysis}/{n_coverage} of covered nodes   ###")
    logger.info(f"  Pattern correlation: {correlation:.3f}   ###")
    logger.info(f"  Amplitude recovery: {amplitude_ratio:.3f}   ###")
    logger.info(f"  Polarity recovery: {polarity_recovery:.3f}   ###")
    logger.info(f"  Well-resolved fraction: {well_resolved_fraction:.3f}   ###")
    logger.info(f"  Variance explained: {variance_reduction:.3f}   ###")

    return metrics


def _create_voronoi_coverage_mask(iterator, model, phase):
    """Create mask for areas covered by well-sampled Voronoi cells"""

    min_rays = iterator.cfg["algorithm"]["min_rays_per_cell"]

    if iterator.sensitivity_matrix is None:
        logger.warning("No sensitivity matrix available for Voronoi masking")
        return np.ones(model.npts, dtype=bool)

    # Get ray counts per Voronoi cell
    nvoronoi = len(iterator.voronoi_cells)
    sensitivity_voronoi = iterator.sensitivity_matrix.tocsr()[:len(iterator.residuals), :nvoronoi]
    sensitivity_coo = sensitivity_voronoi.tocoo()
    ray_counts = np.bincount(sensitivity_coo.col, minlength=nvoronoi)

    valid_cells = ray_counts >= min_rays

    if iterator.projection_matrix is None:
        logger.warning("No projection matrix available for Voronoi masking")
        return np.ones(model.npts, dtype=bool)

    # Create spatial mask based on projection matrix
    proj_csr = iterator.projection_matrix.tocsr()
    node_mask = np.zeros(np.prod(model.npts), dtype=bool)

    for node_idx in range(np.prod(model.npts)):
        voronoi_idx = proj_csr[node_idx, :].indices
        if len(voronoi_idx) > 0:
            if np.any(valid_cells[voronoi_idx]):
                node_mask[node_idx] = True

    node_mask = node_mask.reshape(model.npts)
    logger.info(f"Voronoi mask covers {np.sum(node_mask)}/{np.prod(model.npts)} model nodes")

    return node_mask


def _find_latest_files(results_dir):
    """Find latest iteration files in results directory."""
    event_files = glob.glob(os.path.join(results_dir, "*.events.h5"))
    if not event_files:
        return None, None, None

    iter_nums = []
    for f in event_files:
        basename = os.path.basename(f)
        if basename[:2].isdigit():
            iter_nums.append(int(basename[:2]))

    if not iter_nums:
        return None, None, None

    latest_iter = max(iter_nums)

    latest_events = os.path.join(results_dir, f"{latest_iter:02d}.events.h5")

    pmodel_files = glob.glob(os.path.join(results_dir, f"{latest_iter:02d}.pwave_model*.h5"))
    smodel_files = glob.glob(os.path.join(results_dir, f"{latest_iter:02d}.swave_model*.h5"))

    latest_pmodel = pmodel_files[0] if pmodel_files else None
    latest_smodel = smodel_files[0] if smodel_files else None

    logger.info(f"Loading latest results:")
    logger.info(f"  Catalog: {latest_events}")
    logger.info(f"  P-model: {latest_pmodel}")
    logger.info(f"  S-model: {latest_smodel}")

    return latest_events, latest_pmodel, latest_smodel


# this one includes stations (not yet in use though)
def _find_latest_files_new(results_dir):
    """Find latest iteration files in results directory"""
    event_files = glob.glob(os.path.join(results_dir, "*.events.h5"))
    if not event_files:
        return None, None, None, None

    iter_nums = []
    for f in event_files:
        basename = os.path.basename(f)
        if basename[:2].isdigit():
            iter_nums.append(int(basename[:2]))

    if not iter_nums:
        return None, None, None, None

    latest_iter = max(iter_nums)

    latest_stations = os.path.join(results_dir, f"{latest_iter:02d}.stations.h5")
    latest_events = os.path.join(results_dir, f"{latest_iter:02d}.events.h5")

    pmodel_files = glob.glob(os.path.join(results_dir, f"{latest_iter:02d}.pwave_model*.h5"))
    smodel_files = glob.glob(os.path.join(results_dir, f"{latest_iter:02d}.swave_model*.h5"))

    latest_pmodel = pmodel_files[0] if pmodel_files else None
    latest_smodel = smodel_files[0] if smodel_files else None

    logger.info(f"Loading latest results:")
    logger.info(f" Stations: {latest_stations}")
    logger.info(f"  Catalog: {latest_events}")
    logger.info(f"  P-model: {latest_pmodel}")
    logger.info(f"  S-model: {latest_smodel}")

    return latest_stations, latest_events, latest_pmodel, latest_smodel


def _save_results(output_dir, input_model, recovered_model, metrics, phase, horiz_block_size_km,tag=""):
    """Save checkerboard test results"""

    suffix = f"_checkerboard_{phase}_{horiz_block_size_km}km"
    if tag:
        suffix += f"_{tag}"

    input_model.to_hdf(os.path.join(output_dir, f"input{suffix}.h5"))
    recovered_model.to_hdf(os.path.join(output_dir, f"recovered{suffix}.h5"))
    
    with open(os.path.join(output_dir, f"metrics{suffix}.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Checkerboard results saved with suffix: {suffix}")


def _copy_scalar_field(field):
    """
    Create a DEEP copy of a ScalarField3D object

    Parameters:
    -----------
    field : _picklabel.ScalarField3D
        The field to copy

    Returns:
    --------
    _picklabel.ScalarField3D
        A new field object with copied attributes
    """
    new_field = _picklabel.ScalarField3D(coord_sys=field.coord_sys)
    new_field.min_coords = field.min_coords.copy()
    new_field.node_intervals = field.node_intervals.copy()
    new_field.npts = field.npts.copy()
    new_field.values = field.values.copy()
    return new_field
