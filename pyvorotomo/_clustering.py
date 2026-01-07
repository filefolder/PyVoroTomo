import numpy as np
from scipy.spatial import cKDTree
import pykonal

from . import _utilities

# Get logger handle.
logger = _utilities.get_logger(f"__main__.{__name__}")


@_utilities.log_errors(logger)
def _init_centroids(k, xyz):
    """
    Optimized kmeans++ initialization using cKDTree for faster nearest neighbor lookup

    (xyz already pre-transformed via k_medians)
    """

    n_points = len(xyz)
    
    first_idx = np.random.randint(n_points)
    centroids = [xyz[first_idx]]

    # Select the remaining centroids
    for _ in range(1, k):
        centroid_tree = cKDTree(centroids)
        distances, _ = centroid_tree.query(xyz)

        sum_dist_square = np.sum(distances**2)
        if sum_dist_square > 0:
            prob = distances**2 / sum_dist_square
            next_idx = np.random.choice(n_points, p=prob)
        else:
            next_idx = np.random.choice(n_points)
            logger.warning("sum(distances**2) was zero in _init_centroids... kvoronoi too high?")

        centroids.append(xyz[next_idx])

    return np.array(centroids)


@_utilities.log_errors(logger)
def k_medians(k, points, model_bounds, max_iter=15, change_threshold=0.01):
    """
    Optimized k-medians implementation with early stopping based on percentage change.
    
    Parameters:
    k - Number of clusters
    points - Data points to cluster (in spherical coordinates)
    model bounds - Tuple of (min_coords,max_coords)
    max_iter - Maximum number of iterations
    change_threshold - Exit when median movement is less than this percentage (default: 0.01 = 1%) *usually < .01 in first step)
    """

    xyz = pykonal.transformations.sph2xyz(points, (0, 0, 0))
    n_points = len(xyz)

    if k > n_points:
        logger.warning(f"k_medians: k={k} exceeds point count ({n_points}), reducing k")
        k = n_points    
    medians = _init_centroids(k, xyz)

    # Pre-allocate assignment array
    indexes = np.zeros(n_points, dtype=np.int32)

    for iter_count in range(max_iter):
        # Store previous medians to measure change
        prev_medians = medians.copy()

        # Assign points to nearest median
        tree = cKDTree(medians)
        _, indexes = tree.query(xyz)

        # Update each median based on assigned points
        for i in range(k):
            cluster_points = xyz[indexes == i]
            if len(cluster_points) > 0:
                medians[i] = np.median(cluster_points, axis=0)

        distances = np.sqrt(np.sum((medians - prev_medians)**2, axis=1))

        # Calculate relative change as a percentage of the median's distance from origin
        # Adding a small epsilon to avoid division by zero
        median_magnitudes = np.sqrt(np.sum(prev_medians**2, axis=1)) + 1e-10
        percentage_changes = distances / median_magnitudes

        # Exit if the maximum percentage change is below the threshold
        if np.max(percentage_changes) < change_threshold:
            #logger.debug(f"k_medians broke threshold at {iter_count} loops ({np.max(percentage_changes)})")
            break

        if iter_count == max_iter-1:
            logger.warning(f"warning: k_medians did not converge in {iter_count} loops ({np.max(percentage_changes)})")
    
    # Convert back to spherical coordinates
    medians_sph = pykonal.transformations.xyz2sph(medians, origin=(0, 0, 0))

    return medians_sph
