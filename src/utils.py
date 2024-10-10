import numpy as np
import rasterio
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.morphology import disk, dilation
from skimage.morphology import erosion
from skimage.filters import rank
from scipy.interpolate import CubicSpline, PchipInterpolator
from datetime import datetime
#import rank

def datetime_to_ordinal(dates):
    """Convert datetime objects to days since the first date in the list."""
    base_date = dates[0]
    return np.array([(date - base_date).days for date in dates])

def load_folder(folder, func=None, func_args=None, disable=True):
    files = os.listdir(folder)
    files = [f for f in files if (f.endswith('.tif') or f.endswith('.tiff') or f.endswith('.png'))]
    files.sort()
    data = []
    for file in tqdm(files, disable=disable):
        with rasterio.open(os.path.join(folder, file)) as src:
            mask = src.read().squeeze()
            if func is not None:
                mask = func(mask, **func_args)
            data.append(mask)
    return np.array(data)

def postprocess_cloud_mask(cloud_mask : np.array, n : int = 5, nm : int = 20) -> np.array:
    dilated = dilation(cloud_mask, disk(n))
    dilated = dilated.astype(float)
    mean = rank.mean(dilated, disk(nm)) / 255
    return mean.astype('float32')

import numpy as np
from datetime import datetime
from typing import List, Tuple

def fit_periodic_function_with_harmonics_robust(
    time_series: np.ndarray,
    qa: np.ndarray,
    dates: List[datetime],
    num_harmonics: int = 3,
    max_iter: int = 10,
    tol: float = 5e-2,  # Adjusted tolerance for relative change
    percentile: float = 75.0,  # Percentile for convergence criterion
    min_param_threshold: float = 1e-5,  # Threshold to consider parameter significant
    verbose=0,
    debug=False
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    # Convert dates to 'datetime64' and compute normalized time as fraction of year
    times_datetime64 = np.array(dates, dtype='datetime64[D]')
    start_date = times_datetime64[0]
    days_since_start = (times_datetime64 - start_date).astype(int)
    t_normalized = days_since_start / 365.25  # Normalize to fraction of year

    # Initial design matrix with harmonics and constant term
    harmonics = []
    for k in range(1, num_harmonics + 1):
        t_radians = 2 * np.pi * k * t_normalized
        harmonics.extend([np.cos(t_radians), np.sin(t_radians)])

    A = np.stack(harmonics + [np.ones_like(t_normalized)], axis=-1)  # Design matrix

    # Reshape time_series and qa for vectorized operations
    pixels = time_series.reshape(time_series.shape[0], -1)
    weights = qa.reshape(qa.shape[0], -1)

    # Initialize delta
    delta = 1.345

    # Compute the pseudoinverse of the design matrix
    A_pinv = np.linalg.pinv(A)  # Shape: (num_params, time)
    # Initial least squares fit to estimate parameters
    initial_params = np.dot(A_pinv, pixels).T  # Shape: (n_pixels, num_params)
    params = initial_params.copy()
     # Initialize parameters
    num_params = 2 * num_harmonics + 1

    # Calculate initial residuals
    initial_fitted_values = np.dot(A, params.T)
    initial_residuals = pixels - initial_fitted_values

    # Estimate initial sigma
    sigma_initial = np.std(initial_residuals, axis=0)
    sigma_initial[sigma_initial == 0] = np.finfo(float).eps  # Avoid division by zero

    # Set delta based on initial residuals and do not update it
    delta = 1.345 * sigma_initial
    delta[delta == 0] = np.finfo(float).eps  # Avoid zero delta

    epsilon = 1e-8

    for iteration in range(max_iter):
        params_old = params.copy()

        # Broadcasting for weighted design matrix
        A_expanded = np.expand_dims(A, 2)
        weights_expanded = np.expand_dims(weights, 1)
        A_weighted = A_expanded * weights_expanded

        # Compute the normal equation components
        ATA = np.einsum('ijk,ilk->jlk', A_weighted, A_expanded)
        ATb = np.einsum('ijk,ik->jk', A_weighted, pixels)

        # Solve for parameters
        ATA_reshaped = ATA.transpose(2, 0, 1)
        ATb_reshaped = ATb.T
        params = np.array([solve_params(ATA_reshaped[i], ATb_reshaped[i]) for i in range(ATA_reshaped.shape[0])])
        params = np.nan_to_num(params)  # Replace NaNs with zero

        # Calculate fitted values and residuals
        fitted_values = np.dot(A, params.T)  # Shape: (time, n_pixels)
        residuals = pixels - fitted_values

        # Estimate sigma (standard deviation of residuals)
        sigma_residuals = np.std(residuals, axis=0)
        sigma_residuals[sigma_residuals == 0] = np.finfo(float).eps  # Avoid division by zero


        # Update weights based on residuals using Huber loss
        residuals_abs = np.abs(residuals)
        mask = residuals_abs <= delta
        weights_update = np.where(mask, 1, delta / (residuals_abs + epsilon))
        weights = weights * weights_update

        # Compute relative change, avoiding division by small numbers
        min_param_threshold = 1e-5  # Or another appropriate small value
        param_diff = np.abs(params - params_old)
        relative_change = param_diff / (np.maximum(np.abs(params_old), min_param_threshold))
        relative_change_flat = relative_change.flatten()

        if debug:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            _ = ax.hist(param_diff.flatten(), bins=100)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_title(f"Params diff - Iteration {iteration + 1}")
            plt.show()

        # Compute the desired percentile of relative change
        percentile_value = np.percentile(relative_change_flat, percentile)

        if verbose > 0:
            print(f"Iteration {iteration + 1}: {percentile}th percentile of relative parameter change = {percentile_value}")

        # Check for convergence
        if percentile_value < tol:
            if verbose > 0:
                print(f"Converged after {iteration + 1} iterations.")
            break

    # Reshape parameters
    params_reshaped = params.reshape(time_series.shape[1], time_series.shape[2], num_params).transpose(2, 0, 1)

    # Extract amplitude and phase maps
    amplitude_maps = []
    phase_maps = []

    for i in range(num_harmonics):
        A_params = params_reshaped[2 * i]
        B_params = params_reshaped[2 * i + 1]
        amplitude_map = np.sqrt(A_params ** 2 + B_params ** 2)
        phase_map = np.arctan2(B_params, A_params)

        # Adjust and normalize phases
        phase_adjusted = (phase_map - (2 * np.pi * (i + 1) * t_normalized[0])) % (2 * np.pi)
        phase_normalized = np.where(phase_adjusted > np.pi, phase_adjusted - 2 * np.pi, phase_adjusted)

        amplitude_maps.append(amplitude_map)
        phase_maps.append(phase_normalized)

    # Offset map
    offset_map = params_reshaped[-1]

    return (*amplitude_maps, *phase_maps, offset_map)

def solve_params(ATA: np.ndarray, ATb: np.ndarray) -> np.ndarray:
    """ Solve linear equations with error handling for non-invertible cases. """
    try:
        return np.linalg.solve(ATA, ATb)
    except np.linalg.LinAlgError:
        return np.full(ATb.shape, np.nan)  # Return NaN for non-invertible matrices


