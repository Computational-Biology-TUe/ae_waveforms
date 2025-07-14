import numpy as np
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator


def gaussian_line(x1, y1, x2, y2, s):
    a = max(y1, y2)
    mu = x2 if y2 > y1 else x1
    x = np.arange(int(x1), int(x2))
    y = a * np.exp(-((x - mu) ** 2) / (2 * s ** 2))
    if y2 > y1:
        y = (y - y[0]) * (y2 - y1) / (y[-1] - y[0]) + y1
    else:
        y = (y - y[-1]) * (y1 - y2) / (y[0] - y[-1]) + y2
    return y


def reconstruct_gaussian(x, y, x_t, baseline, sigma_map):
    ecg_syn2 = []

    x_coordinates, y_coordinates, nan_mask = extract_coordinates(x, y, x_t, baseline)

    i_nan = 0
    for i_start in range(len(x_coordinates) - 1):
        i_end = i_start + 1
        x_start, x_end = x_coordinates[i_start], x_coordinates[i_end]
        y_start, y_end = y_coordinates[i_start], y_coordinates[i_end]

        sigma = abs(x_end - x_start) / sigma_map[i_nan]
        while nan_mask[i_nan+1]:
            sigma = abs(x_end - x_start) / 1
            i_nan += 1

        if (x_end - x_start > 1) and (x_end > x_t[0]) and (x_start < x_t[-1]):
            ecg_syn2.extend(gaussian_line(x_start, y_start, x_end, y_end, sigma))

        i_nan += 1

    # Slice off the start and end of the signal if it is outside the range of x_t[0], x_t[-1]


    # Resample ecg_syn2 to match x_t length
    return np.interp(x_t, np.linspace(x_t[0], x_t[-1], len(ecg_syn2)), ecg_syn2)


def extract_coordinates(x, y, x_t, baseline):
    # Sample data
    x_coordinates = np.array([x_t[0]] + list(x) + [x_t[-1]])
    y_coordinates = np.array([baseline] + list(y) + [baseline])

    # Create a nan mask for if either the x or y coordinates are nan and remove the corresponding indices
    nan_mask = np.isnan(x_coordinates) | np.isnan(y_coordinates)
    x_coordinates = x_coordinates[~nan_mask]
    y_coordinates = y_coordinates[~nan_mask]

    # Sort coordinates by ascending values of x_coordinates
    sorted_indices = np.argsort(x_coordinates)
    x_coordinates = x_coordinates[sorted_indices]
    y_coordinates = y_coordinates[sorted_indices]

    # Check for duplicates and adjust only if necessary
    unique_x_coordinates, counts = np.unique(x_coordinates, return_counts=True)

    if np.any(counts > 1):  # Only perform operation if there are duplicates

        # If due to changing we get another duplicate, we need to repeat the process
        while np.any(counts > 1):
            # Find the indices of duplicates
            duplicate_indices = np.where(counts > 1)[0]

            # Adjust duplicates by adding the step size
            step_size = x_t[1] - x_t[0]
            for idx in duplicate_indices:
                # Find the indices where the duplicates occur
                duplicate_positions = np.where(x_coordinates == unique_x_coordinates[idx])[0]
                # Skip the first duplicate and add the step size to subsequent ones
                for i in range(1, len(duplicate_positions)):
                    x_coordinates[duplicate_positions[i]] += step_size

            unique_x_coordinates, counts = np.unique(x_coordinates, return_counts=True)

    return x_coordinates, y_coordinates, nan_mask


def reconstruct_akima(x, y, x_t, baseline):
    x_coordinates, y_coordinates, _ = extract_coordinates(x, y, x_t, baseline)

    # 4. Akima Interpolation
    akima_interp = Akima1DInterpolator(x_coordinates, y_coordinates)
    return akima_interp(x_t)


def reconstruct_pchip(x, y, x_t, baseline):
    x_coordinates, y_coordinates, _ = extract_coordinates(x, y, x_t, baseline)

    # 5. PCHIP Interpolation
    pchip_interp = PchipInterpolator(x_coordinates, y_coordinates)
    return pchip_interp(x_t)


def reconstruct_akima_pchip_mean(akima, pchip):
    return (akima + pchip) / 2

