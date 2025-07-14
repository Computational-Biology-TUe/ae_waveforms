import os
import glob
import torch
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from skopt import gp_minimize
from skopt.space import Integer

from main_ae import autoencoder
from functions_ae.load_data import load_datasets, Datasets
from functions_ae.loss_functions import sample_rmse_loss, sample_prd_loss
from functions_traditional.signal_reconstruction import reconstruct_gaussian

import config as cfg

#%% Load the data

all_x, all_a, all_c = load_datasets(cfg.path_data)
indices_test_file = glob.glob(os.path.join(cfg.path_data, 'indices_test_*.npy'))[0]
indices_test = np.load(indices_test_file)

test_set = Datasets(all_x=all_x, all_a=all_a, all_c=all_c,
                    flag=indices_test, swap_dim=False)

ecg_median_samples = test_set.wave_pt.squeeze(2).numpy()
sample_length = ecg_median_samples.shape[1]

test_true = torch.tensor(ecg_median_samples - ecg_median_samples.max(axis=1, keepdims=True), dtype=torch.float32)
criterion = torch.nn.MSELoss()

# Restructure the first 10*2 (x, y) features to order of occurrence that are now in following order:
# "ECG_P_Peaks", "ECG_P_Onsets", "ECG_P_Offsets",
# "ECG_Q_Peaks", "ECG_R_Onsets", "ECG_R_Offsets", "ECG_S_Peaks",
# "ECG_T_Peaks", "ECG_T_Onsets", "ECG_T_Offsets",
features_x = test_set.aswh_pt[:, [2, 0, 4, 8, 6, 12, 10, 16, 14, 18]].numpy()
features_y = test_set.aswh_pt[:, [3, 1, 5, 9, 7, 13, 11, 17, 15, 19]].numpy()
# insert 0 (or very small y) for R-peak at the 6th position
features_x = np.insert(features_x ,5, 0, axis=1)
features_y = np.insert(features_y, 5, -1e-10, axis=1)

baselines = np.nanmean(test_set.aswh_pt[:, [3, 5, 9, 11, 17, 19]].numpy(), axis=1)

x_time = np.linspace(-106, 214 - 1, sample_length) / cfg.sampling_frequency * 1000

#%% Optimize the sigma map using Bayesian optimization

now = time.time()

# sigma_map0 = [1, 2, 2, 2, 1, 3, 3, 1, 2, 2, 2, 1]
sigma_map0 = [1, 1, 1, 2, 2, 3, 3, 3, 4, 2, 2, 1]

# Define the parameter bounds for sigma_map
param_bounds = [Integer(1, 5) for _ in range(len(sigma_map0))]

# Counter to track the number of optimization calls
call_counter = 0

def objective_function(params):
    global call_counter
    call_counter += 1  # Increment counter
    current_sigma_map = np.array(params)

    print(f"Optimization Call {call_counter} / 2000 - Trying params: {current_sigma_map}")

    # Reconstruct ECG signal for all samples
    y_ecg_syn = []
    with tqdm(total=len(ecg_median_samples), desc=f"Call {call_counter} / 2000 - Processing samples") as func_pbar:
        for s in range(len(ecg_median_samples)):
            # Reconstruct ECG signal based on current parameter set
            y_ecg_syn.append(
                reconstruct_gaussian(features_x[s, :], features_y[s, :], x_time, baselines[s], sigma_map=current_sigma_map))
            func_pbar.update(1)

    y_ecg_syn = np.array(y_ecg_syn)

    # Convert reconstructed signal to tensor for loss computation
    pred = torch.tensor(y_ecg_syn, dtype=torch.float32)

    # Compute loss
    loss = criterion(pred, test_true)

    # Print current loss during each iteration
    print(f"Call {call_counter} - Loss: {loss.item()}")

    return loss.item()

# Perform Bayesian optimization
result = gp_minimize(
    objective_function,
    dimensions=param_bounds,
    n_calls=2000,  # Start with 50 iterations, you can adjust this number based on your time/resource constraints
    random_state=42,
    x0=sigma_map0  # Set the initial value for the optimization
)

# Best found parameters and corresponding test loss
best_params = result.x
best_loss = result.fun

print(f"Best Parameters: {best_params}")
print(f"Best Test Loss: {best_loss}")

print(f"Total Calls: {call_counter}")
print(f"Time taken: {time.time() - now}")

import pickle as pkl

# Save the result to a pickle file
with open('results/test_traditional/sigma_map_optimization_val.pkl', 'wb') as f:
    pkl.dump(result, f)  # type: ignore


#%% Reconstruct ECG signal based on best parameters

# sigma_map = [1, 2, 2, 2, 1, 3, 3, 1, 2, 2, 2, 1]
sigma_map = [1, 2, 2, 3, 2, 3, 3, 2, 3, 2, 2, 1]


y_gauss = []
with tqdm(total=len(ecg_median_samples)) as inner_pbar:
    for i in range(len(ecg_median_samples)):
        # Reconstruct ECG signal based on current parameter set
        y_gauss.append(
            reconstruct_gaussian(features_x[i, :], features_y[i, :], x_time, baselines[i], sigma_map=sigma_map))
        inner_pbar.update(1)

y_gauss = np.array(y_gauss)

#%% Plot Results for sample at index i

fig, axes = plt.subplots(4, 5, figsize=(28, 14))
indices = np.linspace(20, len(ecg_median_samples)-1, 20).astype(int)

for plot, i in enumerate(indices):
    row, col = divmod(plot, 5)

    axes[row, col].plot(x_time, ecg_median_samples[i, :] - ecg_median_samples[i, :].max(), label='Original ECG', linewidth=2)
    axes[row, col].plot(x_time, y_gauss[i, :], label='Gaussian-based Interpolation')
    # axes[row, col].plot(x_time, y_akima[i, :], label='Akima Interpolation')
    # axes[row, col].plot(x_time, y_pchip[i, :], label='PCHIP Interpolation')
    # axes[row, col].plot(x_time, y_akima_pchip_mean[i, :], label='Akima-PCHIP Mean')
    axes[row, col].scatter(features_x[i, :], features_y[i, :], color='black', marker='x', label='Fiducial Points')

    axes[row, col].set_xlabel('Time')
    axes[row, col].set_ylabel('ECG Signal')
    axes[row, col].set_ylim(min(features_y[i, :])-0.5, 0.5)
    axes[row, col].legend()
    axes[row, col].grid()

plt.suptitle(str(sigma_map))
plt.tight_layout()
plt.show()


#%% Measure classical and ae method performance

# Store results for all methods
mse_losses = []
rmse_sample_avgs = []
rmse_sample_medians = []
rmses_per_sample = []
prd_sample_avgs = []
prd_sample_medians = []
prds_per_sample = []
labels = []
test_pred_l = []
test_true_l = []

# Compute results for classical method
target = "Traditional, d=20"

# Compute ground truth values
test_true = torch.tensor(ecg_median_samples, dtype=torch.float32)
test_pred = torch.tensor(y_gauss + ecg_median_samples.max(axis=1, keepdims=True), dtype=torch.float32)

# Define loss function
criterion = torch.nn.MSELoss()
test_loss = criterion(test_pred, test_true)
# print(f"Global test MSE loss: {test_loss:.6f}")

# Calculate and print model performance metrics
# Sample mean RMSE
rmse_sample_avg, rmse_per_sample = sample_rmse_loss(test_pred, test_true)
# print(f"Sample wise average test RMSE loss: {rmse_sample_avg:.6f}")

# Sample mean Percentage RMS Difference (PRD)
prd_sample_avg, prd_per_sample = sample_prd_loss(test_pred, test_true)
# print(f"Sample wise average test PRD loss: {prd_sample_avg:.6f}")

cr = 320 / 20

qs_per_sample = cr / prd_per_sample.cpu().numpy()

# Print the median [Q1-Q3] RMSE and QS values
print(f"Median RMSE: {np.median(rmse_per_sample.cpu().numpy()):.6f}")
print(f"RMSE [Q1-Q3]: {np.percentile(rmse_per_sample.cpu().numpy(), 25):.6f} - {np.percentile(rmse_per_sample.cpu().numpy(), 75):.6f}")
print(f"Median QS: {np.median(qs_per_sample):.6f}")
print(f"QS [Q1-Q3]: {np.percentile(qs_per_sample, 25):.6f} - {np.percentile(qs_per_sample, 75):.6f}")

# Print the mean and sd RMSE and QS values
print(f"Mean RMSE: {rmse_sample_avg:.6f}")
print("RMSE SD: ", np.std(rmse_per_sample.cpu().numpy()))
print(f"Mean QS: {np.mean(qs_per_sample):.6f}")
print("QS SD: ", np.std(qs_per_sample))


# Store data for boxplot
mse_losses.append(test_loss.item())
rmse_sample_avgs.append(rmse_sample_avg.item())
rmse_sample_medians.append(np.median(rmse_per_sample.cpu().numpy()))
rmses_per_sample.append(rmse_per_sample.cpu().numpy())
prd_sample_avgs.append(prd_sample_avg.item())
prd_sample_medians.append(np.median(prd_per_sample.cpu().numpy()))
prds_per_sample.append(prd_per_sample.cpu().numpy())
labels.append(target)
test_pred_l.append(test_pred)
test_true_l.append(test_true)


# Compute results for AE method
args_ls10 = ['--mode', 'test', '--trained_model', f'AUT-231', '--latent_size', str(10), '--seed', str(3), '--plotting', False]
args_ls20 = ['--mode', 'test', '--trained_model', f'AUT-296', '--latent_size', str(20), '--seed', str(7), '--plotting', False]

for args in [args_ls20, args_ls10]:
    test_loss, rmse, rmse_samples, prd, prd_samples, test_pred, test_true, test_c = autoencoder(args)

    cr = 320 / int(args[5])

    qs_samples = cr / prd_samples

    # Print the median [Q1-Q3] RMSE and QS values
    print(f"Median RMSE: {np.median(rmse_samples):.6f}")
    print(f"RMSE [Q1-Q3]: {np.percentile(rmse_samples, 25):.6f} - {np.percentile(rmse_samples, 75):.6f}")
    print(f"Median QS: {np.median(qs_samples):.6f}")
    print(f"QS [Q1-Q3]: {np.percentile(qs_samples, 25):.6f} - {np.percentile(qs_samples, 75):.6f}")

    # Print the mean and sd RMSE and QS values
    print(f"Mean RMSE: {rmse:.6f}")
    print("RMSE SD: ", np.std(rmse_samples))
    print(f"Mean QS: {np.mean(qs_samples):.6f}")
    print("QS SD: ", np.std(qs_samples))

    # Store data for boxplot
    mse_losses.append(test_loss)
    rmse_sample_avgs.append(rmse)
    rmse_sample_medians.append(np.median(rmse_samples))
    rmses_per_sample.append(rmse_samples)
    prd_sample_avgs.append(prd)
    prd_sample_medians.append(np.median(prd_samples))
    prds_per_sample.append(prd_samples)
    labels.append(f"AE, d={args[5]}")
    test_pred_l.append(test_pred)
    test_true_l.append(test_true)


# ---- Plot all box plots in one figure ----
fig, ax = plt.subplots(figsize=(10, 7))

# Create boxplot
box = sns.boxplot(
    data=rmses_per_sample,
    orient='h',
    ax=ax,
    color='skyblue',  # Box color
    width=0.5,  # Controls the thickness of the box
    showfliers=False,  # **Hides values outside Q1-Q3**
    boxprops=dict(edgecolor="black", linewidth=1.5),  # Box outline
    whiskerprops=dict(color="black", linewidth=1.2),
    capprops=dict(color="black", linewidth=1.2),
    medianprops=dict(color="blue", linewidth=1.5),  # Median line
)

# Add vertical red dashed lines for the average RMSE **within each respective boxplot**
for i, avg in enumerate(rmse_sample_avgs):
    ax.plot([avg, avg], [i - 0.3, i + 0.3], color='red', linestyle='--', linewidth=1.5)

# Set labels and title
ax.set_yticklabels(labels, rotation=45)
ax.set_xlabel("RMSE")
ax.set_xlim(0, 0.45)
# ax.set_title(f"RMSE Box plots for Different Targets: {sigma_map}")
ax.grid(True, linestyle="--", alpha=0.6)

# Add a single legend entry for Median and Mean
handles = [
    plt.Line2D([0], [0], color="blue", linewidth=1.5, label="Median RMSE"),
    plt.Line2D([0], [0], color="red", linestyle="--", linewidth=1.5, label="Mean RMSE"),
]
ax.legend(handles=handles, loc="lower right")

# Show plot
plt.tight_layout()
plt.show()

# x, y, x_t, baseline = (features_x[i, :], features_y[i, :], x_time, baselines[i])

#%% Local error

# Calculate the median waveform over the test set
ecg_median = np.median(ecg_median_samples, axis=0)

fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(nrows=2, ncols=3, width_ratios=[1, 1, 1], height_ratios=[0.4, 1])

ax_bottom_left = fig.add_subplot(gs[1, 0])
ax_top_left = fig.add_subplot(gs[0, 0], sharex=ax_bottom_left)  # Share x and y-axis
ax_bottom_center = fig.add_subplot(gs[1, 1], sharey=ax_bottom_left)  # Share y-axis
ax_top_center = fig.add_subplot(gs[0, 1], sharex=ax_bottom_center, sharey=ax_top_left)  # Share x and y-axis
ax_bottom_right = fig.add_subplot(gs[1, 2], sharey=ax_bottom_left)  # Share y-axis
ax_top_right = fig.add_subplot(gs[0, 2], sharex=ax_bottom_right, sharey=ax_top_left)  # Share x and y-axis

col = 0
for (test_pred, test_true, label) in zip(test_pred_l, test_true_l, labels):

    if 'AE' in label:
        test_pred = test_pred.cpu().squeeze(2).numpy()
        test_true = test_true.cpu().squeeze(2).numpy()

    # Calculate the local error
    local_error = test_pred - test_true
    local_abs_error = np.abs(test_pred - test_true)

    # Compute median and IQR
    median_error = np.median(local_error, axis=0)
    q1_error = np.percentile(local_error, 25, axis=0)  # 25th percentile (Q1)
    q3_error = np.percentile(local_error, 75, axis=0)  # 75th percentile (Q3)
    median_abs_error = np.median(local_abs_error, axis=0)
    q1_abs_error = np.percentile(local_abs_error, 25, axis=0)  # 25th percentile (Q1)
    q3_abs_error = np.percentile(local_abs_error, 75, axis=0)  # 75th percentile (Q3)

    # Create X-axis (assuming time or sample index)
    x = np.arange(ecg_median.shape[0])*2

    # ---- First subplot: Median Error with IQR ----
    if col == 0:
        ax1 = ax_top_left
    elif col == 1:
        ax1 = ax_top_center
    else:
        ax1 = ax_top_right
    ax1.plot(x, median_error, color='red', linewidth=2, label='Median Error')
    ax1.fill_between(x, q1_error, q3_error, color="red", alpha=0.3, label="IQR (25th-75th percentile)")
    ax1.plot(x, median_abs_error, color='blue', linewidth=2, label='Median Absolute Error')
    ax1.fill_between(x, q1_abs_error, q3_abs_error, color="blue", alpha=0.3, label="IQR (25th-75th percentile)")
    ax1.set_xlim(x[0], x[-1])
    ax1.set_ylim(-0.80, 1.15)
    ax1.grid()
    ax1.tick_params(labelbottom=False)  # Hide x-axis labels and ticks for top row
    if col >= 1:
        ax1.tick_params(labelleft=False)  # Hide y-axis labels and ticks for right column
    else:
        ax1.set_ylabel("Discrepancy")

    # ---- Third subplot: ECG Median with Shaded Error ----
    if col == 0:
        ax3 = ax_bottom_left
    elif col == 1:
        ax3 = ax_bottom_center
    else:
        ax3 = ax_bottom_right
    ax3.plot(x, ecg_median, color='black', linewidth=2, label='Median ECG')
    ax3.fill_between(x, ecg_median + np.minimum(0, median_error), ecg_median, color='red', alpha=0.3, label="Median Error")
    ax3.fill_between(x, ecg_median + np.minimum(0, q1_error), ecg_median, color='red', alpha=0.3)
    ax3.fill_between(x, ecg_median + np.maximum(0, median_error), ecg_median, color='red', alpha=0.3)
    ax3.fill_between(x, ecg_median + np.maximum(0, q3_error), ecg_median, color='red', alpha=0.3)

    ax3.fill_between(x, ecg_median - median_abs_error, ecg_median + median_abs_error, color="blue", alpha=0.3, label="Median Abs Error")
    ax3.fill_between(x, ecg_median - median_error - q1_abs_error, ecg_median + median_error + q3_abs_error, color="blue", alpha=0.3)

    ax3.set_xlabel("Time (ms)")
    ax3.set_xlim(x[0], x[-1])
    ax3.set_ylim(-1, 5)
    ax3.grid()
    if col >= 1:
        ax3.tick_params(labelleft=False)  # Hide y-axis labels and ticks for right column
    else:
        ax3.set_ylabel("Amplitude")
    col += 1

plt.tight_layout()

fig.savefig("analysis_local_error_ae_classic.svg", format="svg")

plt.show()
