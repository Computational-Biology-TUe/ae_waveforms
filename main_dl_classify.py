import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import time
import csv
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

from functions_dl.model_classes import LSTMClassifier
from functions_dl.load_data import Datasets, load_datasets

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
NUM_EPOCHS = 153
filename_condition = "hyposodium.csv"  # <-- Change here
filename_control = "controls.csv"

# --------------------------------------------------
# Data Loading and Preprocessing
# --------------------------------------------------

filepath_env = 'vars.env'
load_dotenv(filepath_env)
path_data = os.environ['DATA_FOLDER_DEEP_LEARNING_PATH']

# Load subject numbers
control_numbers = np.loadtxt(os.path.join(path_data, filename_control), delimiter=",", dtype=int, skiprows=1, usecols=1)
sick_numbers = np.loadtxt(os.path.join(path_data, filename_condition), delimiter=",", dtype=int, skiprows=1, usecols=1)

# Load data
np_waves, np_features, np_info = load_datasets(path_data)

# Index matching
idx_control = np.nonzero(np.isin(np_info[:, 0], control_numbers))[0]
idx_condition = np.nonzero(np.isin(np_info[:, 0], sick_numbers))[0]

# Get features
control_features = np_waves[idx_control, :12000]
sick_features = np_waves[idx_condition, :12000]

# Fix test set: 250 control + 250 sick
test_size = 250
control_test = control_features[-test_size:]
sick_test = sick_features[-test_size:]
X_test = np.concatenate([control_test, sick_test], axis=0)
y_test = np.concatenate([np.zeros(test_size), np.ones(test_size)])
X_test = np.expand_dims(X_test, axis=-1)

# Remaining for training
control_train_all = control_features[:-test_size]
sick_train_all = sick_features[:-test_size]

# --------------------------------------------------
# Training Function
# --------------------------------------------------

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
                patience=50, min_delta=0.0005):
    best_auc = 0.0
    best_model_wts = None
    epochs_no_improve = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        auc = roc_auc_score(all_labels, all_preds)

        if auc - best_auc > min_delta:
            best_auc = auc
            best_model_wts = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs. Best AUC: {best_auc:.4f}")
            break

        scheduler.step()

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    return model


# --------------------------------------------------
# Create Output Directory
# --------------------------------------------------

output_dir = os.path.join(path_data, f"{NUM_EPOCHS}_{filename_condition.replace('.csv', '')}")
os.makedirs(output_dir, exist_ok=True)

# --------------------------------------------------
# Main Loop for Size Variation (with Saving)
# --------------------------------------------------

start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sizes = [100, 250, 500, 750, 1000, 2000, 3000, 4000, 6000, 10000, 11500]
all_medians, all_q1s, all_q3s = [], [], []

for size in sizes:
    aucs = []
    size_dir = os.path.join(output_dir, f"size_{size}")
    os.makedirs(size_dir, exist_ok=True)

    for seed in range(25):
        rng = np.random.default_rng(seed + 42)
        idx_control = rng.choice(len(control_train_all), size, replace=False)
        idx_sick = rng.choice(len(sick_train_all), size, replace=False)
        X_train = np.concatenate([control_train_all[idx_control], sick_train_all[idx_sick]], axis=0)
        y_train = np.concatenate([np.zeros(size), np.ones(size)])
        X_train = np.expand_dims(X_train, axis=-1)

        X_train_part, X_val_part, y_train_part, y_val_part = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=seed
        )

        train_dataset = Datasets(X_train_part, y_train_part)
        val_dataset = Datasets(X_val_part, y_val_part)
        train_loader = data.DataLoader(train_dataset, batch_size=512, shuffle=True)
        val_loader = data.DataLoader(val_dataset, batch_size=512, shuffle=False)

        model = LSTMClassifier().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.5)

        model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device)

        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            preds = model(X_test_tensor).squeeze().cpu().numpy()
        auc = roc_auc_score(y_test, preds)
        aucs.append(auc)

        # Save model
        model_filename = os.path.join(size_dir, f"model_seed_{seed}.pt")
        torch.save(model.state_dict(), model_filename)

    # Save per-size AUCs to CSV
    auc_csv_path = os.path.join(size_dir, f"aucs_size_{size}.csv")
    with open(auc_csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "auc"])
        for seed, auc_val in enumerate(aucs):
            writer.writerow([seed, auc_val])

    # Compute summary stats
    median_auc = np.median(aucs)
    q1 = np.percentile(aucs, 25)
    q3 = np.percentile(aucs, 75)
    all_medians.append(median_auc)
    all_q1s.append(q1)
    all_q3s.append(q3)

    print(f"Size: {size}, Median AUC: {median_auc:.4f}, IQR: ({q1:.4f}, {q3:.4f})")

# --------------------------------------------------
# Save Summary Statistics
# --------------------------------------------------

summary_csv_path = os.path.join(output_dir, "summary_auc_stats.csv")
with open(summary_csv_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["size", "median_auc", "q1", "q3"])
    for size, med, q1, q3 in zip(sizes, all_medians, all_q1s, all_q3s):
        writer.writerow([size, med, q1, q3])

# --------------------------------------------------
# Plot Results
# --------------------------------------------------

plt.figure(figsize=(8, 6))
sizes = np.array(sizes)
all_medians = np.array(all_medians)
all_q1s = np.array(all_q1s)
all_q3s = np.array(all_q3s)

plt.plot(sizes, all_medians, '-o', label='Median AUC')
plt.fill_between(sizes, all_q1s, all_q3s, alpha=0.2, label='IQR (25th–75th percentile)')

ref_auc = all_medians[-1]
plt.axhline(ref_auc, color='black', linestyle='--', linewidth=1.5, label='100% of Max AUC')
plt.axhline(0.975 * ref_auc, color='gray', linestyle='--', linewidth=1.5, label='97.5% of Max AUC')
plt.axhline(0.95 * ref_auc, color='lightgray', linestyle='--', linewidth=1.5, label='95% of Max AUC')

plt.xlabel('Number of Patients per Class')
plt.ylabel('ROC AUC on Fixed Test Set')
plt.grid(True)
plt.xticks(sizes, rotation=45)
plt.legend(loc='lower right')
plt.title('End-to-end Model Performance by Training Size')
plt.tight_layout()
plt_path = os.path.join(output_dir, "performance_plot.png")
plt.savefig(plt_path)
plt.show()

end_time = time.time()
elapsed_minutes = (end_time - start_time) / 60
print(f"\nTotal execution time: {elapsed_minutes:.2f} minutes")
