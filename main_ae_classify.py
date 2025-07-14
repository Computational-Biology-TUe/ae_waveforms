import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from dotenv import load_dotenv

from functions_ml.latin_hypercube_sampling import generate_lhs_params


def main():
    # Load environment path
    filepath_env = 'vars.env'
    load_dotenv(filepath_env)
    path_data = os.environ['DATA_FOLDER_HYBRID_LEARNING_PATH']

    # Define data file names
    control_file = "controls.csv"
    sick_file = "hypopotassium.csv"  # Change this if needed

    # Dynamically define result folder name based on sick file
    sick_base = os.path.splitext(os.path.basename(sick_file))[0].split("_")[-1].lower()
    result_folder_name = os.path.join(path_data, f"results/{sick_base}")
    os.makedirs(result_folder_name, exist_ok=True)

    # Load data
    control_numbers = np.loadtxt(os.path.join(path_data, control_file), delimiter=",", dtype=int, skiprows=1, usecols=1)
    sick_numbers = np.loadtxt(os.path.join(path_data, sick_file), delimiter=",", dtype=int, skiprows=1, usecols=1)

    np_info = np.load(os.path.join(path_data, "np_info.npy"), allow_pickle=True)
    np_latent_spaces = np.load(os.path.join(path_data, "np_latent_spaces_mimiciv.npy"), allow_pickle=True)

    try:
        np_info_ids = np.array(np_info[:, 0], dtype=int)
    except:
        np_info_ids = np_info[:, 0]

    control_idx = np.nonzero(np.isin(np_info_ids, control_numbers))[0]
    sick_idx = np.nonzero(np.isin(np_info_ids, sick_numbers))[0]

    control_features = np_latent_spaces[control_idx, :][:6500]
    sick_features = np_latent_spaces[sick_idx, :][:6500]

    # Prepare fixed test set
    test_size = min(250, len(control_features) - 1, len(sick_features) - 1)
    X_test = np.concatenate([control_features[-test_size:], sick_features[-test_size:]], axis=0)
    y_test = np.concatenate([np.zeros(test_size), np.ones(test_size)])

    control_train_all = control_features[:-test_size]
    sick_train_all = sick_features[:-test_size]

    sizes = [100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 6000]#, 10000, 11500

    # Grid search on largest training size
    lhs_params = generate_lhs_params(1000, random_state=1)
    model = xgb.XGBClassifier(eval_metric='logloss', random_state=1)

    max_train_size = min(6000, len(control_train_all), len(sick_train_all))
    control_max = control_train_all[:max_train_size]
    sick_max = sick_train_all[:max_train_size]
    X_full = np.concatenate([control_max, sick_max], axis=0)
    y_full = np.concatenate([np.zeros(len(control_max)), np.ones(len(sick_max))])

    print(f"Running grid search on {len(y_full)} training samples...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=lhs_params,
        scoring='roc_auc',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_full, y_full)
    best_params = grid_search.best_params_
    print("Best hyperparameters found:", best_params)

    auc_records = []
    summary_records = []

    for size in sizes:
        aucs = []
        print(f"Training models with sample size per class = {size}")
        for seed in range(25):
            rng = np.random.default_rng(seed + 42)
            idx_control = rng.choice(len(control_train_all), size, replace=False)
            idx_sick = rng.choice(len(sick_train_all), size, replace=False)
            control_sample = control_train_all[idx_control]
            sick_sample = sick_train_all[idx_sick]

            X_train = np.concatenate([control_sample, sick_sample], axis=0)
            y_train = np.concatenate([np.zeros(size), np.ones(size)])

            model = xgb.XGBClassifier(**best_params, eval_metric='logloss', random_state=seed + 42)
            model.fit(X_train, y_train)

            y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            aucs.append(auc)

            auc_records.append({"size": size, "seed": seed, "auc": auc})

        median_auc = np.median(aucs)
        q1 = np.percentile(aucs, 25)
        q3 = np.percentile(aucs, 75)

        summary_records.append({
            "size": size,
            "median_auc": median_auc,
            "q1_auc": q1,
            "q3_auc": q3
        })

        print(f"Size: {size}, Median AUC: {median_auc:.4f}, IQR: ({q1:.4f}, {q3:.4f})")

    # Save results
    df_auc = pd.DataFrame(auc_records)
    df_summary = pd.DataFrame(summary_records)

    df_auc.to_csv(os.path.join(result_folder_name, "individual_aucs.csv"), index=False)
    df_summary.to_csv(os.path.join(result_folder_name, "summary_auc_by_size.csv"), index=False)

    # Plotting
    sizes_np = np.array(sizes)
    medians = df_summary['median_auc'].values
    q1s = df_summary['q1_auc'].values
    q3s = df_summary['q3_auc'].values

    plt.figure(figsize=(8, 6))
    plt.plot(sizes_np, medians, '-o', label='Median AUC')
    plt.fill_between(sizes_np, q1s, q3s, alpha=0.2, label='IQR (25th-75th percentile)')

    reference_auc = medians[-1]
    plt.axhline(reference_auc, color='black', linestyle='--', linewidth=1.5, label='Max Median AUC')
    plt.axhline(0.975 * reference_auc, color='gray', linestyle='--', linewidth=1.5, label='97.5% of Max')
    plt.axhline(0.95 * reference_auc, color='lightgray', linestyle='--', linewidth=1.5, label='95% of Max')

    plt.xlabel("Number of Patients per Class")
    plt.ylabel("ROC AUC on Fixed Test Set")
    plt.title("XGBoost ROC AUC vs Training Data Size")
    plt.grid(True)
    plt.xticks(sizes_np)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder_name, "auc_vs_data_size.png"), dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
