import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
import xgboost as xgb
from dotenv import load_dotenv

from functions_ml.latin_hypercube_sampling import generate_lhs_params


auc_records = []  # List of dicts: each row will be one AUC value with size and seed
summary_records = []  # For summary stats per size


def plot_individual_feature_rocs(X_train, X_val, y_train, y_val, best_params):
    # Define the feature names in the same order as the features in X
    feature_names = [
        "P Duration",
        "PR Interval",
        "QRS Complex",
        "ST Interval",
        "ST Segment",
        #"T Duration",
        "P Prominence",
        "Q Prominence",
        "R Prominence",
        "S Prominence",
        "T Prominence"
    ]
    
    plt.figure(figsize=(8, 6))
    
    # Plot ROC for each individual feature
    for i in range(X_train.shape[1]):
        model = xgb.XGBClassifier(**best_params, eval_metric='logloss', random_state=1)
        # Train on the individual feature (reshape to 2D array)
        model.fit(X_train[:, i].reshape(-1, 1), y_train)
        y_pred_proba = model.predict_proba(X_val[:, i].reshape(-1, 1))[:, 1]
        
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        auc_score = roc_auc_score(y_val, y_pred_proba)
        # Use feature name for the label if available
        if i < len(feature_names):
            label = f'{feature_names[i]} (AUC = {auc_score:.2f})'
        else:
            label = f'Feature {i+1} (AUC = {auc_score:.2f})'
        plt.plot(fpr, tpr, lw=1, label=label)
    
    # Plot ROC for the combined model (using all features)
    full_model = xgb.XGBClassifier(**best_params, eval_metric='logloss', random_state=1)
    full_model.fit(X_train, y_train)
    y_pred_proba_full = full_model.predict_proba(X_val)[:, 1]
    fpr_full, tpr_full, _ = roc_curve(y_val, y_pred_proba_full)
    auc_score_full = roc_auc_score(y_val, y_pred_proba_full)
    plt.plot(fpr_full, tpr_full, color='black', lw=2, label=f'Combined Model (AUC = {auc_score_full:.2f})')
    
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC-ROC Curves for Individual Features and Combined Model')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def evaluate_combined_model_auc(X, y, best_params, seed):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    model = xgb.XGBClassifier(**best_params, eval_metric='logloss', random_state=seed)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, y_pred_proba)
    return auc_score


def main():
    import pandas as pd

    filepath_env = 'vars.env'
    load_dotenv(filepath_env)
    path_data = os.environ['DATA_FOLDER_MACHINE_LEARNING_PATH']

    control_all = np.load(os.path.join(path_data, "control_features14500.npy"))

    # Load hyper_all with filename tracked
    hyper_filename = "hyposodium_features12000.npy"
    hyper_all_path = os.path.join(path_data, hyper_filename)
    hyper_all = np.load(hyper_all_path)
    file_stem = os.path.splitext(hyper_filename)[0]

    # Ensure results directory exists
    results_dir = os.path.join(path_data, "results")
    os.makedirs(results_dir, exist_ok=True)

    sizes = [100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 6000, 10000, 11500]# 14000]
    all_medians = []
    all_q1s = []
    all_q3s = []
    auc_records = []  # for storing individual AUCs
    summary_records = []  # for storing median/q1/q3 by size

    # Fixed test set: 250 control + 250 hyper
    test_size = 250
    control_all = control_all[:12000] #change to match the size of the sick group
    control_test = control_all[-test_size:]
    hyper_test = hyper_all[-test_size:]
    X_test = np.concatenate([control_test, hyper_test], axis=0)
    y_test = np.concatenate([np.zeros(test_size), np.ones(test_size)])
    X_test = np.delete(X_test, 5, axis=1)

    # Remaining for training
    control_train_all = control_all[:-test_size]
    hyper_train_all = hyper_all[:-test_size]

    # Generate fixed best hyperparameters on the largest training data
    lhs_params = generate_lhs_params(1000, random_state=1)
    model = xgb.XGBClassifier(eval_metric='logloss', random_state=1)

    control_max = control_train_all[:6000]
    hyper_max = hyper_train_all[:6000]
    X_full = np.concatenate([control_max, hyper_max], axis=0)
    y_full = np.concatenate([np.zeros(control_max.shape[0]), np.ones(hyper_max.shape[0])])
    X_full = np.delete(X_full, 5, axis=1)

    X_train_full, _, y_train_full, _ = train_test_split(
        X_full, y_full, test_size=0.2, random_state=1, stratify=y_full
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=lhs_params,
        scoring='roc_auc',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train_full, y_train_full)
    best_params = grid_search.best_params_

    # Evaluate each size
    for size in sizes:
        aucs = []
        for seed in range(25):
            rng = np.random.default_rng(seed + 42)
            idx_control = rng.choice(len(control_train_all), size, replace=False)
            idx_hyper = rng.choice(len(hyper_train_all), size, replace=False)
            control_sample = control_train_all[idx_control]
            hyper_sample = hyper_train_all[idx_hyper]
            X_train = np.concatenate([control_sample, hyper_sample], axis=0)
            y_train = np.concatenate([np.zeros(size), np.ones(size)])
            X_train = np.delete(X_train, 5, axis=1)

            model = xgb.XGBClassifier(**best_params, eval_metric='logloss', random_state=seed + 42)
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            aucs.append(auc)

            # Store each individual AUC
            auc_records.append({
                "size": size,
                "seed": seed,
                "auc": auc
            })

        median_auc = np.median(aucs)
        q1 = np.percentile(aucs, 25)
        q3 = np.percentile(aucs, 75)

        all_medians.append(median_auc)
        all_q1s.append(q1)
        all_q3s.append(q3)

        summary_records.append({
            "size": size,
            "median_auc": median_auc,
            "q1_auc": q1,
            "q3_auc": q3
        })

        print(f"Size: {size}, Median AUC: {median_auc:.4f}, IQR: ({q1:.4f}, {q3:.4f})")

    # Plot with IQR as shaded area
    plt.figure(figsize=(8, 6))
    sizes_np = np.array(sizes)
    all_medians = np.array(all_medians)
    all_q1s = np.array(all_q1s)
    all_q3s = np.array(all_q3s)

    plt.plot(sizes_np, all_medians, '-o', label='Median AUC')
    plt.fill_between(sizes_np, all_q1s, all_q3s, alpha=0.2, label='IQR (25th–75th percentile)')

    reference_auc = all_medians[sizes.index(2000)]
    plt.axhline(reference_auc, color='black', linestyle='--', linewidth=1.5, label='100% of Max AUC')
    plt.axhline(0.975 * reference_auc, color='gray', linestyle='--', linewidth=1.5, label='97.5% of Max AUC')
    plt.axhline(0.95 * reference_auc, color='lightgray', linestyle='--', linewidth=1.5, label='95% of Max AUC')

    plt.xlabel('Number of Patients per Class')
    plt.ylabel('ROC AUC on Fixed Test Set')
    plt.grid(True)
    plt.xticks(sizes_np, rotation=45)
    plt.legend(loc='lower right')

    # Save PNG
    plt.savefig(os.path.join(results_dir, f"auc_vs_data_size_{os.path.basename(hyper_all_path).replace('.npy', '')}.png"), dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

    # Save results
    df_auc = pd.DataFrame(auc_records)
    df_auc.to_csv(os.path.join(results_dir, f"{file_stem}_individual_aucs.csv"), index=False)

    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(os.path.join(results_dir, f"{file_stem}_summary_auc_by_size.csv"), index=False)


if __name__ == '__main__':
    main()
