import os
import numpy as np
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score, precision_score

from functions_ml.latin_hypercube_sampling import generate_lhs_params


def main():
    # Load environment path
    filepath_env = 'vars.env'
    load_dotenv(filepath_env)
    path_data = os.environ['DATA_FOLDER_HYBRID_LEARNING_PATH']

    # Define data file names
    control_file = "controls.csv"
    sick_file = "hypopotassium.csv"

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

    # Use only group size 6000
    size = 6000

    # Generate LHS params for grid search
    lhs_params = generate_lhs_params(1000, random_state=1)
    model = xgb.XGBClassifier(eval_metric='logloss', random_state=1)

    max_train_size = min(size, len(control_train_all), len(sick_train_all))
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

    aucs = []
    ppvs = []
    threshold = 0.5

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

        # Convert probabilities to binary prediction at threshold 0.5
        y_pred = (y_pred_proba >= threshold).astype(int)
        ppv = precision_score(y_test, y_pred)
        ppvs.append(ppv)

    median_auc = np.median(aucs)
    q1_auc = np.percentile(aucs, 25)
    q3_auc = np.percentile(aucs, 75)

    median_ppv = np.median(ppvs)
    q1_ppv = np.percentile(ppvs, 25)
    q3_ppv = np.percentile(ppvs, 75)

    print(f"Size: {size}")
    print(f"Median AUC: {median_auc:.4f}, IQR: ({q1_auc:.4f}, {q3_auc:.4f})")
    print(f"Median PPV @ threshold {threshold}: {median_ppv:.4f}, IQR: ({q1_ppv:.4f}, {q3_ppv:.4f})")

if __name__ == "__main__":
    main()
