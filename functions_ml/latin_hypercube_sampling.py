from scipy.stats import qmc


def generate_lhs_params(n_samples, random_state=None):
    sampler = qmc.LatinHypercube(d=7, seed=random_state)
    sample = sampler.random(n_samples)
    params = []
    for s in sample:
        max_depth = int(s[0] * (10 - 2 + 1)) + 2
        learning_rate = 0.01 + s[1] * (0.3 - 0.01)
        n_estimators = (int(s[2] * 10) + 1) * 50
        gamma = s[3] * 5
        min_child_weight = int(s[4] * 10) + 1
        subsample = 0.5 + s[5] * (1 - 0.5)
        colsample_bytree = 0.5 + s[6] * (1 - 0.5)
        param = {
            'max_depth': [max_depth],
            'learning_rate': [learning_rate],
            'n_estimators': [n_estimators],
            'gamma': [gamma],
            'min_child_weight': [min_child_weight],
            'subsample': [subsample],
            'colsample_bytree': [colsample_bytree]
        }
        params.append(param)
    return params
