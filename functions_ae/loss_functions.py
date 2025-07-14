import torch


def rmse_loss(pred, true):
    return torch.sqrt(torch.mean((pred - true) ** 2))


def sample_rmse_loss(pred, true):
    rmse_per_sample = torch.sqrt(torch.mean((pred - true) ** 2, dim=1))
    return torch.mean(rmse_per_sample), rmse_per_sample


def sample_prd_loss(pred, true):
    # Calculate squared differences
    sum_squared_diff = torch.sum((true - pred) ** 2, dim=1)
    # Calculate the sum of squares of the true values for each sample
    sum_squared_true = torch.sum(true ** 2, dim=1)  # axis=1 to handle per-sample
    # Calculate PRMSD for each sample
    prmsd_per_sample = 100 * torch.sqrt(sum_squared_diff / sum_squared_true)
    return torch.mean(prmsd_per_sample), prmsd_per_sample
