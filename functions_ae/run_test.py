import os
import glob
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from functions_ae.load_model import load_model  # Contains the definition of the deep learning model.
from functions_ae.load_data import load_datasets, Datasets  # Contains the definition of the dataset class.
from functions_ae.custom_plots import plot_rmse_distribution  # Creates and saves plots.
from functions_ae.loss_functions import rmse_loss, sample_rmse_loss, sample_prd_loss


use_amp = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main_test(config):
    """
    Main function for testing the model. Loads the data, runs through the test set and saves the predictions
    to .npy files.
    :param config:argparse.Namespace            Contains all configuration parameters
    """

    # %% ================================================== Setup ======================================================
    results_dir = os.path.join(config.results_dir, f'test_autoencoder', config.trained_model)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_file = os.path.join(str(results_dir), 'np_waves_test_pred.npy')

    # Select which training loss function (loss) to use
    if config.loss == 'mae':
        criterion = torch.nn.SmoothL1Loss()
    elif config.loss == 'mse':
        criterion = torch.nn.MSELoss()
    elif config.loss == 'rmse':
        criterion = rmse_loss
    else:
        raise ValueError('Pre-train loss error: not supported loss, current version only support mape or mae')

    # %% ============================================== Load the model =================================================
    model_dir = os.path.join(config.results_dir, f'networks_autoencoder', config.trained_model)

    # Load final model
    print(f"Load model {config.trained_model}")
    model = load_model(model_dir, config)

    # Set model to evaluation mode (turns out dropout and batch normalization)
    model = model.eval()

    # %% ====================== Load the data, split off test data, build dataset and generator ========================
    # Load dataset and test indices
    print("Start loading data")
    all_x, all_a, all_c = load_datasets(config.dataset_dir)
    indices_test_file = glob.glob(os.path.join(str(model_dir), 'indices_test_*.npy'))[0]
    indices_test = np.load(indices_test_file)


    # Create the test dataset generator
    params_test = {'batch_size': config.batch_size, 'shuffle': False, 'num_workers': 0, 'pin_memory': False}

    swap_dimensions = True
    if "LSTM" in model.__class__.__name__:
        swap_dimensions = False

    test_set = Datasets(all_x=all_x, all_a=all_a, all_c=all_c,
                        flag=indices_test, swap_dim=swap_dimensions)
    test_data_generator = DataLoader(test_set, **params_test)
    print(f"Data loading from autoencoder done. Length of Data: {len(all_c)} samples")

    # %% =========================================== Start testing =====================================================
    test_pred = []
    test_true = []
    test_c = []
    latent_spaces = []

    with torch.no_grad():
        with tqdm(total=sum(indices_test)) as pbar:  # define progress bar
            pbar.set_description('Making predictions for test data...')
            for step, batch in enumerate(test_data_generator):
                [batch_x, _, _, batch_c] = batch

                # Move x data to GPU
                batch_x = batch_x.to(device).float()

                # Make predictions and add to true_pred array
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(batch_x)
                    latent_space = model.encode(batch_x)

                test_pred.append(pred)
                test_true.append(batch_x)
                test_c.append(batch_c)
                latent_spaces.append(latent_space)

                pbar.update(len(batch_x))

    test_true = torch.cat(test_true, dim=0)
    test_pred = torch.cat(test_pred, dim=0)
    test_c = torch.cat(test_c, dim=0)
    latent_spaces = torch.cat(latent_spaces, dim=0)

    # Write predictions to .npy file
    if swap_dimensions:
        np.save(results_file, test_pred.squeeze(1).cpu().numpy())
        np.save(os.path.join(str(results_dir), 'np_latent_spaces.npy'), latent_spaces.squeeze(2).cpu().numpy())
    else:
        np.save(results_file, test_pred.squeeze(2).cpu().numpy())
        np.save(os.path.join(str(results_dir), 'np_latent_spaces.npy'), latent_spaces.cpu().numpy())

    # %% ==================================== Calculate test performance and plots =====================================
    test_loss = criterion(test_pred, test_true)
    print(f"Global test {config.loss.upper()} loss: {test_loss:.6f}")

    # Calculate and print model performance metrics
    # Sample mean RMSE
    rmse_sample_avg, rmse_per_sample = sample_rmse_loss(test_pred, test_true)
    print(f"Sample wise average test RMSE loss: {rmse_sample_avg:.6f}")
    # Sample mean Percentage RMS Difference (PRD)
    prd_sample_avg, prd_per_sample = sample_prd_loss(test_pred, test_true)
    print(f"Sample wise average test PRD loss: {prd_sample_avg:.6f}")

    if config.plotting:
        plot_rmse_distribution(rmse_sample_avg.item(), rmse_per_sample.squeeze(1).cpu().numpy(), str(results_dir))

    return (results_file,
            (test_loss.item(), rmse_sample_avg.item(), rmse_per_sample.squeeze(1).cpu().numpy(),
             prd_sample_avg.item(), prd_per_sample.squeeze(1).cpu().numpy()),
            (test_pred, test_true, test_c), latent_spaces)
