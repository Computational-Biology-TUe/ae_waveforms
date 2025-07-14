import argparse
import torch
import torch.backends.cudnn
import numpy as np

from functions_ae import run_train, run_test
import config as cfg


def autoencoder(custom_args=None):
    # Parse arguments, these are the settings for the training or testing of the model with their description

    # General arguments:

    arg_parser = argparse.ArgumentParser(description='Settings for training or testing autoencoder models.')

    arg_parser.add_argument('-m', '--mode', dest='mode', type=str, default='train',
                            help='train, or test (inference) mode')

    arg_parser.add_argument('--dataset_dir', type=str, default=cfg.path_data,
                            help='Directory where dataset .npy binaries exists')

    arg_parser.add_argument('--results_dir', type=str, default='./results',
                            help='Base directory of the project.')

    arg_parser.add_argument('--seed', type=int, default=1,
                            help='Random seed for reproducibility')

    arg_parser.add_argument('--model_class', type=str, default='LSTMAutoencoder',
                            help='Model type to train or test')

    arg_parser.add_argument('--latent_size', type=int, default=10,
                            help='Latent size of the autoencoder')

    arg_parser.add_argument('--split_size_val', type=float, default=0.02,
                            help='Validation dataset split size percentage (split by case_id in chart table)')

    arg_parser.add_argument('--split_size_test', type=float, default=0.01,
                            help='Test dataset split size percentage (split by case_id in chart table)')

    arg_parser.add_argument('--nr_train_ids', type=int, default=0,
                            help='Number of training case_ids to use, set to 0 to use all available IDs')

    arg_parser.add_argument('--use_multiprocessing', type=bool, default=False,
                            help='read and use number of gpus')

    arg_parser.add_argument('--sync_to_neptune', type=bool, default=True,
                            help='turn on//off neptune')

    arg_parser.add_argument('--neptune_project', type=str, default=cfg.npt_project,
                            help='Neptune project path for logging your runs')

    arg_parser.add_argument('--neptune_api_token', type=str, default=cfg.npt_api_token,
                            help='Neptune api token')

    # Arguments used during training:

    arg_parser.add_argument('--batch_size', type=int, default=128,
                            # lower batch size if "RuntimeError: CUDA Out of memory"
                            help='Batch size for training')

    arg_parser.add_argument('--epochs', type=int, default=10,
                            help='Amount of epochs after which training is stopped')

    arg_parser.add_argument('--validation_interval', type=int, default=200,  # default was 10
                            help='Interval after how many batches the validation is performed during training')

    arg_parser.add_argument('--early_stopping_patience', type=int, default=2000,
                            help='How long to wait after last time validation loss improved before early stopping '
                                 'the training process')

    arg_parser.add_argument('--loss', type=str, default='mse',  # ! default was mape
                            help='Loss function')

    arg_parser.add_argument('--weight_decay', type=float, default=0,
                            help='weight decay (L2 penalty) for training')

    arg_parser.add_argument('--dropout', type=float, default=0.2,
                            help='Dropout rate used within the model')

    arg_parser.add_argument('--lr', type=float, default=0.001,
                            help='Learning rate for training')

    arg_parser.add_argument('--lr_scheduler_type', type=str, default="StepLR",
                            help='Learning rate schedule during training. '
                                 'Implemented options: StepLR, CosineAnnealing, CosineAnnealingWarmup')

    arg_parser.add_argument('--lr_scheduler_val_delay', type=int, default=0,
                            help='How many validation loss evaluations to wait after last time validation loss '
                                 'improved before calling the LR scheduler, set to 0 to turn off')

    arg_parser.add_argument('--lr_scheduler_caw_settings', type=float, nargs=3, default=[0.02, 10, 0.00001],
                            help='Settings of the CosineAnnealingWarmup learning rate scheduler. Structured as follows:'
                                 '[resets_per_epoch, initial_warmup_steps_val, lr_min]')

    arg_parser.add_argument('--optimizer_type', type=str, default='adam',
                            help='=Optimizer type for training')

    arg_parser.add_argument('--sgd_momentum', type=float, default=0.9,
                            help='Momentum for training if the stochastic gradient descent (SGD) optimizer is selected')

    arg_parser.add_argument('--ranger_momentum', type=float, default=0.95,
                            help='Momentum for training if the ranger optimizer is selected')

    arg_parser.add_argument('--ranger_threshold', type=int, default=5,
                            help='Threshold for training if the ranger optimizer is selected')

    arg_parser.add_argument('--train_shuffle', type=bool, default=True,
                            help='use shuffle for training')

    arg_parser.add_argument('--val_shuffle', type=bool, default=False,
                            help='use shuffle for validation')

    # Arguments used during testing:

    arg_parser.add_argument('--trained_model', type=str, default='AUT-231',
                            help='Model version to test')

    arg_parser.add_argument('--plotting', type=bool, default=True,
                            help='Plot the results')


    if custom_args is not None:
        args = arg_parser.parse_args(custom_args)
    else:
        args = arg_parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if str(args.mode).lower() == 'train':
        output_dir = run_train.main_train(args)
        print('Training done, results and trained model saved at: {}'.format(output_dir))

    elif str(args.mode).lower() in ('test', 'inference'):
        (output_file, (test_loss, rmse, rmse_samples, prd, prd_samples),
         (test_pred, test_true, test_c), latent_spaces) = run_test.main_test(args)
        return test_loss, rmse, rmse_samples, prd, prd_samples, test_pred, test_true, test_c


if __name__ == '__main__':

    autoencoder()
