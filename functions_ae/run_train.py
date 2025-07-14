import os
import gc
import time
import glob
import math
import csv
import logging
import numpy as np

import torch
import torch.distributed
from torch.utils.data import DataLoader

import neptune
from neptune_pytorch import NeptuneLogger

from functions_ae.load_data import Datasets, load_datasets, split_datasets
from functions_ae.model_classes import model_classes

from functions_ae import ranger_optimizer as ranger  # ranger optimization method (RAdam + Lookahead)
from functions_ae.loss_functions import rmse_loss


# Whether to use automatic mixed precision (AMP) for faster training
use_amp = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience. Also saves the best model.
    """

    def __init__(self, npt_logger, results_dir, patience=7, verbose=False):
        """
        Initialization of the EarlyStopping object
        :param npt_logger:NeptuneLogger
        :param results_dir:str
        :param patience:int                 How long to wait after last time validation loss improved.
        :param verbose:bool                 If True, prints a message for each validation loss improvement.
        """

        self.logger = logging.getLogger('EarlyStopping')  # For logging and printing

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.best_model = None
        self.best_model_filepath = None
        self.valid_loss_min = np.Inf
        self.npt_logger = npt_logger
        self.results_dir = results_dir

    def __call__(self, valid_loss, model):
        """
        Checks if validation loss has decreased, and if it has, saves the model. If not, increases a counter.
        If the counter reaches the patience, early stops the training process.
        """
        if valid_loss < self.valid_loss_min:
            self.save_checkpoint(valid_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def force_init_counter(self):
        self.counter = 0
        self.early_stop = False

    def save_checkpoint(self, valid_loss, model):
        """
        Saves model when validation loss decreases
        :param valid_loss:float         Current validation loss
        :param model:torch.nn.Module    Model to save
        """
        if self.verbose:
            self.logger.info(
                f"Validation loss decreased ({self.valid_loss_min:.6f} --> {valid_loss:.6f}).  Saving model ...")

        # Remove previous best model
        for del_models in glob.glob(os.path.join(self.results_dir, 'model_min_val_loss-*')):
            os.remove(del_models)

        # Save new best model with the lowest validation loss
        filepath = os.path.join(self.results_dir, f'model_min_val_loss-{np.around(valid_loss, 4)}.pth')
        filepath_sd = os.path.join(self.results_dir, f'model_min_val_loss-{np.around(valid_loss, 4)}_state_dict.pth')
        torch.save(model, filepath)
        torch.save(model.state_dict(), filepath_sd)
        try:  # Save the model checkpoint using Neptune logger, different versions of neptune have different methods
            if self.npt_logger is not None:
                self.npt_logger.save_checkpoint()
        except AttributeError:
            if self.npt_logger is not None:
                self.npt_logger.log_checkpoint()

        # Update self
        self.best_model = model
        self.best_model_filepath = filepath
        self.valid_loss_min = valid_loss


def validate(model_, valid_data_generator, loss_criterion):
    """Functions to validate the model"""
    valid_pred = []
    valid_true = []
    c_valid = []

    model_.eval()

    with torch.no_grad():
        for batch in valid_data_generator:
            # Extract the data from the batch, and add true labels to valid_true
            _batch_x, _batch_y, _batch_f, _batch_c = batch

            # Move model input to device (CPU/GPU)
            _batch_x = _batch_x.to(device).float()

            # Calculate model predictions
            with torch.cuda.amp.autocast(enabled=use_amp):
                if "Inception1DNet_NL_compact_dilated" in model_.__class__.__name__:
                    val_pred = model_([_batch_f, _batch_x])
                else:
                    val_pred = model_(_batch_x)

            valid_pred.append(val_pred)
            valid_true.append(_batch_y)
            c_valid.append(_batch_c)

    # Concatenate the lists to tensors
    valid_pred = torch.cat(valid_pred, dim=0).to(device)
    valid_true = torch.cat(valid_true, dim=0).to(device)
    c_valid = torch.cat(c_valid, dim=0)

    if "Autoencoder" not in model_.__class__.__name__:
        valid_true = torch.unsqueeze(valid_true, 1)

    # Calculate the validation loss
    val_loss = loss_criterion(valid_pred, valid_true)

    return val_loss.cpu(), valid_pred.cpu(), valid_true.cpu(), c_valid


def cosine_annealing_warmup_lr(optimizer, t_0, num_warmup_steps=200, min_lambda=0, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to eta_min, with several hard restarts, after a warmup period during which it
    increases linearly between 0 and the initial lr set in the optimizer.
    :param optimizer:torch.optim.Optimizer      The optimizer for which to schedule the learning rate.
    :param t_0:int                              Number of iterations (after warmup) for the first restart.
    :param num_warmup_steps:int                 The number of steps for the warmup phase.
    :param min_lambda:int                       Minimum lambda to multiply with the initial_lr. Default: 0.
    :param last_epoch:int                       The index of last epoch. Default: -1. When -1, sets lr as
                                                initial_lr*lambda*1 where lambda is the cosine annealing function.
    :return:
           torch.optim.lr_scheduler.LambdaLR    A PyTorch learning rate scheduler with the appropriate schedule.
    """

    def cosine_annealing_warmup_lambda(current_step):
        """
        Lambda multiplier for the cosine annealing scheduler with warmup, where lambda is between 1 and min_lambda.
        :param current_step:
        :return:
        """
        # Warmup with linear increase of lambda (only happens at the start of training)
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine annealing function (with hard restarts)
        else:
            t_cur = (current_step - num_warmup_steps) % t_0  # Number of steps since last restart
            max_lambda = 1.0
            if t_cur == 0:  # When (current_step - num_warmup_steps) == T_0, set lambda to min_lambda
                return max_lambda
            return min_lambda + 0.5 * (max_lambda - min_lambda) * (1.0 + math.cos(t_cur / t_0 * math.pi))

    return torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=cosine_annealing_warmup_lambda,
                                             last_epoch=last_epoch)


def train_with_lr_scheduler(model, train_data_generator, valid_data_generator, results_dir, cfg, npt_logger, npt_run):
    """
    Function for training the model with a learning rate scheduler
    :param model:torch.nn.Module                Initialized model
    :param train_data_generator:DataLoader      Generator for the training dataset returning batches of data
    :param valid_data_generator:DataLoader      Generator for the validation dataset returning batches of data
    :param results_dir:str                      Directory to store results for current training run: model, logs, etc.
    :param cfg:ArgumentParser                   The configurations of the training process (see main_ae.py for all configs)
    :param npt_logger:NeptuneLogger             Object used for logging info about the run to Neptune
    :param npt_run:NeptuneRun                   Object used for logging info about the run to Neptune
    :return:
           best_model:torch.nn.Module           Trained model with the lowest validation loss
           best_model_filepath:str              Filepath of the model with the lowest validation loss
           valid_loss_min:float                 Lowest validation loss
           train_loss_best:float                Training loss of the model with the lowest validation loss
    """
    logger = logging.getLogger('train_with_lr_scheduler')
    npt_logging = False
    if npt_run is not None:
        npt_logging = True

    # Select which training loss function (cfg.loss) to use
    if cfg.loss == 'mae':
        loss_criterion = torch.nn.SmoothL1Loss()
    elif cfg.loss == 'mse':
        loss_criterion = torch.nn.MSELoss()
    elif cfg.loss == 'rmse':
        loss_criterion = rmse_loss
    else:
        raise ValueError('Train loss error: not supported loss, current version only support mape, mae, or rmse')

    # Select the optimizer type to use
    if cfg.optimizer_type == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer_type == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                                    momentum=cfg.sgd_momentum)
    elif cfg.optimizer_type == 'ranger':
        optimizer = ranger.Ranger(model.parameters(), lr=cfg.lr, n_sma_threshold=cfg.ranger_n_threshold,
                                  betas=(cfg.ranger_momentum, 0.999), weight_decay=cfg.weight_decay)
    elif cfg.optimizer_type == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        if cfg.optimizer_type != 'adam':
            logger.warning(f'{cfg.optimizer_type} optimizer is not supported in pretrain: Use default Adam optimizer')
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Select the learning rate scheduler type to use
    if cfg.lr_scheduler_type == 'CosineAnnealing':
        resets_per_epoch = cfg.lr_scheduler_caw_settings[0]
        t0 = int(len(train_data_generator) / resets_per_epoch)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=t0, eta_min=0.0001)
    elif cfg.lr_scheduler_type == 'CosineAnnealingWarmup':
        resets_per_epoch = cfg.lr_scheduler_caw_settings[0]
        t0 = int(len(train_data_generator) / resets_per_epoch)
        warmup_steps = int(cfg.lr_scheduler_caw_settings[1] * cfg.validation_interval)
        eta_min = cfg.lr_scheduler_caw_settings[2]
        min_lambda = eta_min / cfg.lr
        scheduler = cosine_annealing_warmup_lr(optimizer=optimizer, t_0=t0, num_warmup_steps=warmup_steps,
                                               min_lambda=min_lambda)
    else:
        if cfg.lr_scheduler_type != 'StepLR':
            logger.warning(f'{cfg.lr_scheduler_type} scheduler is not supported in pretrain: Use default StepLR scheduler')
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.5)

    # Initialize the early stopping object and counter (also records the best model and best validation loss)
    early_stopping = EarlyStopping(patience=cfg.early_stopping_patience, verbose=True, npt_logger=npt_logger, results_dir=results_dir)
    early_stopping.force_init_counter()

    lr_valid_counter = 0
    best_model = None
    best_model_filepath = None
    valid_loss_min = np.Inf
    train_loss_best = np.Inf
    training_logs = []  # 'epoch', 'step', 'lr', 'train_loss'

    logger.info('Start model training')
    logger.info('--------------------')

    # If GPU available, load model on GPU, else load on CPU
    model = model.to(device)
    model.train()

    # Loop over the epochs
    for epoch in range(1, cfg.epochs + 1):
        start_time_epoch = time.time()
        start_time_valid_step = time.time()

        # Code for lstm model
        if model.__class__.__name__ in ('CNN_LSTM_inception', 'CNN_NL_LSTM_inception'):
            model.lstm_part.hidden = model.lstm_part.init_hidden(batch_size=cfg.batch_size)

        # Data logging to neptune
        if npt_logging:
            npt_run["epoch"].append(epoch)

        # Loop over the batches
        for step, batch in enumerate(train_data_generator):
            (batch_x, batch_y, batch_f, batch_c) = batch

            # Move the tensors to GPU
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()

            # Clear gradients of the model parameters (using optimizer)
            optimizer.zero_grad()

            # Train the model, forward pass with mixed precision (amp)
            with torch.cuda.amp.autocast(enabled=use_amp):
                if "Inception1DNet_NL_compact_dilated" in model.__class__.__name__:
                    output = model([batch_f, batch_x])
                else:
                    output = model(batch_x)
                loss = loss_criterion(output, batch_y)

                # Backpropagation and optimization
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # Data logging for .csv file with logging interval of 10 batches
            lr_current = optimizer.param_groups[0]['lr']
            train_loss_local = loss.item()
            if step % 10 == 0 and step != 0:
                training_logs.append([epoch, step, lr_current, train_loss_local])
            # Logging training results to Neptune (if enabled)
            if npt_logging:
                npt_run["Train Loss"].append(train_loss_local)
                npt_run["Batch number"].append(step)
                npt_run["Learning Rate"].append(lr_current)

            # Validate the model on the validation dataset every cfg.validation_interval batches
            if step % cfg.validation_interval == 0:

                valid_loss, _, _, _ = validate(model, valid_data_generator, loss_criterion)
                model.train()

                valid_loss_local = valid_loss.item()
                logger.info(f"[Epoch: {epoch}, step: {step}] / lr = {lr_current}")
                logger.info(f"\tTrain loss: {train_loss_local}, \t Validation loss: {valid_loss_local}")

                # Logging train and validation results .csv files
                with open(os.path.join(results_dir, "training_log.csv"), "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(training_logs)
                training_logs.clear()
                with open(os.path.join(results_dir, "training_log_validation.csv"), "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, step, lr_current, train_loss_local, valid_loss_local])
                # Logging validation results to Neptune (if enabled)
                if npt_logging:
                    npt_run["Validation loss"].append(valid_loss_local)

                elapsed_time_step = time.time() - start_time_valid_step
                logger.info("  Elapsed Time for {} step: {}s".format(cfg.validation_interval, elapsed_time_step))
                start_time_valid_step = time.time()

                # LR scheduler based on lr_scheduler_val_delay amount of validation loss increases
                if (cfg.lr_scheduler_type == "StepLR") and (cfg.lr_scheduler_val_delay > 0):
                    if valid_loss_local >= valid_loss_min:  # If validation loss does not decrease
                        lr_valid_counter += 1
                    if lr_valid_counter == cfg.lr_scheduler_val_delay:
                        scheduler.step()
                        lr_valid_counter = 0

                # Checking for early stopping and saving the best model if validation loss is lower than current best
                early_stopping(valid_loss=valid_loss_local, model=model)
                if valid_loss_local < valid_loss_min:
                    train_loss_best = train_loss_local
                valid_loss_min = early_stopping.valid_loss_min
                if early_stopping.early_stop:  # Break out of the step loop
                    logger.warning(f"Early Stopping @ epoch : {epoch} / step : {step}")
                    break

                torch.cuda.empty_cache()
                gc.collect()

            # Adjust the learning rate using the scheduler if necessary
            if cfg.lr_scheduler_type in ['CosineAnnealing', 'CosineAnnealingWarmup']:
                scheduler.step()

        best_model = early_stopping.best_model
        best_model_filepath = early_stopping.best_model_filepath
        elapsed_time_epoch = time.time() - start_time_epoch
        logger.info('End training epoch number {} from {} / at {}s'.format(epoch, cfg.epochs, elapsed_time_epoch))
        logger.info('Current best validation loss = {}'.format(valid_loss_min))
        logger.info('--------------------')
        logger.info(' ')
        if early_stopping.early_stop:
            break  # If early stopping was triggered, break out of the epoch loop

    del batch_x, batch_y, output

    return best_model, best_model_filepath, valid_loss_min, train_loss_best


def main_train(cfg):
    """
    Main function for training the model. Loads the data by calling load_datasets. Then splits the data into train,
    validation and test sets and creates generators for the train and validation datasets. Finally, creates network
    structure and calls train_with_lr_scheduler to start the training process.
    :param cfg:ArgumentParser      The configurations of the training process (see main_ae.py for all configs)
    :return:
           results_dir:str         The directory where the model and other results are saved
    """

    # %% ============================================ Initialize logging ==============================================

    # Set up the directory where the results will be saved
    if cfg.sync_to_neptune:
        tags = [cfg.model_class, f'latent_size={cfg.latent_size}', f'batch_size={cfg.batch_size}',
                f"lr={cfg.lr}", cfg.lr_scheduler_type, f"validation_interval={cfg.validation_interval}",
                f"early_stopping_patience={cfg.early_stopping_patience}", cfg.loss, cfg.optimizer_type,
                f"dropout={cfg.dropout}", f"weight_decay={cfg.weight_decay}", f"seed={cfg.seed}",
                f"dataset_dir={cfg.dataset_dir}", f"nr_train_ids={cfg.nr_train_ids}"]
        if cfg.lr_scheduler_type in ['CosineAnnealing', 'CosineAnnealingWarmup']:
            tags.append(f"lr_scheduler_caw_settings={cfg.lr_scheduler_caw_settings}")
        elif cfg.lr_scheduler_type == 'StepLR':
            tags.append(f"lr_scheduler_val_delay={cfg.lr_scheduler_val_delay}")
        npt_run = neptune.init_run(project=cfg.neptune_project, api_token=cfg.neptune_api_token,
                                   tags=tags, dependencies="infer")
        run_id = npt_run["sys/id"].fetch()
        results_dir = os.path.join(cfg.results_dir, f'networks_autoencoder', str(run_id))
    else:
        npt_run = None
        current_datetime = time.strftime("%y-%m-%d-%H%M")
        results_dir = os.path.join(cfg.results_dir, f'networks_autoencoder', current_datetime)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Set up logging of all prints in console to console.log file
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s  %(levelname)-8s  %(name)-32s \t| %(message)s',
                        datefmt='%H:%M:%S',
                        filename=os.path.join(results_dir, 'console.log'),
                        filemode='w+')
    # Define a Handler which writes INFO messages or higher to the console (sys.stderr)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # Set a format which is simpler for console use and tell the handler to use this format
    console.setFormatter(logging.Formatter('\033[0;37m%(asctime)s  %(levelname)-8s  %(name)-32s '
                                           '\t\033[0;2m| %(message)s\033[0m',
                                           datefmt='%H:%M:%S'))
    # Add the handler to the root logger
    logging.getLogger().addHandler(console)
    logger = logging.getLogger('main_train')

    # %% ========================================== Define the model structure =========================================

    # Initialize a model
    model_class = model_classes[cfg.model_class]
    model = model_class(latent_size=cfg.latent_size, dropout=cfg.dropout)

    # Save the model initialization to the results directory
    torch.save(model, os.path.join(results_dir, f'model_init.pth'))
    torch.save(model.state_dict(), os.path.join(results_dir, f'model_init_state_dict.pth'))

    if device.type == 'cuda':
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 0

    # If using multiple GPUs, use DistributedDataParallel
    if cfg.use_multiprocessing and num_gpus > 1:
        workers = num_gpus
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = "29500"
        torch.distributed.init_process_group(backend='gloo', world_size=num_gpus, rank=0)
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        workers = 0

    # %% ================== Load the data and split into train, validation, test (without KFold!) ======================

    start_time_data_build = time.time()
    logger.info(f"Start loading data from dataset version: {cfg.dataset_dir}")
    all_x, all_a, all_c = load_datasets(cfg.dataset_dir)
    logger.info(f"Data loading done. Length of Data: {len(all_x)} samples")

    indices_train, indices_val, indices_test = split_datasets(data_c=all_c, dataset_dir=cfg.dataset_dir,
                                                              results_dir=results_dir,
                                                              sampling_rate_val=cfg.split_size_val,
                                                              sampling_rate_test=cfg.split_size_test,
                                                              nr_train_ids=cfg.nr_train_ids,)

    if cfg.nr_train_ids > 0:
        cfg.epochs = int(np.ceil(228543 / (sum(indices_train) / cfg.batch_size)))

    # %% ======================================= Build datasets and generators =========================================
    # batch_size: How many samples per batch to load (default: 1).
    # shuffle: Set to True to have the data reshuffled at every epoch (default: False).
    # num_workers: How many subprocesses to use for data loading. 0 means that the data will be loaded in the main
    # process. (default: 0)
    params_train = {
        # run on multiple gpus
        'batch_size': cfg.batch_size,
        'shuffle': cfg.train_shuffle,
        'num_workers': workers,
        'pin_memory': False}
    params_val = {
        # run on multiple gpus
        'batch_size': cfg.batch_size,
        'shuffle': cfg.val_shuffle,
        'num_workers': workers,
        'pin_memory': False}

    logger.info(f"Available GPUs: {num_gpus}, Number of workers set to: {workers}")

    # Whether to swap the dimensions of the input data (batch_size * 1 * len_data) vs (batch_size * len_data * 1)
    swap_dimensions = True
    if "LSTM" in model.__class__.__name__:
        swap_dimensions = False

    logger.info("\tStart building train dataset generator")
    train_set = Datasets(all_x=all_x, all_a=all_a, all_c=all_c, flag=indices_train, swap_dim=swap_dimensions)
    train_data_generator = DataLoader(train_set, **params_train)

    logger.info("\tStart building validation dataset generator")
    valid_set = Datasets(all_x=all_x, all_a=all_a, all_c=all_c, flag=indices_val, swap_dim=swap_dimensions)
    valid_data_generator = DataLoader(valid_set, **params_val)

    logger.info(f"Data loading and generator building finished, "
                f"elapsed time: {time.strftime('%M:%S', time.localtime(time.time() - start_time_data_build))}")

    # %% =========================================== Start training ====================================================

    # Logging to Neptune
    if cfg.sync_to_neptune:
        npt_logger = NeptuneLogger(run=npt_run, model=model, log_model_diagram=False, log_gradients=False,
                                   log_parameters=False, log_freq=30)
    else:
        npt_logger = None

    # K-fold cross validation can be implemented here
    current_k_fold_step = 0
    start_time_kf = time.time()

    logger.info(f"Start training with learning rate scheduler: {cfg.lr_scheduler_type}")
    best_model, _, valid_loss_min, _ = train_with_lr_scheduler(
        model=model, train_data_generator=train_data_generator, valid_data_generator=valid_data_generator,
        results_dir=results_dir, cfg=cfg, npt_logger=npt_logger, npt_run=npt_run)

    logger.info('-------------------Train step done-------------------\n')
    logger.info(f"End KFold number {current_k_fold_step}, "
                f"elapsed time: {time.strftime('%H:%M:%S', time.localtime(time.time() - start_time_kf))}")
    logger.info(f"Best validation loss: {valid_loss_min}.\n")
    logger.info('===================================================================================\n')
    # K-fold cross validation can be implemented here

    logger.info('Training is finished.')
    logger.info(f"Model mean loss = {np.mean(np.array(valid_loss_min))}")
    torch.save(best_model, os.path.join(results_dir, f'model_autoencoder.pth'))
    torch.save(best_model.state_dict(), os.path.join(results_dir, f'model_autoencoder_state_dict.pth'))
    logger.info(f"Execution finished and saved at: {results_dir}")

    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    if cfg.sync_to_neptune:
        npt_run.stop()
    return results_dir

