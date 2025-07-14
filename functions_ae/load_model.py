import collections
import os
import glob

import torch


from functions_ae.model_classes import LSTMAutoencoder, LeNet5AutoencoderAvgPool


class SetupConfiguration(object):
    """input: args (Namespace)"""
    def __init__(self, model_class, latent_size, dropout):
        self.model_filters = int(32)
        self.model_layers = int(15)
        self.model_class = model_class
        self.latent_size = latent_size
        self.dropout = dropout

        # Initialize the model based on the model_class
        if self.model_class == 'LSTMAutoencoder':
            self.model = LSTMAutoencoder(latent_size=self.latent_size, dropout=self.dropout)
        elif self.model_class == 'LeNet5AutoencoderAvgPool':
            self.model = LeNet5AutoencoderAvgPool(latent_size=self.latent_size, dropout=self.dropout)
        else:
            raise ValueError('Not supporting model type')

    def view_model_class(self):
        return self.model_layers, self.model_filters

    def return_model(self):
        return self.model


def load_model(model_dir, config):
    """
    Load a trained model from the given directory.
    :param model_dir:str                        The directory where the model is saved
    :param config:argparse.Namespace            Contains all configuration parameters
    :return:
           torch.nn.Module                      The loaded model
    """

    model_config = SetupConfiguration(config.model_class, config.latent_size, config.dropout)
    model0 = model_config.return_model()

    try:
        model_path = glob.glob(os.path.join(model_dir, 'model_min_val_loss-*'))[-1]
    except IndexError:
        # Load final model if best model is not available
        model_path = os.path.join(model_dir, f'model_autoencoder_state_dict.pth')
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"No model found in directory: {model_dir}")

    # If GPU available, load model on GPU, else load on CPU
    loaded_model = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

    # If the model is already a state_dict, load it directly, if not, extract the state_dict
    if type(loaded_model) == collections.OrderedDict:
        loaded_state_dict = loaded_model
    else:
        loaded_state_dict = loaded_model.state_dict()

    # If model was trained using DataParallel, remove the module
    try:
        model0.load_state_dict(loaded_state_dict)
        print('no dataparallel module detected')
    except RuntimeError:
        print('runtime error occured: remove dataparallel module')
        new_state_dict = collections.OrderedDict()
        for k, v in loaded_state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model0.load_state_dict(new_state_dict)

    if torch.cuda.is_available():
        model = model0.cuda()
        print(f'Loaded model on GPU: {model_path}')
    else:
        model = model0.cpu()
        print(f'Loaded model on CPU: {model_path}')

    return model
