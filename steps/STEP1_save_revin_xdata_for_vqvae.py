import argparse
import numpy as np
import os
import pdb
import random
import torch

from data_provider.data_factory_vqvae_no_shuffle import data_provider_flexPath
from lib.models.revin import RevIN

from omegaconf import DictConfig, OmegaConf
import hydra

import logging

log = logging.getLogger(__name__)

@hydra.main(config_path="../conf", version_base="1.1")
def main(cfg: DictConfig) -> None:
    log.info("STEP1: Save RevIN x data for training VQ-VAE")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    log.info(f'Working directory {os.getcwd()}')

    args = cfg.preprocessing

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    ## Only allow gpu usage if cuda is available
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    ## print 
    print('Args in experiment:')
    print(args)

    Exp = ExtractData
    exp = Exp(args)  # set experiments
    exp.extract_data()



class ExtractData:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda:' + str(self.args.gpu)
        self.revin_layer_x = RevIN(num_features=self.args.enc_in, affine=False, subtract_last=False)
        self.revin_layer_y = RevIN(num_features=self.args.enc_in, affine=False, subtract_last=False)

    def _get_data(self, root_path, data_path, flag):
        data_set, data_loader = data_provider_flexPath(args = self.args, root_path = root_path, data_path = data_path, flag=flag)
        return data_set, data_loader

    def one_loop(self, loader):
        x_in_revin_space = []
        y_in_revin_space = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            # data going into revin should have dim:[bs x seq_len x nvars]
            x_in_revin_space.append(np.array(self.revin_layer_x(batch_x, "norm").detach().cpu()))
            y_in_revin_space.append(np.array(self.revin_layer_y(batch_y, "norm").detach().cpu()))

        x_in_revin_space_arr = np.concatenate(x_in_revin_space, axis=0)
        y_in_revin_space_arr = np.concatenate(y_in_revin_space, axis=0)

        print(x_in_revin_space_arr.shape, y_in_revin_space_arr.shape)
        return x_in_revin_space_arr, y_in_revin_space_arr

    def extract_data(self):

        x_train_arr_list = []  # Initialize an empty list to hold x_train_arr from each iteration
        x_val_arr_list = []
        x_test_arr_list = []

        for root_path_name, data_path_name in zip(self.args.root_paths, self.args.data_paths):          
            _, train_loader = self._get_data(root_path_name, data_path_name, flag='train')
            _, vali_loader = self._get_data(root_path_name, data_path_name, flag='val')
            _, test_loader = self._get_data(root_path_name, data_path_name, flag='test')

            print('got loaders starting revin')

            # These have dimension [bs, ntime, nvars]
            x_train_in_revin_space_arr, y_train_in_revin_space_arr = self.one_loop(train_loader)
            print('starting val')
            x_val_in_revin_space_arr, y_val_in_revin_space_arr = self.one_loop(vali_loader)
            print('starting test')
            x_test_in_revin_space_arr, y_test_in_revin_space_arr = self.one_loop(test_loader)

            print('Flattening Sensors Out')
            if self.args.seq_len != self.args.pred_len:
                print('HoUstoN wE haVE A prOblEm')
                pdb.set_trace()
            else:
                # These have dimension [bs x nvars, ntime]
                x_train_arr = np.swapaxes(x_train_in_revin_space_arr, 1,2).reshape((-1, self.args.pred_len))
                x_val_arr = np.swapaxes(x_val_in_revin_space_arr, 1, 2).reshape((-1, self.args.pred_len))
                x_test_arr = np.swapaxes(x_test_in_revin_space_arr, 1, 2).reshape((-1, self.args.pred_len))
                print("Final output")
                print(x_train_arr.shape, x_val_arr.shape, x_test_arr.shape)

                x_train_arr_list.append(x_train_arr)
                x_val_arr_list.append(x_val_arr)
                x_test_arr_list.append(x_test_arr)

                print(len(x_train_arr_list), len(x_val_arr_list), len(x_test_arr_list))


        # Concatenate all arrays into a single array
        x_train_arr = np.concatenate(x_train_arr_list, axis=0)
        x_val_arr = np.concatenate(x_val_arr_list, axis=0)
        x_test_arr = np.concatenate(x_test_arr_list, axis=0)

        # Now, x_train_arr contains the concatenated array from all iterations
        print("Final x_train_arr shape:", x_train_arr.shape)
        print("Final x_val_arr shape:", x_val_arr.shape)
        print("Final x_test_arr shape:", x_test_arr.shape)

        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)
        np.save(self.args.save_path + '/train_data_x.npy', x_train_arr)
        np.save(self.args.save_path + '/val_data_x.npy', x_val_arr)
        np.save(self.args.save_path + '/test_data_x.npy', x_test_arr)


if __name__ == '__main__':
    main()
