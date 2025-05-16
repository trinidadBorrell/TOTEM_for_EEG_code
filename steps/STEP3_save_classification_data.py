import argparse
import numpy as np
import os
import pdb
import pickle
import torch
import torch.nn as nn

from data_provider.data_factory_vqvae_no_shuffle import data_provider_flexPath
from lib.models.revin import RevIN
from lib.models.vqvae import vqvae

from omegaconf import DictConfig, OmegaConf
import hydra

import logging

log = logging.getLogger(__name__)

@hydra.main(config_path="../conf", version_base="1.1")
def main(cfg: DictConfig) -> None:
    log.info("STEP3: Save classification data")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    log.info(f'Working directory {os.getcwd()}')

    args = cfg.preprocessing

    # random seed
    fix_seed = args.random_seed
    # random.seed(fix_seed)  # This isn't used
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print('Args in experiment:')
    print(args)

    if len(args.train_root_paths) == 0 and len(args.train_data_paths) == 0 and len(args.test_root_paths) == 0 and len(args.test_data_paths) == 0:
        ## Single session data extraction (train, val, test all from one file)
        print("Saving single file")
        Exp = ExtractData
        exp = Exp(args)  # set experiments
        exp.extract_data(save_data=True)
    else:
        ## Multi-session data extraction (train, val from train_data_paths) (test from test_data_paths)
        assert len(args.train_root_paths) == len(args.train_data_paths), "Train Root paths and data_paths lengths don't match! Cannot get multiple datas"
        assert len(args.test_root_paths) == len(args.test_data_paths), "Test Root paths and data_paths lengths don't match! Cannot get multiple datas"

        train_val_exps = []
        for i in range(len(args.train_root_paths)): 
            root_path_name = args.train_root_paths[i]
            data_path_name = args.train_data_paths[i]
            
            Exp = ExtractData
            exp = Exp(args)
            exp.extract_data(root_path_name=root_path_name, data_path_name=data_path_name, save_data=False)
            train_val_exps.append(exp)
        
        test_exps = []
        for i in range(len(args.test_root_paths)): 
            root_path_name = args.test_root_paths[i]
            data_path_name = args.test_data_paths[i]
            
            Exp = ExtractData
            exp = Exp(args)
            exp.extract_data(root_path_name=root_path_name, data_path_name=data_path_name, save_data=False)
            test_exps.append(exp)

        ## Create train val dictionaries concatenated across files 
        train_dict = {}
        val_dict = {} 
        for train_val_exp in train_val_exps: 
            ## Train data from train_val data paths will be used in the train
            for k, v in train_val_exp.train_data_dict.items(): 
                if k not in train_dict: 
                    train_dict[k] = v
                elif k != "codebook": # Only concatenate if not codebook
                    train_dict[k] = np.concatenate([train_dict[k], v], axis=0)
            ## Val data from train_val data paths will be used in the val
            for k, v in train_val_exp.val_data_dict.items(): 
                if k not in val_dict: 
                    val_dict[k] = v
                elif k != "codebook": # Only concatenate if not codebook
                    val_dict[k] = np.concatenate([val_dict[k], v], axis=0)
            
        ## Create test dictionaries concatenated across files 
        test_dict = {} 
        for test_exp in test_exps: 
            for k, v in test_exp.test_data_dict.items():
                if k not in test_dict: 
                    test_dict[k] = v
                elif k != "codebook": # Only concatenate if not codebook
                    test_dict[k] = np.concatenate([test_dict[k], v], axis=0)

        ## Save files 
        if len(train_val_exps) > 0: 
            save_files_classification(args.save_path, train_dict, mode="train", save_codebook=True)
            save_files_classification(args.save_path, val_dict, mode="val", save_codebook=False)
        if len(test_exps) > 0:
            save_files_classification(args.save_path, test_dict, mode="test", save_codebook=False)

class ExtractData:
    def __init__(self, args):
        self.args = args
        if not args.use_gpu or not torch.cuda.is_available():
            self.device = 'cpu'
        else:
            self.device = 'cuda:' + str(self.args.gpu)
    
        print(f"Using device: {self.device}")
        self.revin_layer_x = RevIN(num_features=self.args.enc_in, affine=False, subtract_last=False)
        self.revin_layer_y = RevIN(num_features=self.args.enc_in, affine=False, subtract_last=False)       
        
    # def _get_data(self, flag):
    #     data_set, data_loader = data_provider(self.args, flag)
    #     return data_set, data_loader
    
    def _get_data(self, root_path, data_path, flag):
        data_set, data_loader = data_provider_flexPath(args = self.args, root_path = root_path, data_path = data_path, flag=flag)
        return data_set, data_loader

    def one_loop_classification(self, loader, vqvae_model):
        x_original_all = []
        x_code_ids_all = []
        x_reverted_all = []
        x_labels_all = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            batch_x = batch_x[:, 1:, :] # making time go from 1001 to 1000 by dropping the first time step

            x_original_all.append(batch_x)
            batch_x = batch_x.float().to(self.device)

            # data going into revin should have dim:[bs x time x nvars]
            x_in_revin_space = self.revin_layer_x(batch_x, "norm")

            # expects time to be dim [bs x nvars x time]
            x_codes, x_code_ids, codebook = time2codes(x_in_revin_space.permute(0, 2, 1), self.args.compression_factor, vqvae_model.encoder, vqvae_model.vq)

            x_code_ids_all.append(np.array(x_code_ids.detach().cpu()))

            # expects code to be dim [bs x nvars x compressed_time]
            x_predictions_revin_space, x_predictions_original_space = codes2time(x_code_ids, codebook, self.args.compression_factor, vqvae_model.decoder, self.revin_layer_x)

            x_reverted_all.append(np.array(x_predictions_original_space.detach().cpu()))

            x_labels_all.append(batch_x_mark)


        x_original_arr = np.concatenate(x_original_all, axis=0)
        x_code_ids_all_arr = np.concatenate(x_code_ids_all, axis=0)
        x_reverted_all_arr = np.concatenate(x_reverted_all, axis=0)
        x_labels_all_arr = np.concatenate(x_labels_all, axis=0)

        data_dict = {}
        data_dict['x_original_arr'] = x_original_arr
        data_dict['x_code_ids_all_arr'] = np.swapaxes(x_code_ids_all_arr, 1, 2) # order will be [bs x compressed_time x sensors)
        data_dict['x_reverted_all_arr'] = x_reverted_all_arr
        data_dict['x_labels_all_arr'] = x_labels_all_arr
        data_dict['codebook'] = np.array(codebook.detach().cpu())

        print("x_original_arr shape", data_dict['x_original_arr'].shape)
        print("x_code_ids_all_arr shape", data_dict['x_code_ids_all_arr'].shape)
        print("x_reverted_all_arr shape", data_dict['x_reverted_all_arr'].shape)
        print("x_labels_all_arr shape", data_dict['x_labels_all_arr'].shape)
        print("codebook shape", data_dict['codebook'].shape)

        return data_dict

    def extract_data(self, root_path_name, data_path_name, save_data=True):
        """
        vqvae_model = torch.load(self.args.trained_vqvae_model_path)
        vqvae_model.to(self.device)
        vqvae_model.eval()
        """
    #    torch.serialization.add_safe_globals([vqvae])
    
        vqvae_model = torch.load(self.args.trained_vqvae_model_path, weights_only=False)
        vqvae_model.to(self.device)
        vqvae_model.eval()

        _, train_loader = self._get_data(root_path_name, data_path_name, flag='train')
        _, vali_loader = self._get_data(root_path_name, data_path_name, flag='val')
        _, test_loader = self._get_data(root_path_name, data_path_name, flag='test')


        print('CLASSIFYING')
        # These have dimension [bs, ntime, nvars]
        print('-------------TRAIN-------------')
        self.train_data_dict = self.one_loop_classification(train_loader, vqvae_model)
        if save_data: 
            save_files_classification(self.args.save_path, self.train_data_dict, 'train', save_codebook=True)

        print('-------------VAL-------------')
        self.val_data_dict = self.one_loop_classification(vali_loader, vqvae_model)
        if save_data: 
            save_files_classification(self.args.save_path, self.val_data_dict, 'val', save_codebook=False)

        print('-------------Test-------------')
        self.test_data_dict = self.one_loop_classification(test_loader, vqvae_model)
        if save_data: 
            save_files_classification(self.args.save_path, self.test_data_dict, 'test', save_codebook=False)


def save_files_classification(path, data_dict, mode, save_codebook, description=''):
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(os.path.join(path, mode + f'{"_"+description if description != "" else ""}' + '_x_original.npy'), data_dict['x_original_arr'])
    np.save(os.path.join(path, mode + f'{"_"+description if description != "" else ""}' +'_x_codes.npy'), data_dict['x_code_ids_all_arr'])
    np.save(os.path.join(path, mode + f'{"_"+description if description != "" else ""}' + '_x_reverted.npy'), data_dict['x_reverted_all_arr'])
    np.save(os.path.join(path, mode + f'{"_"+description if description != "" else ""}' + '_x_labels.npy'), data_dict['x_labels_all_arr'])

    if save_codebook:
        np.save(os.path.join(path, description + 'codebook.npy'), data_dict['codebook'])




def time2codes(revin_data, compression_factor, vqvae_encoder, vqvae_quantizer):
    '''
    Args:
        revin_data: [bs x nvars x pred_len or seq_len]
        compression_factor: int
        vqvae_model: trained vqvae model
        use_grad: bool, if True use gradient, if False don't use gradients

    Returns:
        codes: [bs, nvars, code_dim, compressed_time]
        code_ids: [bs, nvars, compressed_time]
        embedding_weight: [num_code_words, code_dim]

    Helpful VQVAE Comments:
        # Into the vqvae encoder: batch.shape: [bs x seq_len] i.e. torch.Size([256, 12])
        # into the quantizer: z.shape: [bs x code_dim x (seq_len/compresion_factor)] i.e. torch.Size([256, 64, 3])
        # into the vqvae decoder: quantized.shape: [bs x code_dim x (seq_len/compresion_factor)] i.e. torch.Size([256, 64, 3])
        # out of the vqvae decoder: data_recon.shape: [bs x seq_len] i.e. torch.Size([256, 12])
    '''

    bs = revin_data.shape[0]
    nvar = revin_data.shape[1]
    T = revin_data.shape[2]  # this can be either the prediction length or the sequence length
    compressed_time = int(T / compression_factor)  # this can be the compressed time of either the prediction length or the sequence length

    with torch.no_grad():
        flat_revin = revin_data.reshape(-1, T)  # flat_y: [bs * nvars, T]
        latent = vqvae_encoder(flat_revin.to(torch.float), compression_factor)  # latent_y: [bs * nvars, code_dim, compressed_time]
        vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = vqvae_quantizer(latent)  # quantized: [bs * nvars, code_dim, compressed_time]
        code_dim = quantized.shape[-2]
        codes = quantized.reshape(bs, nvar, code_dim,
                                  compressed_time)  # codes: [bs, nvars, code_dim, compressed_time]
        code_ids = encoding_indices.view(bs, nvar, compressed_time)  # code_ids: [bs, nvars, compressed_time]

    return codes, code_ids, embedding_weight


def codes2time(code_ids, codebook, compression_factor, vqvae_decoder, revin_layer):
    '''
    Args:
        code_ids: [bs x nvars x compressed_pred_len]
        codebook: [num_code_words, code_dim]
        compression_factor: int
        vqvae_model: trained vqvae model
        use_grad: bool, if True use gradient, if False don't use gradients
        x_or_y: if 'x' use revin_denorm_x if 'y' use revin_denorm_y
    Returns:
        predictions_revin_space: [bs x original_time_len x nvars]
        predictions_original_space: [bs x original_time_len x nvars]
    '''
    # print('CHECK in codes2time - should be TRUE:', vqvae_decoder.training)
    bs = code_ids.shape[0]
    nvars = code_ids.shape[1]
    compressed_len = code_ids.shape[2]
    num_code_words = codebook.shape[0]
    code_dim = codebook.shape[1]
    device = code_ids.device
    input_shape = (bs * nvars, compressed_len, code_dim)

    with torch.no_grad():
        # scatter the label with the codebook
        one_hot_encodings = torch.zeros(int(bs * nvars * compressed_len), num_code_words, device=device)  # one_hot_encodings: [bs x nvars x compressed_pred_len, num_codes]
        one_hot_encodings.scatter_(1, code_ids.reshape(-1, 1).to(device),1)  # one_hot_encodings: [bs x nvars x compressed_pred_len, num_codes]
        quantized = torch.matmul(one_hot_encodings, torch.tensor(codebook)).view(input_shape)  # quantized: [bs * nvars, compressed_pred_len, code_dim]
        quantized_swaped = torch.swapaxes(quantized, 1,2)  # quantized_swaped: [bs * nvars, code_dim, compressed_pred_len]
        prediction_recon = vqvae_decoder(quantized_swaped.to(device), compression_factor)  # prediction_recon: [bs * nvars, pred_len]
        prediction_recon_reshaped = prediction_recon.reshape(bs, nvars, prediction_recon.shape[-1])  # prediction_recon_reshaped: [bs x nvars x pred_len]
        predictions_revin_space = torch.swapaxes(prediction_recon_reshaped, 1,2)  # prediction_recon_nvars_last: [bs x pred_len x nvars]
        predictions_original_space = revin_layer(predictions_revin_space, 'denorm')  # predictions:[bs x pred_len x nvars]

    return predictions_revin_space, predictions_original_space


if __name__ == '__main__':
    main()