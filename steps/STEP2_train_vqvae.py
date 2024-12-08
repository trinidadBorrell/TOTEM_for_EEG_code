import argparse
import copy

import comet_ml
import json
import numpy as np
import os
import pdb
import random
import time
import torch

from lib.models import get_model_class
from time import gmtime, strftime

from omegaconf import DictConfig, OmegaConf
import hydra

import logging

log = logging.getLogger(__name__)

@hydra.main(config_path="../conf", version_base="1.1")
def main(cfg: DictConfig) -> None:
    log.info("STEP2: Train VQ-VAE")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    log.info(f'Working directory {os.getcwd()}')

    exp_cfg = cfg.exp
    logging_cfg = cfg.logging
    save_dir = exp_cfg.save_dir 

    # (6) Setting up the comet logger
    if logging_cfg.comet:
        comet_config = logging_cfg.comet
        # Create an experiment with your api key
        comet_logger = comet_ml.Experiment(
            api_key=comet_config['api_key'],
            project_name=comet_config['project_name'],
            workspace=comet_config['workspace'],
        )
        comet_logger.add_tag(comet_config.comet_tag)
        comet_logger.set_name(comet_config.comet_name)
    else:
        print('PROBLEM: not saving to comet')
        comet_logger = None
        pdb.set_trace()

    # (7) Set up GPU / CPU
    if torch.cuda.is_available() and exp_cfg.gpu_id >= 0:
        assert exp_cfg.gpu_id < torch.cuda.device_count()  # sanity check
        device = 'cuda:{:d}'.format(exp_cfg.gpu_id)
    else:
        device = 'cpu'

    # (8) Where to init data for training (cpu or gpu) -->  will be trained wherever args.model_init_num_gpus says
    if exp_cfg.data_init_cpu_or_gpu == 'gpu':
        data_init_loc = device  # we do this so that data_init_loc will have the correct cuda:X if gpu
    else:
        data_init_loc = 'cpu'

    # (9) call runner
    runner(device, exp_cfg, save_dir, comet_logger, data_init_loc)


def runner(device, config, save_dir, logger, data_init_loc):
    # (1) Create/overwrite checkpoints folder and results folder
    # Create model checkpoints folder
    if os.path.exists(os.path.join(save_dir, 'checkpoints')):
        print("Checkpoint Directory:", os.path.join(save_dir, 'checkpoints'))
        print('Checkpoint Directory Already Exists - if continue will overwrite files inside. Press c to continue.')
        # pdb.set_trace()
    else:
        os.makedirs(os.path.join(save_dir, 'checkpoints'))


    # (3) log the config parameters to comet_ml
    logger.log_parameters(config)

    # (4) Run start training
    vqvae_config, summary = start_training(device=device, vqvae_config=config['vqvae_config'], save_dir=save_dir,
                                           logger=logger, data_init_loc=data_init_loc)


def start_training(device, vqvae_config, save_dir, logger, data_init_loc):
    # (1) Create summary dictionary
    summary = {}

    # (2) Sample and fix a random seed if not set
    if 'general_seed' not in vqvae_config:
        vqvae_config['seed'] = random.randint(0, 9999)

    general_seed = vqvae_config['general_seed']
    summary['general_seed'] = general_seed
    torch.manual_seed(general_seed)
    random.seed(general_seed)
    np.random.seed(general_seed)
    # if use another random library need to set that seed here too

    torch.backends.cudnn.deterministic = True  # makes cuDNN to only have to only use determinisitic convolution algs.

    # summary['dataset'] = datamodule.summary  # add dataset name to the summary
    summary['data initialization location'] = data_init_loc
    summary['device'] = device  # add the cpu/gpu to the summary

    # (4) Setup model
    model_class = get_model_class(vqvae_config['model_name'].lower())
    model = model_class(vqvae_config)  # Initialize model

    # Uncomment if want to know the total number of trainable parameters
    print('Total # trainable parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if vqvae_config['pretrained']:
        # pretrained needs to be the path to the trained model if you want it to load
        model = torch.load(vqvae_config['pretrained'])  # Get saved pytorch model.
    summary['vqvae_config'] = vqvae_config  # add the model information to the summary

    # (5) Start training the model
    start_time = time.time()
    model = train_model(model, device, vqvae_config, save_dir, logger)

    # (6) Once the model has trained - Save full pytorch model
    torch.save(model, os.path.join(save_dir, 'checkpoints/final_model.pth'))
    # logger.log_model("Prog_Model", os.path.join(save_dir, 'checkpoints'))  # save model information to comet

    # (7) Save and return
    summary['total_time'] = round(time.time() - start_time, 3)
    return vqvae_config, summary


def train_model(model, device, vqvae_config, save_dir, logger):
    # Set the optimizer
    optimizer = model.configure_optimizers(lr=vqvae_config['learning_rate'])
    # If want a learning rate scheduler instead uncomment this
    # lr_lambda = lambda epoch: max(1e-6, 0.9**(int(epoch/250))*train_config['learning_rate'])
    # lr_scheduler = LambdaLR(optimizer, lr_lambda)

    # Setup model (send to device, set to train)
    model.to(device)
    start_time = time.time()

    print('BATCHSIZE:', vqvae_config["batch_size"])
    train_loader, vali_loader, test_loader = create_datloaders(batchsize=vqvae_config["batch_size"], dataset=vqvae_config["dataset"], base_path=vqvae_config.get("dataset_base_path", "/data/sabera/mts_v2_datasets/pipeline/revin_data_to_train_vqvae"))

    # do + 0.5 to ciel it
    for epoch in range(int(vqvae_config['num_epochs'])):
        model.train()
        for i, (batch_x) in enumerate(train_loader):
            tensor_all_data_in_batch = torch.tensor(batch_x, dtype=torch.float, device=device)

            loss, vq_loss, recon_error, x_recon, perplexity, embedding_weight, encoding_indices, encodings = \
                model.shared_eval(tensor_all_data_in_batch, optimizer, 'train', comet_logger=logger)

        if epoch % 10 == 0:
            with (torch.no_grad()):
                model.eval()
                for i, (batch_x) in enumerate(vali_loader):
                    tensor_all_data_in_batch = torch.tensor(batch_x, dtype=torch.float, device=device)

                    val_loss, val_vq_loss, val_recon_error, val_x_recon, val_perplexity, val_embedding_weight, \
                        val_encoding_indices, val_encodings = \
                        model.shared_eval(tensor_all_data_in_batch, optimizer, 'val', comet_logger=logger)

        if epoch % 10 == 0:
            # save the model checkpoints locally and to comet
            torch.save(model, os.path.join(save_dir, f'checkpoints/model_epoch_{epoch}.pth'))
            print('Saved model from epoch ', epoch)

    print('total time: ', round(time.time() - start_time, 3))
    return model


def create_datloaders(batchsize=100, dataset="dummy", base_path="/data/sabera/mts_v2_datasets/pipeline/revin_data_to_train_vqvae"):

    if dataset == 'weather':
        print('weather')
        full_path = base_path + '/weather'
        
    elif dataset == 'electricity':
        print('electricity')
        full_path = base_path + '/electricity'

    elif dataset == 'ETTh1':
        print('ETTh1')
        full_path = base_path + '/ETTh1'

    elif dataset == 'ETTm1':
        print('ETTm1')
        full_path = base_path + '/ETTm1'

    elif dataset == 'ETTh2':
        print('ETTh2')
        full_path = base_path + '/ETTh2'

    elif dataset == 'ETTm2':
        print('ETTm2')
        full_path = base_path + '/ETTm2'

    elif dataset == 'pt12':
        print('PT 12')
        full_path = base_path + '/pt12'

    elif dataset == 'pt2':
        print('PT 2')
        full_path = base_path + '/pt2'

    elif dataset == 'pt5':
        print('PT 5')
        full_path = base_path + '/pt5'

    elif dataset == 'earthquake_0split':
        print('earthquake_0shotsplot')
        # it says compression4 but that is just where the original data is saved --> FIX ME LATER
        full_path = '/data/sabera/mts_v2_datasets/earthquake_clean_0shotsplit/vqvae_results/compression4'

    elif dataset == 'earthquake_randomsplit':
        print('earthquake random split')
        # it says compression4 but that is just where the original data is saved --> FIX ME LATER
        full_path = '/data/sabera/mts_v2_datasets/earthquake_clean_randomsplit/vqvae_results/compression4'

    elif dataset == 'neuro_train2and5_test12':
        print('neuro_train2and5_test12')
        full_path = base_path + '/trainpt2and5_testpt12'
    else:
        full_path = base_path + f'/{dataset}'
        print(f'using {full_path} for dataset')

    train_data = np.load(os.path.join(full_path, "train_data_x.npy"), allow_pickle=True)
    val_data = np.load(os.path.join(full_path, "val_data_x.npy"), allow_pickle=True)
    test_data = np.load(os.path.join(full_path, "test_data_x.npy"), allow_pickle=True)

    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=batchsize,
                                                   shuffle=True,
                                                   num_workers=1,
                                                   drop_last=True)

    val_dataloader = torch.utils.data.DataLoader(val_data,
                                                batch_size=batchsize,
                                                shuffle=False,
                                                num_workers=1,
                                                drop_last=False)

    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                batch_size=batchsize,
                                                shuffle=False,
                                                num_workers=1,
                                                drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    main()
