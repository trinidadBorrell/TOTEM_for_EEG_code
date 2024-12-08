import comet_ml
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
import pdb
import argparse
from lib.models.revin import RevIN
from lib.models.classif import SimpleMLP, EEGNet, SensorTimeEncoder
from lib.utils.checkpoint import EarlyStopping
from sklearn.metrics import confusion_matrix
from lib.utils.env import seed_all_rng
from datetime import datetime 
import json 


from omegaconf import DictConfig, OmegaConf
import hydra

import logging

log = logging.getLogger(__name__)

@hydra.main(config_path="../conf", version_base="1.1")
def main(cfg: DictConfig) -> None:
    log.info("STEP4: Train xFormer classifier")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    log.info(f'Working directory {os.getcwd()}')

    exp_cfg = cfg.exp
    logging_cfg = cfg.logging


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

    ## Set device 
    device = torch.device("cuda:%d" % (exp_cfg.cuda_id))
    torch.cuda.set_device(device)

    # Log the config parameters to comet_ml
    comet_logger.log_parameters(exp_cfg["classifier_config"])
    train(classifier_config=exp_cfg["classifier_config"], comet_logger=comet_logger, device=device)


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def create_dataloader(datapath, num_classes=1, batchsize=8):
    dataloaders = {}
    for split in ["train", "val", "test"]:
        x_file = os.path.join(datapath, "%s_x_original.npy" % (split))
        x = np.load(x_file)
        x = torch.from_numpy(x).to(dtype=torch.float32)

        y_file = os.path.join(datapath, "%s_x_labels.npy" % (split)) 
        y = np.load(y_file)
        if num_classes > 1: 
            y = to_categorical(y, num_classes=num_classes) 
        y = torch.from_numpy(y).to(dtype=torch.float32)

        codes_file = os.path.join(datapath, "%s_x_codes.npy" % (split))
        codes = np.load(codes_file)
        codes = torch.from_numpy(codes).to(dtype=torch.int64)

        print("[Dataset][%s] %d examples" % (split, x.shape[0]))

        dataset = torch.utils.data.TensorDataset(x, codes, y)
        dataloaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batchsize,
            shuffle=True if split == "train" else False,
            num_workers=1,
            drop_last=True if split == "train" else False,
        )

    return dataloaders


def train_one_epoch(
    dataloader,
    model,
    codebook,
    optimizer,
    scheduler,
    epoch,
    device,
    comet_logger, 
    normalize_trial=False,
):
    running_loss, last_loss = 0.0, 0.0
    running_acc, last_acc = 0.0, 0.0
    log_every = max(len(dataloader) // 3, 3)
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    y_pred = []
    y_true = []
    for i, data in enumerate(dataloader):
        # ----- LOAD DATA ------ #
        x, code_ids, y = data
        # x: (B, T, S)
        # codes: (B, TC, S)  TC = T // compression
        # y: (B,)
        x = x.to(device)
        code_ids = code_ids.to(device)
        y = y.to(device)

        # reshape data
        B, T, S = x.shape
        B, TC, S = code_ids.shape

        # get codewords for input x
        code_ids = code_ids.flatten()
        xcodes = codebook[code_ids]  # (B*TC*S, D)
        xcodes = xcodes.reshape((B, TC, S, xcodes.shape[-1]))  # (B, TC, S, D)

        # revin time series
        norm_x = model.revin(x, "norm")

        if isinstance(model, SimpleMLP):
            x = x.flatten(start_dim=1)
            predy = model(x)
        elif isinstance(model, EEGNet):
            if normalize_trial: 
                ## Use the normalized trial as input to the model
                norm_x = torch.permute(norm_x, (0, 2, 1))  # (B, S, T)
                norm_x = norm_x.unsqueeze(1)  # (B, 1, S, T)
                predy = model(norm_x)
            else: 
                ## Use the original time series as input to the model
                x = torch.permute(x, (0, 2, 1))  # (B, S, T)
                x = x.unsqueeze(1)  # (B, 1, S, T)
                predy = model(x)
        elif isinstance(model, SensorTimeEncoder):
            scale = torch.cat((model.revin.mean, model.revin.stdev), dim=1)
            scale = torch.permute(scale, (0, 2, 1))
            predy = model(xcodes, scale)
        else:
            raise ValueError("womp womp")
        loss = F.cross_entropy(predy, y) #F.binary_cross_entropy_with_logits(predy, y)

        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log
        running_loss += loss.item()
        with torch.no_grad():
            prediction_logits = predy.argmax(dim=1)
            y_logits = y.argmax(dim=1)
            y_pred.extend(list(prediction_logits.cpu().numpy()))
            y_true.extend(list(y_logits.cpu().numpy()))
            # pdb.set_trace()
            correct = (prediction_logits == y_logits)
            running_acc += correct.sum().float() / float(y.size(0)) #((predy.sigmoid() > 0.5) == y).float().mean()
            total_acc += correct.sum().float() / float(y.size(0))
            total_loss += loss.item()
            num_batches += 1
        if i % log_every == log_every - 1:
            last_loss = running_loss / log_every  # loss per batch
            last_acc = running_acc / log_every
            lr = optimizer.param_groups[0]["lr"]
            # lr = scheduler.get_last_lr()[0]
            print(
                f"| epoch {epoch:3d} | {i+1:5d}/{len(dataloader):5d} batches | "
                f"lr {lr:02.5f} | loss {last_loss:5.3f} | acc {last_acc:5.3f}"
            )
            running_loss = 0.0
            running_acc = 0.0

        if scheduler is not None:
            scheduler.step()
    
    total_acc = total_acc / num_batches # acc per element 
    total_loss = total_loss / num_batches # loss per element

    comet_logger.log_metric(f'train_acc', total_acc, epoch=epoch)
    comet_logger.log_metric(f'train_loss', total_loss, epoch=epoch)
    matrix = confusion_matrix(y_true, y_pred)
    comet_logger.log_confusion_matrix(title=f'Train Confusion Matrix', matrix=matrix, epoch=epoch, file_name=f'train_confusion_matrix.json')


def inference(
    data,
    model,
    codebook,
    device,
    normalize_trial=False
):
    x, code_ids, _ = data
    x = x.to(device)
    code_ids = code_ids.to(device)

    # reshape data
    B, T, S = x.shape
    B, TC, S = code_ids.shape

    # get codewords for input x
    code_ids = code_ids.flatten()
    xcodes = codebook[code_ids]  # (B*TC*S, D)
    xcodes = xcodes.reshape((B, TC, S, xcodes.shape[-1]))  # (B, TC, S, D)

    # revin time series
    norm_x = model.revin(x, "norm")

    if isinstance(model, SimpleMLP):
        x = x.flatten(start_dim=1)
        predy = model(x)
    elif isinstance(model, EEGNet):
        if normalize_trial: 
            ## Use the normalized trial as input to the model
            norm_x = torch.permute(norm_x, (0, 2, 1))  # (B, S, T)
            norm_x = norm_x.unsqueeze(1)  # (B, 1, S, T)
            predy = model(norm_x)
        else: 
            ## Use the original time series as input to the model
            x = torch.permute(x, (0, 2, 1))  # (B, S, T)
            x = x.unsqueeze(1)  # (B, 1, S, T)
            predy = model(x)
    elif isinstance(model, SensorTimeEncoder):
        scale = torch.cat((model.revin.mean, model.revin.stdev), dim=1)
        scale = torch.permute(scale, (0, 2, 1))
        predy = model(xcodes, scale)
    else:
        raise ValueError("wamp wamp")

    return predy


def train(classifier_config, comet_logger, device):

    # -------- SET SEED ------- #
    print('Setting seed to {}'.format(classifier_config['seed']))
    seed_all_rng(None if classifier_config['seed'] < 0 else classifier_config['seed'])

    # -------- PARAMS ------- #
    batchsize = classifier_config['batchsize'] 
    datapath = classifier_config['data_path'] 
    expname = classifier_config['exp_name'] 
    nsensors = classifier_config['nsensors'] 
    d_out = classifier_config['nclasses'] 

    # -------- CHECKPOINT ------- #
    checkpath = None
    if classifier_config['checkpoint']:
        checkpath = os.path.join(classifier_config['checkpoint_path'], expname)
        os.makedirs(checkpath, exist_ok=True)
        os.makedirs(os.path.join(checkpath, 'configs'), exist_ok=True)
        os.makedirs(os.path.join(checkpath, 'checkpoints'), exist_ok=True)
    early_stopping = EarlyStopping(patience=classifier_config['patience'], path=checkpath)
    
    # # Save the json copy
    # with open(os.path.join(checkpath, 'configs', 'config_file.json'), 'w+') as f:
    #     json.dump(classifier_config, f, indent=4)

    # ------ DATA LOADERS ------- #
    dataloaders = create_dataloader(
        datapath=datapath, num_classes=d_out, batchsize=batchsize
    )
    train_dataloader = dataloaders["train"]
    val_dataloader = dataloaders["val"]
    test_dataloader = dataloaders["test"]

    # -------- CODEBOOK ------- #
    codebook = np.load(os.path.join(datapath, "codebook.npy"), allow_pickle=True)
    codebook = torch.from_numpy(codebook).to(device=device, dtype=torch.float32)
    vocab_size, vocab_dim = codebook.shape
    assert vocab_size == classifier_config['codebook_size']
    dim = vocab_size if classifier_config['onehot'] else vocab_dim

    # ------- MODEL -------- #
    if classifier_config['model_type'] == "mlp":
        # time --> class (baseline)
        model = SimpleMLP(
            in_dim=nsensors * classifier_config['Tin'], out_dim=1, hidden_dims=[1024, 512, 256], dropout=0.0
        )
    elif classifier_config['model_type'] == "eeg":
        # time --> class (baseline)
        model = EEGNet(
            chunk_size=classifier_config['Tin'],
            num_electrodes=nsensors,
            F1=8,
            F2=16,
            D=2,
            kernel_1=64,
            kernel_2=16,
            dropout=0.25,
            num_classes=d_out, 
        )
    elif classifier_config['model_type'] == "xformer":
        # code --> class (ours)
        model = SensorTimeEncoder(
            d_in=dim,
            d_model=classifier_config['d_model'],
            nheadt=classifier_config['nhead'],
            nheads=classifier_config['nhead'],
            d_hid=classifier_config['d_hid'],
            nlayerst=classifier_config['nlayers'],
            nlayerss=classifier_config['nlayers'],
            seq_lent=classifier_config['Tin'] // classifier_config['compression'],
            seq_lens=nsensors,
            dropout=0.25,
            d_out=d_out, 
            scale=classifier_config['scale'],
        )
    else:
        raise ValueError("Unknown model type %s" % (classifier_config['model_type']))
    model.revin = RevIN(num_features=nsensors, affine=False)  # expects as input (B, T, S)
    model.to(device)

    # ------- OPTIMIZER -------- #
    num_iters = classifier_config['epochs'] * len(train_dataloader)
    step_lr_in_iters = classifier_config['steps'] * len(train_dataloader)
    model_params = list(model.parameters())
    if classifier_config['optimizer'] == "sgd":
        optimizer = torch.optim.SGD(model_params, lr=classifier_config['baselr'], momentum=0.9)
    elif classifier_config['optimizer'] == "adam":
        optimizer = torch.optim.Adam(model_params, lr=classifier_config['baselr'])
    elif classifier_config['optimizer'] == "adamw":
        optimizer = torch.optim.AdamW(model_params, lr=classifier_config['baselr'])
    else:
        raise ValueError("Uknown optimizer type %s" % (classifier_config['optimizer']))
    if classifier_config['scheduler'] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_lr_in_iters, gamma=0.1
        )
    elif classifier_config['scheduler'] == "onecycle":
        # The learning rate will increate from max_lr / div_factor to max_lr in the first pct_start * total_steps steps,
        # and decrease smoothly to max_lr / final_div_factor then.
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=classifier_config['baselr'],
            steps_per_epoch=len(train_dataloader),
            epochs=classifier_config['epochs'],
            pct_start=0.2,
        )
    else:
        raise ValueError("Uknown scheduler type %s" % (classifier_config['scheduler']))

    # ------- TRAIN & EVAL -------- #
    best_val_loss = float("inf")
    for epoch in range(classifier_config['epochs']):
        model.train()
        train_one_epoch(
            train_dataloader,
            model,
            codebook,
            optimizer,
            scheduler,
            epoch,
            device,
            comet_logger=comet_logger,
            normalize_trial=classifier_config.get('normalize_trial', False)
        )

        if epoch % 10 == 0:
            # save the model checkpoints locally and to comet
            torch.save(model.state_dict(), os.path.join(checkpath, f'checkpoints/model_epoch_{epoch}.pth'))
            print('Saved model from epoch ', epoch)

        if val_dataloader is not None:
            model.eval()
            running_acc = 0.0
            running_loss = 0.0
            total_num = 0.0
            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                y_pred = [] 
                y_true = []
                for i, vdata in enumerate(val_dataloader):
                    pred = inference(
                        vdata,
                        model,
                        codebook,
                        device,
                        normalize_trial=classifier_config.get('normalize_trial', False)
                    )
                    y = vdata[-1]
                    y = y.to(device)

                    prediction_logits = pred.argmax(dim=1)
                    y_logits = y.argmax(dim=1)
                    y_pred.extend(list(prediction_logits.cpu().numpy()))
                    y_true.extend(list(y_logits.cpu().numpy()))
                    correct = (prediction_logits == y_logits)
                    running_acc += correct.sum().float()

                    running_loss += F.cross_entropy(
                        pred, y, reduction="sum"
                    )
                    total_num += y.size(0)
            curr_acc = running_acc / total_num
            curr_loss = running_loss / total_num
            print(f"| [Val] loss {curr_loss:5.3f} | acc {curr_acc:5.3f}")
            comet_logger.log_metric(f'val_acc', curr_acc, epoch=epoch)
            comet_logger.log_metric(f'val_loss', curr_loss, epoch=epoch)
            matrix = confusion_matrix(y_true, y_pred)
            comet_logger.log_confusion_matrix(title=f'Val Confusion Matrix', matrix=matrix, epoch=epoch, file_name=f'val_confusion_matrix.json')

            if curr_loss < best_val_loss: 
                best_val_loss = curr_loss
                torch.save(model.state_dict(), os.path.join(checkpath, f'checkpoints/model_best.pth'))
                print('Saved best model from epoch ', epoch)
            
        if test_dataloader is not None:
            model.eval()
            running_acc = 0.0
            total_num = 0.0
            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, tdata in enumerate(test_dataloader):
                    pred = inference(
                        tdata,
                        model,
                        codebook,
                        device,
                    )
                    y = tdata[-1]
                    y = y.to(device)

                    prediction_logits = pred.argmax(dim=1)
                    y_logits = y.argmax(dim=1)
                    correct = (prediction_logits == y_logits)
                    running_acc += correct.sum().float()

                    total_num += y.size(0)
            running_acc = running_acc / total_num
            print(f"| [Test] acc {running_acc:5.3f}")
            comet_logger.log_metric(f'test_acc', running_acc, epoch=epoch)

            if early_stopping.early_stop:
                print("Early stopping....")
                return
    
    ## Final save
    torch.save(model.state_dict(), os.path.join(checkpath, 'checkpoints/final_model.pth'))

    ## Log the final model test acc to comet 
    best_model_path = os.path.join(checkpath, 'checkpoints', 'model_best.pth')
    state_dict = torch.load(best_model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if test_dataloader is not None:
        model.eval()
        running_acc = 0.0
        total_num = 0.0
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, tdata in enumerate(test_dataloader):
                pred = inference(
                    tdata,
                    model,
                    codebook,
                    device,
                )
                y = tdata[-1]
                y = y.to(device)

                prediction_logits = pred.argmax(dim=1)
                y_logits = y.argmax(dim=1)
                correct = (prediction_logits == y_logits)
                running_acc += correct.sum().float()

                total_num += y.size(0)
        running_acc = running_acc / total_num
        print(f"| [Test best model] acc {running_acc:5.3f}")
        comet_logger.log_metric(f'best_model_test_acc', running_acc)


if __name__ == "__main__":
    main() 