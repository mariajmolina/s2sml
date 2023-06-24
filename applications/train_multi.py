import os
import sys
import yaml
import logging
import traceback
import random
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from skimage import filters

import torch
from torch.utils.data import DataLoader
import lpips

from echo.src.base_objective import BaseObjective
from collections import defaultdict

import optuna
import shutil

import s2sml.torch_s2s_dataset as torch_s2s_dataset
from s2sml.load_loss import load_loss
from s2sml.load_model import load_model
from s2sml.pareto import pareto_front
from s2sml.scheduler import CosineAnnealingWarmupRestarts

import gc
from piqa import SSIM


def device_assignment_using_trials(trl_number):
    """
    Assign the device to use based on trial number
    """
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        if np.isin(trl_number, np.arange(0,2000,4)):
            device = torch.device("cuda:"+str(0))
        elif np.isin(trl_number, np.arange(1,2000,4)):
            device = torch.device("cuda:"+str(1))
        elif np.isin(trl_number, np.arange(2,2000,4)):
            device = torch.device("cuda:"+str(2))
        else:
            device = torch.device("cuda:"+str(3))
    else:
        device = torch.device("cpu")
    return device


def seed_everything(seed=1234):
    """
    Set the seeds for stuff
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True


def avg_grad(X, T, Y):
    """Average Magnitude of the Gradient
    (from AI2ES sharpness repo)
    
    Edge magnitude is computed as:
        sqrt(Gx^2 + Gy^2)
    """
    def _f(x): return np.mean(filters.sobel(x))
    return _f(X), _f(T), _f(Y)


def batch_grad(X, T, Y):
    """Average Magnitude of the Gradient
    (from AI2ES sharpness repo)
    
    Edge magnitude is computed as:
        sqrt(Gx^2 + Gy^2)
    """
    def _f(x): return filters.sobel(x)
    return _f(X), _f(T), _f(Y)


def grab_extremes(lbl_, other, var_):
    """
    Grab most `extreme` samples from batch labels.
    Grab corresponding input or ml output for mse.
    
    Args:
        lbl_: labels
        other: other data
        var_: variable; string
    """
    # use both max and min for temp
    if var_ == "tas2m":
        lbl_ext = lbl_[torch.where((lbl_==lbl_.max())|(lbl_==lbl_.min()))[0]]
        oth_ext = other[torch.where((lbl_==lbl_.max())|(lbl_==lbl_.min()))[0]]
    
    # only use max for precip
    elif var_ == "prsfc":
        lbl_ext = lbl_[torch.where(lbl_==lbl_.max())[0]]
        oth_ext = other[torch.where(lbl_==lbl_.max())[0]]
        
    return oth_ext, lbl_ext


def mae_land_batch(inp, obs, mask, csm=False):
    """
    Mean absolute error for masked data.
    
    Args:
        inp: input data
        obs: label data
        mask: mask
        csm: cesm data
    """
    if not np.any(csm):
    
        masked_inp = inp[xr.where(mask==1, True, False)]
        masked_obs = obs[xr.where(mask==1, True, False)]

        return np.mean(np.abs(masked_inp - masked_obs))
    
    if np.any(csm):
        
        masked_inp = inp[xr.where(mask==1, True, False)]
        masked_csm = csm[xr.where(mask==1, True, False)]
        masked_obs = obs[xr.where(mask==1, True, False)]
        
        return np.mean(np.abs(masked_inp - masked_obs)) / np.mean(
                       np.abs(masked_csm - masked_obs))

    
def mse_land_batch(inp, obs, mask, csm=False):
    """
    Mean absolute error for masked data.
    
    Args:
        inp: input data
        obs: label data
        mask: mask
        csm: cesm data
    """
    if not np.any(csm):
    
        masked_inp = inp[xr.where(mask==1, True, False)]
        masked_obs = obs[xr.where(mask==1, True, False)]

        return ((masked_inp - masked_obs)**2).mean()
    
    if np.any(csm):
        
        masked_inp = inp[xr.where(mask==1, True, False)]
        masked_csm = csm[xr.where(mask==1, True, False)]
        masked_obs = obs[xr.where(mask==1, True, False)]
        
        return ((masked_inp - masked_obs)**2).mean() / ((masked_csm - masked_obs)**2).mean()
    

class SSIMLoss(SSIM):
    """
    Structural Similarity Index
    Its a perceptual metric to measure similarity of two images
    """
    def forward(self, x, y):
        try:
            return super().forward(x, y).item()
        except Exception as E:
            return -10


def reverse_negone(ds, minv, maxv):
    """
    Reversal of the negative one to positive one scaling
    """
    return (((ds + 1) / 2) * (maxv - minv)) + minv


def train_one_epoch(model, dataloader, optimizer, criterion, nc, device,
                    clip=1.0, lr_schedule_name=False, lr_scheduler=False):
    """
    Training function.

    Args:
        model (torch): pytorch neural network
        dataloader (torch): pytorch dataloader
        optimizer: training optimizer
        criterion: loss function
        nc: number of channels
        clip: clipping of gradients, defaults to 1.0
    """
    # set the model to train mode
    model.train()
    
    # setting the running metrics to zero
    running_loss  = 0.0
    
    # mse/mae per batch
    mse_custom = 0.0
    mae_custom = 0.0
    mse_metric_batch = torch.nn.MSELoss().to(device)
    mae_metric_batch = torch.nn.L1Loss().to(device)
    
    # sharpness metric
    grad_inp = 0.0 # cesm
    grad_lbl = 0.0 # era5
    grad_out = 0.0 # ML
    mae_grad = 0.0
    
    # loop through data in the loader
    for data in dataloader:
        
        # grab the data input features
        img_noisy = data["input"].squeeze(dim=2)
        img_noisy = img_noisy.to(device, dtype=torch.float)
        
        # grab the data labels
        img_label = data["label"].squeeze(dim=2)
        img_label = img_label.to(device, dtype=torch.float)
        
        optimizer.zero_grad() # set the gradients to zero
        
        outputs = model(img_noisy) # predict using the ML model
        
        loss = criterion(outputs, img_label) # loss: ML vs labels
        
        # mse/mae per batch
        mse_loss = mse_metric_batch(
            outputs, img_label) / mse_metric_batch(img_noisy[:,nc-1:nc,:,:], img_label)
        mae_loss = mae_metric_batch(
            outputs, img_label) / mae_metric_batch(img_noisy[:,nc-1:nc,:,:], img_label)
        
        # sharpness
        ginp, glbl, gout = avg_grad(img_noisy[:,nc-1:nc,:,:].cpu().detach().numpy().astype(float),
                                    img_label.cpu().detach().numpy().astype(float),
                                    outputs.cpu().detach().numpy().astype(float))
        
        # sharpness mae
        ginp_shrp, glbl_shrp, gout_shrp = batch_grad(
                                    img_noisy[:,nc-1:nc,:,:].cpu().detach().numpy().astype(float),
                                    img_label.cpu().detach().numpy().astype(float),
                                    outputs.cpu().detach().numpy().astype(float))
        mae_shrp = np.mean(np.abs(gout_shrp - glbl_shrp)) / np.mean(np.abs(ginp_shrp - glbl_shrp))
        
        # update weights
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        # update metrics
        running_loss += loss.item()
        mse_custom += mse_loss.item()
        mae_custom += mae_loss.item()
        grad_inp += ginp.item()
        grad_lbl += glbl.item()
        grad_out += gout.item()
        mae_grad += mae_shrp.item()
        
        if lr_schedule_name == "Cosine":
            lr_scheduler.step()
        
    # updates
    train_loss = running_loss / len(dataloader)
    mse_cust = mse_custom / len(dataloader)
    mae_cust = mae_custom / len(dataloader)
    grad_inp_cust = grad_inp / len(dataloader)
    grad_lbl_cust = grad_lbl / len(dataloader)
    grad_out_cust = grad_out / len(dataloader)
    mae_gradient = mae_grad / len(dataloader)

    # clear the cached memory from the gpu
    torch.cuda.empty_cache()
    gc.collect()
    
    return (train_loss, mse_cust, mae_cust, 
            grad_inp_cust, grad_lbl_cust, grad_out_cust, mae_gradient)


@torch.no_grad()
def validate(model, dataloader, criterion, metrics, second_metrics, 
             nc, device, epoch, trial_num, gen_img, gen_scatter, 
             img_iters, save_loc, var, data_split="valid"):
    """
    Validation function.

    Args:
        model: pytorch neural network
        dataloader: pytorch dataloader
        criterion: loss function
        metrics: additional metrics for skill assessment
        second_metrics: metrics for input data (e.g., data to be bias corrected; cesm)
        nc: number of channels
        epoch: current epoch number
        trial_num: number of trial in optuna study
        gen_img: boolean; True if images wanted
        gen_scatter: boolean; True if scatter plots wanted
        img_iters: integer; spaces out image frequency
        save_loc: location to save figures
        var: variable string
        data_split: validation or test data, for figure saving; defaults to valid
    """
    # set model to eval mode
    model.eval()
    
    # set running metrics to zero
    running_loss = 0.0
    
    # grab metrics dictionary
    metrics_dict = defaultdict(list)
    second_metrics_dict = defaultdict(list)
    
    # mse/mae custom metrics
    mse_custom = 0.0
    mae_custom = 0.0
    mse_metric_batch = torch.nn.MSELoss().to(device)
    mae_metric_batch = torch.nn.L1Loss().to(device)
    
    # sharpness
    grad_inp = 0.0 # cesm
    grad_lbl = 0.0 # era5
    grad_out = 0.0 # ML
    mae_grad = 0.0
    
    # extremes mse
    mse_extr_loss = 0.0 # ml vs era5
    mse_extr_true = 0.0 # cesm vs era5
    
    # land only
    land_mae = 0.0
    land_mae_cust = 0.0
    land_mse = 0.0
    land_mse_cust = 0.0
    
    # loop thru data in loader
    for i, data in enumerate(dataloader):
        
        # load input features
        img_noisy = data["input"].squeeze(dim=2)
        img_noisy = img_noisy.to(device, dtype=torch.float)
        
        # load labels
        img_label = data["label"].squeeze(dim=2)
        img_label = img_label.to(device, dtype=torch.float)
        
        outputs = model(img_noisy) # predict the model output
        
        # if images are to be saved, do this
        if gen_img:
            
            # only if frequency criteria is met, based on epoch
            if int(epoch) % img_iters == 0:
                
                # save figs for later reference
                # selecting random sample in batch to visualize
                a_inp = img_noisy.cpu().detach().numpy()[:,nc-1,:,:]
                samp_ = np.random.choice(np.arange(0,int(a_inp.shape[0]),1))

                # change plt logging level otherwise get a lot of debug output
                plt.set_loglevel(level='warning')

                # input
                im = plt.imshow(a_inp[samp_])
                plt.colorbar(im)
                plt.savefig(
                    f"{save_loc}/trial{str(trial_num)}/{data_split}_inp_{str(epoch)}_{str(trial_num)}.png", 
                    bbox_inches='tight')
                plt.close()

                # output
                b_out = outputs.cpu().detach().numpy()[:,0,:,:][samp_]
                im = plt.imshow(b_out)
                plt.colorbar(im)
                plt.savefig(
                    f"{save_loc}/trial{str(trial_num)}/{data_split}_out_{str(epoch)}_{str(trial_num)}.png", 
                    bbox_inches='tight')
                plt.close()

                # label
                c_lbl = img_label.cpu().detach().numpy()[:,0,:,:][samp_]
                im = plt.imshow(c_lbl)
                plt.colorbar(im)
                plt.savefig(
                    f"{save_loc}/trial{str(trial_num)}/{data_split}_lbl_{str(epoch)}_{str(trial_num)}.png", 
                    bbox_inches='tight')
                plt.close()
        
        # compute metrics
        loss = criterion(outputs, img_label) # loss: ML vs era5
        
        # if scatter images are to be saved, do this
        if gen_scatter:

            # only if frequency criteria is met, based on epoch
            if int(epoch) % img_iters == 0:

                # save figs for later reference
                # input data
                a_inp = img_noisy.cpu().detach().numpy()[:,nc-1,:,:].ravel()
                # output data
                b_out = outputs.cpu().detach().numpy()[:,0,:,:].ravel()
                # label
                c_lbl = img_label.cpu().detach().numpy()[:,0,:,:].ravel()

                # change plt logging level otherwise get a lot of debug output
                plt.set_loglevel(level='warning')

                # cesm vs era5
                im = plt.scatter(a_inp, c_lbl, s=5, color='k')
                plt.ylabel('ERA5')
                plt.xlabel('CESM')
                
                plt.savefig(
                    f"{save_loc}/trial{str(trial_num)}/scatter_{data_split}_inp_{str(epoch)}_{str(trial_num)}.png", 
                    bbox_inches='tight')
                plt.close()
                
                # ML vs era5
                im = plt.scatter(b_out, c_lbl, s=5, color='k')
                plt.ylabel('ERA5')
                plt.xlabel('ML')
                
                plt.savefig(
                    f"{save_loc}/trial{str(trial_num)}/scatter_{data_split}_out_{str(epoch)}_{str(trial_num)}.png", 
                    bbox_inches='tight')
                plt.close()
        
        # mse/mae per batch
        mse_loss = mse_metric_batch(outputs, img_label) / mse_metric_batch(img_noisy[:,nc-1:nc,:,:], img_label)
        mae_loss = mae_metric_batch(outputs, img_label) / mae_metric_batch(img_noisy[:,nc-1:nc,:,:], img_label)
        
        # sharpness
        ginp, glbl, gout = avg_grad(img_noisy[:,nc-1:nc,:,:].cpu().detach().numpy().astype(float),
                                    img_label.cpu().detach().numpy().astype(float),
                                    outputs.cpu().detach().numpy().astype(float))
        
        # sharpness mae
        ginp_shrp, glbl_shrp, gout_shrp = batch_grad(
                                    img_noisy[:,nc-1:nc,:,:].cpu().detach().numpy().astype(float),
                                    img_label.cpu().detach().numpy().astype(float),
                                    outputs.cpu().detach().numpy().astype(float))
        mae_shrp = np.mean(np.abs(gout_shrp - glbl_shrp)) / np.mean(np.abs(ginp_shrp - glbl_shrp))
        
        # mse for extreme cases in batch (ml output)
        out_tmp, lbl_tmp = grab_extremes(img_label, outputs, var)
        mse_extr_outp = mse_metric_batch(out_tmp, lbl_tmp)
        
        # cesm fields
        img_tmp, lbl_tmp = grab_extremes(img_label, img_noisy[:,nc-1:nc,:,:], var)
        mse_extr_cesm = mse_metric_batch(img_tmp, lbl_tmp)
        
        # mae and mse for land only
        landmae = mae_land_batch(
            inp=outputs.cpu().detach().numpy().astype(float),
            obs=img_label.cpu().detach().numpy().astype(float),
            mask=data["lmask"].squeeze(dim=2).cpu().detach().numpy().astype(float),
        )
        
        landmae_cust = mae_land_batch(
            inp=outputs.cpu().detach().numpy().astype(float),
            obs=img_label.cpu().detach().numpy().astype(float),
            mask=data["lmask"].squeeze(dim=2).cpu().detach().numpy().astype(float),
            csm=img_noisy[:,nc-1:nc,:,:].cpu().detach().numpy().astype(float),
        )
        
        landmse = mse_land_batch(
            inp=outputs.cpu().detach().numpy().astype(float),
            obs=img_label.cpu().detach().numpy().astype(float),
            mask=data["lmask"].squeeze(dim=2).cpu().detach().numpy().astype(float),
        )
        
        landmse_cust = mse_land_batch(
            inp=outputs.cpu().detach().numpy().astype(float),
            obs=img_label.cpu().detach().numpy().astype(float),
            mask=data["lmask"].squeeze(dim=2).cpu().detach().numpy().astype(float),
            csm=img_noisy[:,nc-1:nc,:,:].cpu().detach().numpy().astype(float),
        )

        # ml model output eval 
        for k, v in metrics.items():
            try:
                metrics_dict[k].append(
                    v(outputs, img_label).cpu().numpy().mean().item()
                )
            except AttributeError:  # AttributeError
                metrics_dict[k].append(v(outputs, img_label))
                
        # cesm eval
        for k, v in second_metrics.items():
            try:
                second_metrics_dict[k].append(
                    v(img_noisy[:, nc - 1 : nc, :, :], 
                      img_label).cpu().numpy().mean().item()
                )
            except AttributeError:  # AttributeError
                second_metrics_dict[k].append(v(img_noisy[:, nc - 1 : nc, :, :], 
                                                img_label))
        
        # update metrics
        running_loss += loss.item()
        mse_custom += mse_loss.item()
        mae_custom += mae_loss.item()
        grad_inp += ginp.item()
        grad_lbl += glbl.item()
        grad_out += gout.item()
        mae_grad += mae_shrp.item()
        mse_extr_loss += mse_extr_outp.item()
        mse_extr_true += mse_extr_cesm.item()
        land_mae += landmae.item()
        land_mae_cust += landmae_cust.item()
        land_mse += landmse.item()
        land_mse_cust += landmse_cust.item()
        
    # update running metrics
    val_loss = running_loss / len(dataloader)
    mse_cust = mse_custom / len(dataloader)
    mae_cust = mae_custom / len(dataloader)
    grad_inp_cust = grad_inp / len(dataloader)
    grad_lbl_cust = grad_lbl / len(dataloader)
    grad_out_cust = grad_out / len(dataloader)
    mae_gradient = mae_grad / len(dataloader)
    mse_ex_loss = mse_extr_loss / len(dataloader)
    mse_ex_true = mse_extr_true / len(dataloader)
    lnd_mae = land_mae / len(dataloader)
    lnd_mae_cust = land_mae_cust / len(dataloader)
    lnd_mse = land_mse / len(dataloader)
    lnd_mse_cust = land_mse_cust / len(dataloader)
    
    # place stuff in dictionary
    metrics_dict = {k: np.mean(v) for k, v in metrics_dict.items()}
    second_metrics_dict = {k: np.mean(v) for k, v in second_metrics_dict.items()}

    return (val_loss, mse_cust, mae_cust,
            grad_inp_cust, grad_lbl_cust, grad_out_cust, mae_gradient,
            mse_ex_loss, mse_ex_true, lnd_mae, lnd_mae_cust, lnd_mse, lnd_mse_cust,
            metrics_dict, second_metrics_dict)


@torch.no_grad()
def gen_images_only(model, dataloader, nc, device, epoch, trial_num, save_loc, var, data_split="valid"):
    """
    Images function.

    Args:
        model: pytorch neural network
        dataloader: pytorch dataloader
        nc: number of channels
        epoch: current epoch number
        trial_num: number of trial in optuna study
        save_loc: location to save figures
        var: variable string
        data_split: validation or test data, for figure saving; defaults to valid
    """
    # set model to eval mode
    model.eval()
    
    # loop thru data in loader
    for i, data in enumerate(dataloader):
        
        # load input features
        img_noisy = data["input"].squeeze(dim=2)
        img_noisy = img_noisy.to(device, dtype=torch.float)
        
        # load labels
        img_label = data["label"].squeeze(dim=2)
        img_label = img_label.to(device, dtype=torch.float)
        
        outputs = model(img_noisy) # predict the model output
                
        # save figs for later reference
        # selecting random sample in batch to visualize
        a_inp = img_noisy.cpu().detach().numpy()[:,nc-1,:,:]
        samp_ = np.random.choice(np.arange(0,int(a_inp.shape[0]),1))

        # change plt logging level otherwise get a lot of debug output
        plt.set_loglevel(level='warning')

        # input
        im = plt.imshow(a_inp[samp_])
        plt.colorbar(im)
        plt.savefig(
            f"{save_loc}/trial{str(trial_num)}/{data_split}_inp_{str(epoch)}_{str(trial_num)}.png", 
            bbox_inches='tight')
        plt.close()

        # output
        b_out = outputs.cpu().detach().numpy()[:,0,:,:][samp_]
        im = plt.imshow(b_out)
        plt.colorbar(im)
        plt.savefig(
            f"{save_loc}/trial{str(trial_num)}/{data_split}_out_{str(epoch)}_{str(trial_num)}.png", 
            bbox_inches='tight')
        plt.close()

        # label
        c_lbl = img_label.cpu().detach().numpy()[:,0,:,:][samp_]
        im = plt.imshow(c_lbl)
        plt.colorbar(im)
        plt.savefig(
            f"{save_loc}/trial{str(trial_num)}/{data_split}_lbl_{str(epoch)}_{str(trial_num)}.png", 
            bbox_inches='tight')
        plt.close()

        # save figs for later reference
        # input data
        a_inp = img_noisy.cpu().detach().numpy()[:,nc-1,:,:].ravel()
        # output data
        b_out = outputs.cpu().detach().numpy()[:,0,:,:].ravel()
        # label
        c_lbl = img_label.cpu().detach().numpy()[:,0,:,:].ravel()

        # change plt logging level otherwise get a lot of debug output
        plt.set_loglevel(level='warning')

        # cesm vs era5
        im = plt.scatter(a_inp, c_lbl, s=5, color='k')
        plt.ylabel('ERA5')
        plt.xlabel('CESM')
                
        plt.savefig(
            f"{save_loc}/trial{str(trial_num)}/scatter_{data_split}_inp_{str(epoch)}_{str(trial_num)}.png", 
            bbox_inches='tight')
        plt.close()
                
        # ML vs era5
        im = plt.scatter(b_out, c_lbl, s=5, color='k')
        plt.ylabel('ERA5')
        plt.xlabel('ML')
                
        plt.savefig(
            f"{save_loc}/trial{str(trial_num)}/scatter_{data_split}_out_{str(epoch)}_{str(trial_num)}.png", 
            bbox_inches='tight')
        plt.close()
        
        break # just one set of images needed
        
    return


def trainer(conf, trial=False, verbose=True):
    
    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    # Trainer params
    train_batch_size = conf["trainer"]["train_batch_size"]
    valid_batch_size = conf["trainer"]["valid_batch_size"]
    
    epochs = conf["trainer"]["epochs"]
    batches_per_epoch = conf["trainer"]["batches_per_epoch"]

    lr_patience = conf["trainer"]["lr_patience"]
    stopping_patience = conf["trainer"]["stopping_patience"]
    callback_metric = conf["trainer"]["metric"]
    callback_direction = conf["trainer"]["direction"]
    
    nc = conf["model"]["in_channels"]
    feattopo = conf["data"]["feat_topo"]
    featcoord = conf["data"]["feat_coord"]
    
    save_loc = conf["save_loc"]
    os.makedirs(save_loc, exist_ok=True)
    if not os.path.join(save_loc, "model.yml"):
        shutil.copyfile(config, os.path.join(save_loc, "model.yml"))
        
    homedir = conf["data"]["homedir"]
    
    # whether to generate image output and frequency
    gen_img = conf["img_gen"]
    gen_scatter = conf["scatter_gen"]
    if gen_img:
        img_iters = conf["img_iter"]
    if not gen_img:
        img_iters = None
    
    last_images = conf["only_last_img"]
    save_models = conf["save_models"]

    # Data
    var = conf["data"]["var"]
    wks = conf["data"]["wks"]
    dxdy = conf["data"]["dxdy"]
    lat0 = conf["data"]["lat0"]
    lon0 = conf["data"]["lon0"]
    norm = conf["data"]["norm"]
    norm_pixel = conf["data"]["norm_pixel"]
    dual_norm = conf["data"]["dual_norm"]
    region = conf["data"]["region"]
    
    # folder to save trial images
    trial_number = "" if not trial else int(trial.number)
    os.makedirs(save_loc+"/trial"+str(trial_number), exist_ok=True)
    
    # assign device
    device = device_assignment_using_trials(trial_number)
    
    # Load model
    model = load_model(conf["model"]).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=conf["optimizer"]["learning_rate"],
        weight_decay=conf["optimizer"]["weight_decay"],
    )

    # Loss
    train_loss = load_loss(conf["trainer"]["training_loss"]).to(device)
    valid_loss = load_loss(conf["trainer"]["training_loss"]).to(device)

    # Metrics
    validation_metrics = {
        "perc": lpips.LPIPS(net="alex").to(device),
        "mse": torch.nn.MSELoss().to(device),
        "mae": torch.nn.L1Loss().to(device),
        "ssim": SSIMLoss(n_channels=1).to(device).eval(),
    }
    
    # metrics for the baseline data (cesm)
    validation_metrics_cesm = {
        "cesm_perc": lpips.LPIPS(net="alex").to(device),
        "cesm_mse": torch.nn.MSELoss().to(device),
        "cesm_mae": torch.nn.L1Loss().to(device),
        "cesm_ssim": SSIMLoss(n_channels=1).to(device).eval(),
    }
    
    # dictionary to store final metrics
    results_dict = defaultdict(list)
    
    # train and validate
    for epoch in list(range(epochs)):
        
        # create train/test data each epoch if random regions used, otherwise, just once for fixed region
        if (epoch == 0 and region == "fixed") or (region == "random") or (region == "quasi") or (
            epoch == 0 and region == "global"):
        
            train = torch_s2s_dataset.S2SDataset(
                week=wks,
                variable=var,
                norm=norm,
                norm_pixel=norm_pixel,
                dual_norm=dual_norm,
                region=region,
                minv=None,
                maxv=None,
                mini=None,
                maxi=None,
                mnv=None,
                stdv=None,
                mni=None,
                stdi=None,
                lon0=lon0,
                lat0=lat0,
                dxdy=dxdy,
                feat_topo=feattopo,
                feat_lats=featcoord,
                feat_lons=featcoord,
                startdt="1999-02-01",
                enddt="2014-12-31",
                homedir=homedir,
            )

            # set lr_scheduler
            if epoch == 0:

                if conf["optimizer"]["lr_scheduler"] == "Cosine":

                    lr_scheduler = CosineAnnealingWarmupRestarts(
                        optimizer,
                        first_cycle_steps=train.__len__(),
                        cycle_mult=1.0,
                        max_lr=conf["optimizer"]["learning_rate"],
                        min_lr=1e-3 * conf["optimizer"]["learning_rate"],
                        warmup_steps=50,
                        gamma=0.8,
                    )

                if conf["optimizer"]["lr_scheduler"] == "ReduceOnPlateau":

                    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, patience=lr_patience, verbose=verbose, min_lr=1.0e-13
                    )

            if not norm or norm == "None":
                
                # min-max
                tmin = None # era5
                tmax = None # era5
                tmin_inp = None # cesm
                tmax_inp = None # cesm
                
                # z-score
                tmu = None
                tsig = None
                tmu_inp = None
                tsig_inp = None
                
            elif norm in ["minmax", "negone"]:
                
                # min-max
                tmin = train.min_val
                tmax = train.max_val
                tmin_inp = train.min_inp
                tmax_inp = train.max_inp
                
                # z-score
                tmu = None
                tsig = None
                tmu_inp = None
                tsig_inp = None
                
            elif norm == "zscore":
                
                # min-max
                tmin = None
                tmax = None
                tmin_inp = None
                tmax_inp = None
                
                # z-score
                tmu = train.mean_val
                tsig = train.std_val
                tmu_inp = train.mean_inp
                tsig_inp = train.std_inp

            valid = torch_s2s_dataset.S2SDataset(
                week=wks,
                variable=var,
                norm=norm,
                norm_pixel=norm_pixel,
                dual_norm=dual_norm,
                region=region,
                minv=tmin,
                maxv=tmax,
                mini=tmin_inp,
                maxi=tmax_inp,
                mnv=tmu,
                stdv=tsig,
                mni=tmu_inp,
                stdi=tsig_inp,
                lon0=lon0,
                lat0=lat0,
                dxdy=dxdy,
                feat_topo=feattopo,
                feat_lats=featcoord,
                feat_lons=featcoord,
                startdt="2015-01-01",
                enddt="2017-12-31",
                homedir=homedir,
            )

            tests = torch_s2s_dataset.S2SDataset(
                week=wks,
                variable=var,
                norm=norm,
                norm_pixel=norm_pixel,
                dual_norm=dual_norm,
                region=region,
                minv=tmin,
                maxv=tmax,
                mini=tmin_inp,
                maxi=tmax_inp,
                mnv=tmu,
                stdv=tsig,
                mni=tmu_inp,
                stdi=tsig_inp,
                lon0=lon0,
                lat0=lat0,
                dxdy=dxdy,
                feat_topo=feattopo,
                feat_lats=featcoord,
                feat_lons=featcoord,
                startdt="2018-01-01",
                enddt="2020-12-31",
                homedir=homedir,
            )

            train_loader = DataLoader(
                train, batch_size=train_batch_size, shuffle=True, drop_last=True
            )
            valid_loader = DataLoader(
                valid, batch_size=valid_batch_size, shuffle=True, drop_last=True
            )
            tests_loader = DataLoader(
                tests, batch_size=valid_batch_size, shuffle=False, drop_last=False
            )
        
        # train
        tloss, tmsecust, tmaecust, tginp, tglbl, tgout, tgmae = train_one_epoch(
            model, train_loader, optimizer, train_loss, nc, device,
            lr_schedule_name=conf["optimizer"]["lr_scheduler"], lr_scheduler=lr_scheduler
        )
        # validate
        vloss, vmsecust, vmaecust, vginp, vglbl, vgout, vgmae, vmse_x, vmse_x_, vlnd_mae, vlnd_maec, vlnd_mse, vlnd_msec, metrics, cesm_metrics = validate(
            model, valid_loader, valid_loss, validation_metrics, validation_metrics_cesm, 
            nc, device, epoch, trial_number, gen_img, gen_scatter, img_iters, save_loc, var, data_split="valid"
        )
        # test
        eloss, emsecust, emaecust, eginp, eglbl, egout, egmae, emse_x, emse_x_, elnd_mae, elnd_maec, elnd_mse, elnd_msec, emetrics, cesm_emetrics = validate(
            model, tests_loader, valid_loss, validation_metrics, validation_metrics_cesm, 
            nc, device, epoch, trial_number, gen_img, gen_scatter, img_iters, save_loc, var, data_split="eval"
        )
        
        assert np.isfinite(vloss), "Something is wrong, the validation loss is NaN"
        
        # place metrics from the training and validation in the dictionary to save out
        results_dict["epoch"].append(epoch) # the epoch
        results_dict["lr"].append(optimizer.param_groups[0]["lr"]) # learning rate
        
        # training set
        results_dict["train_loss"].append(tloss) # loss from echo/optuna
        results_dict["tmse_cust"].append(tmsecust) # mse (ML/era5//cesm/era5)
        results_dict["tmae_cust"].append(tmaecust) # mae (ML/era5//cesm/era5)
        results_dict["tgrad_inp"].append(tginp) # horizontal gradient (cesm)
        results_dict["tgrad_lbl"].append(tglbl) # horizontal gradient (era5)
        results_dict["tgrad_out"].append(tgout) # horizontal gradient (ML)
        results_dict["tgrad_mae"].append(tgmae) # horizontal gradient mae 
        
        # validation set
        results_dict["valid_loss"].append(vloss) # loss from echo/optuna
        results_dict["vmse_cust"].append(vmsecust) # mse (ML/era5//cesm/era5)
        results_dict["vmae_cust"].append(vmaecust) # mae (ML/era5//cesm/era5)
        results_dict["vgrad_inp"].append(vginp) # horizontal gradient (cesm)
        results_dict["vgrad_lbl"].append(vglbl) # horizontal gradient (era5)
        results_dict["vgrad_out"].append(vgout) # horizontal gradient (ML)
        results_dict["vgrad_mae"].append(vgmae) # horizontal gradient mae 
        results_dict["vmse_extreme_outp"].append(vmse_x) # mse for extremes (ML vs era5)
        results_dict["vmse_extreme_cesm"].append(vmse_x_) # mse for extremes (cesm vs era5)
        results_dict["vland_mae"].append(vlnd_mae) # mae over land only
        results_dict["vland_mae_cust"].append(vlnd_maec) # mae over land only (ratio with cesm)
        results_dict["vland_mse"].append(vlnd_mse) # mse over land only
        results_dict["vland_mse_cust"].append(vlnd_msec) # mse over land only (ratio with cesm)
        
        # other metrics (validation set)
        for k, v in metrics.items():
            results_dict[f"valid_{k}"].append(v)
        for k, v in cesm_metrics.items():
            results_dict[f"valid_{k}"].append(v)
        
        # evaluation set
        results_dict["evals_loss"].append(eloss) # loss from echo/optuna
        results_dict["emse_cust"].append(emsecust) # mse (ML/era5//cesm/era5)
        results_dict["emae_cust"].append(emaecust) # mae (ML/era5//cesm/era5)
        results_dict["egrad_inp"].append(eginp) # horizontal gradient (cesm)
        results_dict["egrad_lbl"].append(eglbl) # horizontal gradient (era5)
        results_dict["egrad_out"].append(egout) # horizontal gradient (ML)
        results_dict["egrad_mae"].append(egmae) # horizontal gradient mae
        results_dict["emse_extreme_outp"].append(emse_x) # mse for extremes (ML vs era5)
        results_dict["emse_extreme_cesm"].append(emse_x_) # mse for extremes (cesm vs era5)
        results_dict["eland_mae"].append(vlnd_mae) # mae over land only
        results_dict["eland_mae_cust"].append(vlnd_maec) # mae over land only (ratio with cesm)
        results_dict["eland_mse"].append(vlnd_mse) # mse over land only
        results_dict["eland_mse_cust"].append(vlnd_msec) # mse over land only (ratio with cesm)
        
        # other metrics (evaluation set)
        for k, v in emetrics.items():
            results_dict[f"evals_{k}"].append(v)
        for k, v in cesm_emetrics.items():
            results_dict[f"evals_{k}"].append(v)
        
        # number of ML model parameters and trainable parameters, respectively
        results_dict["total_params"].append(sum(p.numel() for p in model.parameters()))
        results_dict["train_params"].append(sum(p.numel() for p in model.parameters() if p.requires_grad))

        # Save the dataframe to disk
        df = pd.DataFrame.from_dict(results_dict).reset_index()
        if verbose:
            df.to_csv(f"{save_loc}/trial{str(trial_number)}/training_log{str(trial_number)}.csv", index=False)
        
        # Call pareto and check the training callbacks
        costs = df[callback_metric].values
        best_epochs = np.where(pareto_front(costs, callback_direction))[0] # this zero is fine
        best_costs = list(zip(best_epochs, list(costs[best_epochs])))
        
        # choose metric/direction for stopping during first epoch
        if epoch == 0:
            if isinstance(callback_direction, list):
                metric_index = np.random.choice(len(callback_metric))
                metric_callback_direction = callback_direction[metric_index]
                metric = callback_metric[metric_index]
            else:
                metric_index = 0
                metric_callback_direction = callback_direction
                metric = callback_metric
        
        sign = False if "minimize" in metric_callback_direction else True
        best_costs.sort(key = lambda x: x[1][metric_index], reverse=sign)
        best_epoch, best_cost = best_costs[0] # this zero is fine
        offset = epoch - best_epoch
        
        # Stop training if we have not improved after X epochs based on metric
        if offset >= stopping_patience:
            
            # save model
            if save_models:
                
                state_dict = {
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "stopping_patience": stopping_patience,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "training_loss": conf["trainer"]["training_loss"],
                    "callback_metric_choice": metric,
                }
                
                torch.save(state_dict, 
                           f"{save_loc}/trial{str(trial_number)}/model_{str(trial_number)}.pt")
            
            break
    
    if trial is False:
        return pd.DataFrame.from_dict(results_dict).reset_index()
    
    # return the results from the respective dictionary
    results = {k: v[best_epoch] for k, v in results_dict.items()}
    
    if last_images:
        gen_images_only(
            model, valid_loader, nc, device, epoch, trial_number, save_loc, var, data_split="valid"
        )
        gen_images_only(
            model, tests_loader, nc, device, epoch, trial_number, save_loc, var, data_split="eval"
        )
    
    return results


class Objective(BaseObjective):
    def __init__(self, config, metric="val_loss", device="cpu"):

        # Initialize the base class
        BaseObjective.__init__(self, config, metric, device)

    def train(self, trial, conf):
        
        try:
            return trainer(conf, trial=trial, verbose=True)
        
        except Exception as E:
            
            if "CUDA" in str(E) or "cuDNN" in str(E):
                logging.warning(
                    f"Pruning trial {trial.number} due to CUDA memory overflow: {str(E)}."
                )
                logging.warning(traceback.print_tb(E.__traceback__))
                raise optuna.TrialPruned()
                
            elif "Xception" in str(E) or "VGG" in str(E) or "Given input size:" in str(E) or "downsampling" in str(E):
                logging.warning(
                    f"Pruning trial {trial.number} due to encoder/encoder weights mismatch: {str(E)}."
                )
                logging.warning(traceback.print_tb(E.__traceback__))
                raise optuna.TrialPruned()
                
            elif "reraise" in str(E):
                logging.warning(
                    f"Pruning trial {trial.number} due to unspecified error: {str(E)}."
                )
                logging.warning(traceback.print_tb(E.__traceback__))
                raise optuna.TrialPruned()
                
            else:
                logging.warning(f"Trial {trial.number} failed due to error: {str(E)}.")
                logging.warning(traceback.print_tb(E.__traceback__))
                raise E


def launch_pbs_jobs(config, save_path="./"):
    from pathlib import Path
    import subprocess

    script_path = Path(__file__).absolute()
    script = f"""
    #!/bin/bash -l
    #PBS -N holo-trainer
    #PBS -l select=1:ncpus=8:ngpus=1:mem=128GB
    #PBS -l walltime=24:00:00
    #PBS -l gpu_type=v100
    #PBS -A NAML0001
    #PBS -q casper
    #PBS -o {os.path.join(save_path, "out")}
    #PBS -e {os.path.join(save_path, "out")}

    source ~/.bashrc
    ncar_pylib /glade/work/$USER/py37
    python {script_path} {config}
    """
    with open("launcher.sh", "w") as fid:
        fid.write(script)
    jobid = subprocess.Popen(
        "qsub launcher.sh",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).communicate()[0]
    jobid = jobid.decode("utf-8").strip("\n")
    print(jobid)
    os.remove("launcher.sh")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python train_mlp.py model.yml")
        sys.exit()

    # Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)
    
    # Load the configuration and get the relevant variables
    config = sys.argv[1]
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)
    
    results = trainer(conf)
    
    print(results)