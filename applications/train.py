import os
import sys
import yaml
import logging
import traceback
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import filters

import torch
from torch.utils.data import DataLoader
# from fvcore.nn import FlopCountAnalysis # doesn't work with certain ML models
import lpips

from echo.src.base_objective import BaseObjective
from collections import defaultdict

import tqdm
import optuna
import shutil

import s2sml.torch_funcs as torch_funcs
import s2sml.torch_s2s_dataset as torch_s2s_dataset
from s2sml.load_loss import load_loss
from s2sml.load_model import load_model
import gc
from piqa import SSIM

#print('loading cuda')
is_cuda = torch.cuda.is_available()
device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")


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


def train_one_epoch(model, dataloader, optimizer, criterion, nc, clip=1.0):
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
    corrcoef_loss = 0.0
    corrcoef_true = 0.0
    corrcoef_cust = 0.0
    
    # mse/mae per batch
    mse_custom = 0.0
    mae_custom = 0.0
    mse_metric_batch = torch.nn.MSELoss().to(device)
    mae_metric_batch = torch.nn.L1Loss().to(device)
    
    # sharpness metric
    grad_inp = 0.0 # cesm
    grad_lbl = 0.0 # era5
    grad_out = 0.0 # ML
    
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
        closs = torch_funcs.corrcoef(outputs, img_label) # corr: ML vs labels
        tloss = torch_funcs.corrcoef(img_noisy[:,nc-1:nc,:,:], img_label) # corr: input vs labels
        
        # mse/mae per batch
        mse_loss = mse_metric_batch(
            outputs, img_label) / mse_metric_batch(img_noisy[:,nc-1:nc,:,:], img_label)
        mae_loss = mae_metric_batch(
            outputs, img_label) / mae_metric_batch(img_noisy[:,nc-1:nc,:,:], img_label)
        
        # sharpness
        ginp, glbl, gout = avg_grad(img_noisy[:,nc-1:nc,:,:].cpu().detach().numpy().astype(float),
                                    img_label.cpu().detach().numpy().astype(float),
                                    outputs.cpu().detach().numpy().astype(float))
        
        # update weights
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        # update metrics
        running_loss += loss.item()
        corrcoef_loss += closs.item()
        corrcoef_true += tloss.item()
        corrcoef_cust += tloss.item() / closs.item()
        mse_custom += mse_loss.item()
        mae_custom += mae_loss.item()
        grad_inp += ginp.item()
        grad_lbl += glbl.item()
        grad_out += gout.item()
        
    # updates
    train_loss = running_loss / len(dataloader)
    coef_loss = corrcoef_loss / len(dataloader)
    coef_true = corrcoef_true / len(dataloader)
    coef_cust = corrcoef_cust / len(dataloader)
    mse_cust = mse_custom / len(dataloader)
    mae_cust = mae_custom / len(dataloader)
    grad_inp_cust = grad_inp / len(dataloader)
    grad_lbl_cust = grad_lbl / len(dataloader)
    grad_out_cust = grad_out / len(dataloader)

    # clear the cached memory from the gpu
    torch.cuda.empty_cache()
    gc.collect()
    
    return (train_loss, coef_loss, coef_true, coef_cust, mse_cust, mae_cust, 
            grad_inp_cust, grad_lbl_cust, grad_out_cust)


@torch.no_grad()
def validate(model, dataloader, criterion, metrics, second_metrics, 
             nc, epoch, trial_num, gen_img, gen_scatter, 
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
    corrcoef_loss = 0.0
    corrcoef_true = 0.0
    corrcoef_cust = 0.0
    
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
    
    # extremes mse
    mse_extr_loss = 0.0 # ml vs era5
    mse_extr_true = 0.0 # cesm vs era5
    
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
        closs = torch_funcs.corrcoef(outputs, img_label) # corr: ML vs era5
        tloss = torch_funcs.corrcoef(img_noisy[:,nc-1:nc,:,:], img_label) # corr: cesm vs era5
        
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
        mse_loss = mse_metric_batch(
            outputs, img_label) / mse_metric_batch(img_noisy[:,nc-1:nc,:,:], img_label)
        mae_loss = mae_metric_batch(
            outputs, img_label) / mae_metric_batch(img_noisy[:,nc-1:nc,:,:], img_label)
        
        # sharpness
        ginp, glbl, gout = avg_grad(img_noisy[:,nc-1:nc,:,:].cpu().detach().numpy().astype(float),
                                    img_label.cpu().detach().numpy().astype(float),
                                    outputs.cpu().detach().numpy().astype(float))
        
        # mse for extreme cases in batch (ml output)
        out_tmp, lbl_tmp = grab_extremes(img_label, outputs, var)
        mse_extr_outp = mse_metric_batch(out_tmp, lbl_tmp)
        # cesm fields
        img_tmp, lbl_tmp = grab_extremes(img_label, img_noisy[:,nc-1:nc,:,:], var)
        mse_extr_cesm = mse_metric_batch(img_tmp, lbl_tmp)

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
        corrcoef_loss += closs.item()
        corrcoef_true += tloss.item()
        corrcoef_cust += tloss.item() / closs.item()
        mse_custom += mse_loss.item()
        mae_custom += mae_loss.item()
        grad_inp += ginp.item()
        grad_lbl += glbl.item()
        grad_out += gout.item()
        mse_extr_loss += mse_extr_outp.item()
        mse_extr_true += mse_extr_cesm.item()
        
    # update running metrics
    val_loss = running_loss / len(dataloader)
    coef_loss = corrcoef_loss / len(dataloader)
    coef_true = corrcoef_true / len(dataloader)
    coef_cust = corrcoef_cust / len(dataloader)
    mse_cust = mse_custom / len(dataloader)
    mae_cust = mae_custom / len(dataloader)
    grad_inp_cust = grad_inp / len(dataloader)
    grad_lbl_cust = grad_lbl / len(dataloader)
    grad_out_cust = grad_out / len(dataloader)
    mse_ex_loss = mse_extr_loss / len(dataloader)
    mse_ex_true = mse_extr_true / len(dataloader)
    
    # place stuff in dictionary
    metrics_dict = {k: np.mean(v) for k, v in metrics_dict.items()}
    second_metrics_dict = {k: np.mean(v) for k, v in second_metrics_dict.items()}
    
    # flops = FlopCountAnalysis(model, img_noisy)
    # flop_count = flops.total()

    return (val_loss, coef_loss, coef_true, coef_cust, mse_cust, mae_cust,
            grad_inp_cust, grad_lbl_cust, grad_out_cust, # flop_count, 
            mse_ex_loss, mse_ex_true, metrics_dict, second_metrics_dict)


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
    nc = conf["model"]["in_channels"]
    metric = conf["trainer"]["metric"]
    
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

    # Data
    var = conf["data"]["var"]
    wks = conf["data"]["wks"]
    dxdy = conf["data"]["dxdy"]
    lat0 = conf["data"]["lat0"]
    lon0 = conf["data"]["lon0"]
    norm = conf["data"]["norm"]
    
    train = torch_s2s_dataset.S2SDataset(
        week=wks,
        variable=var,
        norm=norm,
        region="fixed",
        minv=None,
        maxv=None,
        mnv=None,
        stdv=None,
        lon0=lon0,
        lat0=lat0,
        dxdy=dxdy,
        feat_topo=True,
        feat_lats=True,
        feat_lons=True,
        startdt="1999-02-01",
        enddt="2014-12-31",
        homedir=homedir,
    )
    
    if not norm or norm == "None":
        tmin = None
        tmax = None
        tmu = None
        tsig = None
    elif norm in ["minmax", "negone"]:
        tmin = train.min_val
        tmax = train.max_val
        tmu = None
        tsig = None
    elif norm == "zscore":
        tmin = None
        tmax = None
        tmu = train.mean_val
        tsig = train.std_val
    
    valid = torch_s2s_dataset.S2SDataset(
        week=wks,
        variable=var,
        norm=norm,
        region="fixed",
        minv=tmin,
        maxv=tmax,
        mnv=tmu,
        stdv=tsig,
        lon0=lon0,
        lat0=lat0,
        dxdy=dxdy,
        feat_topo=True,
        feat_lats=True,
        feat_lons=True,
        startdt="2015-01-01",
        enddt="2017-12-31",
        homedir=homedir,
    )
    
    tests = torch_s2s_dataset.S2SDataset(
        week=wks,
        variable=var,
        norm=norm,
        region="fixed",
        minv=tmin,
        maxv=tmax,
        mnv=tmu,
        stdv=tsig,
        lon0=lon0,
        lat0=lat0,
        dxdy=dxdy,
        feat_topo=True,
        feat_lats=True,
        feat_lons=True,
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
    
    model = load_model(conf["model"]).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=conf["optimizer"]["learning_rate"],
        weight_decay=conf["optimizer"]["weight_decay"],
    )

    # Loss
    train_loss = load_loss(conf["trainer"]["loss"]).to(device)
    valid_loss = load_loss(conf["trainer"]["loss"]).to(device)

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
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=lr_patience, verbose=verbose, min_lr=1.0e-13
    )
    
    # dictionary to store final metrics
    results_dict = defaultdict(list)
    
    # folder to save trial images
    if gen_img:
        os.makedirs(save_loc+"/trial"+str(int(trial.number)), exist_ok=True)
    
    # train and validate
    for epoch in list(range(epochs)):
        
        # train
        tloss, tcorr, ttrue, tcust, tmsecust, tmaecust, tginp, tglbl, tgout = train_one_epoch(
            model, train_loader, optimizer, train_loss, nc
        )
        # validate
        vloss, vcorr, vtrue, vcust, vmsecust, vmaecust, vginp, vglbl, vgout, vmse_x, vmse_x_, metrics, cesm_metrics = validate(
            model, valid_loader, valid_loss, validation_metrics, validation_metrics_cesm, 
            nc, epoch, int(trial.number), gen_img, gen_scatter, img_iters, save_loc, var, data_split="valid"
        )
        # test
        eloss, ecorr, etrue, ecust, emsecust, emaecust, eginp, eglbl, egout, emse_x, emse_x_, emetrics, cesm_emetrics = validate(
            model, tests_loader, valid_loss, validation_metrics, validation_metrics_cesm, 
            nc, epoch, int(trial.number), gen_img, gen_scatter, img_iters, save_loc, var, data_split="eval"
        )
        
        assert np.isfinite(vloss), "Something is wrong, the validation loss is NaN"
        
        # place metrics from the training and validation in the dictionary to save out
        results_dict["epoch"].append(epoch) # the epoch
        results_dict["lr"].append(optimizer.param_groups[0]["lr"]) # learning rate
        
        # training set
        results_dict["train_loss"].append(tloss) # loss from echo/optuna
        results_dict["train_corr"].append(tcorr) # correlation (ML/era5)
        results_dict["tcesm_corr"].append(ttrue) # correlation (cesm/era5)
        results_dict["tcorr_cust"].append(tcust) # correlation (ML/era5//cesm/era5)
        results_dict["tmse_cust"].append(tmsecust) # mse (ML/era5//cesm/era5)
        results_dict["tmae_cust"].append(tmaecust) # mae (ML/era5//cesm/era5)
        results_dict["tgrad_inp"].append(tginp) # horizontal gradient (cesm)
        results_dict["tgrad_lbl"].append(tglbl) # horizontal gradient (era5)
        results_dict["tgrad_out"].append(tgout) # horizontal gradient (ML)
        
        # validation set
        results_dict["valid_loss"].append(vloss) # loss from echo/optuna
        results_dict["valid_corr"].append(vcorr) # correlation (ML/era5)
        results_dict["vcesm_corr"].append(vtrue) # correlation (cesm/era5)
        results_dict["vcorr_cust"].append(vcust) # correlation (ML/era5//cesm/era5)
        results_dict["vmse_cust"].append(vmsecust) # mse (ML/era5//cesm/era5)
        results_dict["vmae_cust"].append(vmaecust) # mae (ML/era5//cesm/era5)
        results_dict["vgrad_inp"].append(vginp) # horizontal gradient (cesm)
        results_dict["vgrad_lbl"].append(vglbl) # horizontal gradient (era5)
        results_dict["vgrad_out"].append(vgout) # horizontal gradient (ML)
        # results_dict["vflop"].append(vflop) # flops (floating point operations per second)
        results_dict["vmse_extreme_outp"].append(vmse_x) # mse for extremes (ML vs era5)
        results_dict["vmse_extreme_cesm"].append(vmse_x_) # mse for extremes (cesm vs era5)
        
        # other metrics (validation set)
        for k, v in metrics.items():
            results_dict[f"valid_{k}"].append(v)
        for k, v in cesm_metrics.items():
            results_dict[f"valid_{k}"].append(v)
        
        # evaluation set
        results_dict["evals_loss"].append(eloss) # loss from echo/optuna
        results_dict["evals_corr"].append(ecorr) # correlation (ML/era5)
        results_dict["ecesm_corr"].append(etrue) # correlation (cesm/era5)
        results_dict["ecorr_cust"].append(ecust) # correlation (ML/era5//cesm/era5)
        results_dict["emse_cust"].append(emsecust) # mse (ML/era5//cesm/era5)
        results_dict["emae_cust"].append(emaecust) # mae (ML/era5//cesm/era5)
        results_dict["egrad_inp"].append(eginp) # horizontal gradient (cesm)
        results_dict["egrad_lbl"].append(eglbl) # horizontal gradient (era5)
        results_dict["egrad_out"].append(egout) # horizontal gradient (ML)
        # results_dict["eflop"].append(eflop) # flops (floating point operations per second)
        results_dict["emse_extreme_outp"].append(emse_x) # mse for extremes (ML vs era5)
        results_dict["emse_extreme_cesm"].append(emse_x_) # mse for extremes (cesm vs era5)
        
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
            df.to_csv(f"{save_loc}/training_log{str(int(trial.number))}.csv", index=False)
        
        # had to comment out due to multi-obj optimization not available in optuna yet
        # update the echo trial
        #if trial:
            # update trails using the (single) defined metric from yml file (important!)
            #trial.report(results_dict[metric][-1], step=epoch)
            #if trial.should_prune():
            #    raise optuna.TrialPruned()
        
        # anneal the learning rate using just the (single) metric
        lr_scheduler.step(results_dict[metric][-1])
        
        # save the best model (only if not using echo)
        if results_dict[metric][-1] == min(results_dict[metric]) and trial is False:
            state_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": min(results_dict[metric]),
            }
            torch.save(state_dict, f"{save_loc}/best.pt")

        # Stop training if we have not improved after X epochs based on the defined metric
        best_epoch = [
            i
            for i, j in enumerate(results_dict[metric])
            if j == min(results_dict[metric])
        ][0]
        offset = epoch - best_epoch
        if offset >= stopping_patience:
            break

    #     # Select the best model
    #     checkpoint = torch.load(
    #         f"{save_loc}/mlp.pt", map_location=lambda storage, loc: storage
    #     )
    #     model.load_state_dict(checkpoint["model_state_dict"])

    #     # Predict on the three splits and compute metrics

    if trial is False:
        return pd.DataFrame.from_dict(results_dict).reset_index()
    
    # the best epoch is based on the single chosen metric!
    best_epoch = [
        i for i, j in enumerate(results_dict[metric]) if j == min(results_dict[metric])
    ][0]
    
    # return the results from the respective dictionary
    results = {k: v[best_epoch] for k, v in results_dict.items()}

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
