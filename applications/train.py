import traceback
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

import lpips
import logging
import sys
import yaml
import random
import os
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
        model (torch): pytorch neural network.
        dataloader (torch): pytorch dataloader.
    """
    # set the model to train mode
    model.train()
    
    # setting the running metrics to zero
    running_loss = 0.0
    corrcoef_loss = 0.0
    corrcoef_true = 0.0
    
    # loop through data in the loader
    for data in dataloader:
        
        # grab the data input features
        img_noisy = data["input"].squeeze(dim=2)
        img_noisy = img_noisy.to(device, dtype=torch.float)
        
        # grab the data labels
        img_label = data["label"].squeeze(dim=2)
        img_label = img_label.to(device, dtype=torch.float)
        
        # set the gradients to zero
        optimizer.zero_grad()
        
        # predict using the model
        outputs = model(img_noisy)
        outputs[outputs==1] = 0.999 # in order to avoid correlation being None
        outputs[outputs==-1] = -0.999 # in order to avoid correlation being None

        # measure the loss with model output vs labels
        loss = criterion(outputs, img_label)
        
        # correlation coefficient measurement for the model output vs labels
        closs = torch_funcs.corrcoef(outputs, img_label)
        # correlation coefficient measurement for the raw input vs labels
        tloss = torch_funcs.corrcoef(img_noisy[:, nc - 1 : nc, :, :], img_label)
        
        # update weights
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        # update metrics
        running_loss += loss.item()
        corrcoef_loss += closs.item()
        corrcoef_true += tloss.item()
        
    # updates
    train_loss = running_loss / len(dataloader)
    coef_loss = corrcoef_loss / len(dataloader)
    coef_true = corrcoef_true / len(dataloader)

    # clear the cached memory from the gpu
    torch.cuda.empty_cache()
    gc.collect()
    
    return train_loss, coef_loss, coef_true


@torch.no_grad()
def validate(model, dataloader, criterion, metrics, second_metrics, nc):
    """
    Validation function.

    Args:
        model: pytorch neural network.
        dataloader: pytorch dataloader.
    """
    # set model to eval mode
    model.eval()
    
    # setting running metrics to zero
    running_loss = 0.0
    corrcoef_loss = 0.0
    corrcoef_true = 0.0
    
    # grab the metrics dictionary
    metrics_dict = defaultdict(list)
    second_metrics_dict = defaultdict(list)
    
    # loop thru data in loader
    for i, data in enumerate(dataloader):
        
        # load input features
        img_noisy = data["input"].squeeze(dim=2)
        img_noisy = img_noisy.to(device, dtype=torch.float)
        
        # load labels
        img_label = data["label"].squeeze(dim=2)
        img_label = img_label.to(device, dtype=torch.float)
        
        # predict the model output
        outputs = model(img_noisy)
        outputs[outputs==1] = 0.999 # in order to avoid correlation being None
        outputs[outputs==-1] = -0.999 # in order to avoid correlation being None

        # evaluate the model output vs labels
        loss = criterion(outputs, img_label)
        # evaluate correlation coefficient of model output vs labels
        closs = torch_funcs.corrcoef(outputs, img_label)
        # evaluate correlation coefficient of cesm vs labels
        tloss = torch_funcs.corrcoef(img_noisy[:, nc - 1 : nc, :, :], img_label)

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
        
    # update running metrics
    val_loss = running_loss / len(dataloader)
    coef_loss = corrcoef_loss / len(dataloader)
    coef_true = corrcoef_true / len(dataloader)
    
    # place stuff in dictionary
    metrics_dict = {k: np.mean(v) for k, v in metrics_dict.items()}
    second_metrics_dict = {k: np.mean(v) for k, v in second_metrics_dict.items()}

    return val_loss, coef_loss, coef_true, metrics_dict, second_metrics_dict


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
        enddt="2015-12-31",
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
        startdt="2016-01-01",
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
        valid, batch_size=valid_batch_size, shuffle=False, drop_last=False
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
        "rmse": torch.nn.MSELoss().to(device),
        "mae":  torch.nn.L1Loss().to(device),
        "ssim": SSIMLoss(n_channels=1).to(device).eval(),
    }
    # metrics for the baseline data (cesm)
    validation_metrics_cesm = {
        "cesm_perc": lpips.LPIPS(net="alex").to(device),
        "cesm_rmse": torch.nn.MSELoss().to(device),
        "cesm_mae":  torch.nn.L1Loss().to(device),
        "cesm_ssim": SSIMLoss(n_channels=1).to(device).eval(),
    }
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=lr_patience, verbose=verbose, min_lr=1.0e-13
    )
    
    # Train and validate
    results_dict = defaultdict(list)
    
    for epoch in list(range(epochs)):
        
        t_loss, t_corr, t_true = train_one_epoch(
            model, train_loader, optimizer, train_loss, nc
        )
        v_loss, v_corr, v_true, metrics, cesm_metrics = validate(
            model, valid_loader, valid_loss, validation_metrics, validation_metrics_cesm, nc
        )
        e_loss, e_corr, e_true, emetrics, cesm_emetrics = validate(
            model, tests_loader, valid_loss, validation_metrics, validation_metrics_cesm, nc
        )

        assert np.isfinite(v_loss), "Something is wrong, the validation loss is NaN"
        
        # place metrics from the training and validation in the dictionary to save out
        # the epoch
        results_dict["epoch"].append(epoch)
        
        # the train loss and correlation value
        results_dict["train_loss"].append(t_loss)
        results_dict["train_corr"].append(t_corr)
        results_dict["tcesm_corr"].append(t_true)
        results_dict["train_custom"].append(t_true / t_corr)
        
        # the validation loss and corr value
        results_dict["valid_loss"].append(v_loss)
        results_dict["valid_corr"].append(v_corr)
        results_dict["vcesm_corr"].append(v_true)
        results_dict["valid_custom"].append(v_true / v_corr)
        
        # other metrics computed from the validation set
        for k, v in metrics.items():
            results_dict[f"valid_{k}"].append(v)
        for k, v in cesm_metrics.items():
            results_dict[f"valid_{k}"].append(v)
        
        results_dict["valid_custom_second"].append(
            results_dict["valid_rmse"][-1] / results_dict["valid_cesm_rmse"][-1])
        results_dict["valid_custom_third"].append(
            results_dict["valid_mae"][-1] / results_dict["valid_cesm_mae"][-1])
        
        # the test/evaluation loss and corr value
        results_dict["evals_loss"].append(e_loss)
        results_dict["evals_corr"].append(e_corr)
        results_dict["ecesm_corr"].append(e_true)
        results_dict["evals_custom"].append(e_true / e_corr)
        
        # other metrics computed from the tests/evaluation set
        for k, v in emetrics.items():
            results_dict[f"evals_{k}"].append(v)
        for k, v in cesm_emetrics.items():
            results_dict[f"evals_{k}"].append(v)
        
        results_dict["evals_custom_second"].append(
            results_dict["evals_rmse"][-1] / results_dict["evals_cesm_rmse"][-1])
        results_dict["evals_custom_third"].append(
            results_dict["evals_mae"][-1] / results_dict["evals_cesm_mae"][-1])
        
        # save the learning rate
        results_dict["lr"].append(optimizer.param_groups[0]["lr"])

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
