# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from models import DiT_models
from diffusion import create_diffusion
from torch.utils.data import TensorDataset, DataLoader

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1-decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def patchify(x):
    """
    input:  (B, T, C)
    output: (B, C, H, W)
    """
    t = int(x.shape[1] ** 0.5)
    x = x.permute(0, 2, 1)
    x = x.reshape(x.shape[0], x.shape[1], t, t)
    return x

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    print(f"Args: {args}")
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    device = torch.device('cuda')

    # set up an experiment folder
    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

    # create model
    model = DiT_models[args.model](num_classes=args.num_classes, in_channels=args.in_channels, num_patches=args.num_patches)

    # note that parameter initialization is done within the DiT constructor
    model = model.to(device)
    model_without_ddp = model

    if args.compile:
        model = torch.compile(model)

    # TODO: do I need to compile ema too?    
    ema = deepcopy(model_without_ddp).to(device)  # create an EMA of the model for use after training
    requires_grad(ema, False)

    print(f"Model: {model_without_ddp}")
    print(f"Number of params (M): {(sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad) / 1.e6)}")
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule

    # set up optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper, add scheduler?)
    opt = torch.optim.AdamW(model_without_ddp.parameters(), lr=1e-4, weight_decay=0, fused=True)
    scaler = torch.amp.GradScaler('cuda')

    # set up data
    train_data = torch.load(args.train_data_path, map_location='cpu')
    dataset = TensorDataset(train_data['latents'], train_data['targets'])
    loader = DataLoader(
        dataset, 
        sampler=None, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        shuffle=True,
        pin_memory=True,
        drop_last=True
        )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.train_data_path})")

    # prepare models for training
    # TODO: do I need to wrap update_ema? Probly not, but CHECK
    update_ema(ema, model_without_ddp, decay=0)  # ensure ema is initialized with synced weights
    model.train()  # this enables embedding dropout for classifier-free guidance
    ema.eval()  # ema model should always be in eval mode

    # variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    
    # infinite stream of training data
    for epoch in range(args.epochs):
        logger.info(f"Starting epoch {epoch} ...")
        for _, (samples, targets) in enumerate(loader):
            # move to gpu
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # reshape batch
            samples = patchify(samples)

            # TODO: check if need to downcast t
            t = torch.randint(0, diffusion.num_timesteps, (samples.shape[0],), device=device)
            model_kwargs = dict(y=targets)

            # FIXME: inconsistent use of 'cuda' & device 
            with torch.amp.autocast('cuda'):
                loss_dict = diffusion.training_losses(model, samples, t, model_kwargs)
            
            loss = loss_dict["loss"].mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            opt.zero_grad()
            # TODO: do I need to wrap epdate_ema? Probly not, but CHECK
            update_ema(ema, model_without_ddp)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                # measure training speed
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                
                # reduce loss history over all processes
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                
                # reset monitoring variables
                running_loss = 0
                log_steps = 0
                start_time = time()

            # save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint = {
                    "model": model_without_ddp.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

    # TODO: optionally do sampling/FID calculation etc. with ema (or model) in eval mode
    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', default='', type=str)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT_XL")
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--in_channels", type=int, default=64)
    parser.add_argument("--num_patches", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--tae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=50000)
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model for improved efficiency (default: false)')        
    parser.add_argument('--num_workers', default=16, type=int)    

    args = parser.parse_args()
    main(args)