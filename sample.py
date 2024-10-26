# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import math
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from download import find_model
from models import DiT_models
import argparse
import tae
import utils


def patchify(x):
    """
    input: (B, C, H, W) 
    output: (B, T, C)
    """
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    return x


def main(args):

    print(f"Args: {args}")

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    model = DiT_models[args.model](num_classes=args.num_classes, in_channels=args.in_channels, num_patches=args.num_patches).to(device)
    # TODO: add model compile
    # load a pretrained DiT checkpoint
    state_dict = find_model(args.model_ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))
    tae_model = tae.__dict__[args.tae]()
    utils.load_model(args.tae_ckpt, tae_model)
    tae_model = tae_model.to(device)

    # labels to condition the model with (feel free to change)
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

    # create sampling noise
    n = len(class_labels)
    latent_size = int(math.sqrt(args.num_patches))
    z = torch.randn(n, args.in_channels, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # setup classifier-free guidance
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # sample images
    samples = diffusion.p_sample_loop(model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device)
    print(samples.shape)
    samples, _ = samples.chunk(2, dim=0)  # remove null class samples
    print(samples.shape)
    samples = tae_model.forward_decoder(patchify(samples / 0.18215))  # do I need this divisive factor, 0.18215? (probly not)
    print(samples.shape)
    samples = tae_model.unpatchify(samples)
    print(samples.shape)
    print(samples.min(), samples.max())

    # save and display images
    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT_XL")
    parser.add_argument("--model_ckpt", type=str)
    parser.add_argument("--tae", type=str)
    parser.add_argument("--tae_ckpt", type=str)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--in_channels", type=int, default=64)
    parser.add_argument("--num_patches", type=int, default=256)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None, help="path to a pretrained DiT checkpoint")
    args = parser.parse_args()
    main(args)