import os
import sys
import torch


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        print('Launched with torch.distributed.launch')
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
        print('world size, rank, gpu, device count:', args.world_size, args.rank, args.gpu, torch.cuda.device_count())
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        print('Launched with slurm')
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
        print('world size, rank, gpu, device count:', args.world_size, args.rank, args.gpu, torch.cuda.device_count())
    elif torch.cuda.is_available():
        # launched naively with `python main_dino.py`
        # we manually add MASTER_ADDR and MASTER_PORT to env variables
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    torch.distributed.init_process_group(backend="nccl", init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    torch.distributed.barrier()
    setup_for_distributed(args.rank==0)


def interpolate_pos_embed(model, checkpoint_model):
    """
    Interpolate position embeddings for higher resolution inputs (reference: https://github.com/facebookresearch/deit)
    """
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(f"Interpolating position embeddings from {orig_size} to {new_size}")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


def load_model(ckpt, model_without_ddp, optimizer=None, loss_scaler=None, optim_resume=False):
    if ckpt:
        if ckpt.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(ckpt, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(ckpt, map_location='cpu')

        # interpolate position embeddings
        interpolate_pos_embed(model_without_ddp, checkpoint['model'])        

        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print(f"Resumed checkpoint {ckpt}")
        if 'optimizer' in checkpoint and optim_resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")