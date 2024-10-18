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