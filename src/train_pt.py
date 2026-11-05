"""
The main training script for training on synthetic data
"""

import torch
import torch.utils.data
import torch.nn as nn

import argparse
import json
import os
import multiprocessing
import time

import numpy as np
import src.utils as utils
from src.training.tain_val import train_epoch, test_epoch
import shutil

try:
    import wandb
except ImportError:  # optional; --no_wandb works without it installed
    wandb = None

VAL_SEED = 0
CURRENT_EPOCH = 0
TRAIN_SEED = 0


class _NoOpWandbRun:
    def log(self, *_args, **_kwargs):
        return None

    def finish(self, *_args, **_kwargs):
        return None


def seed_from_epoch(seed):
    global CURRENT_EPOCH

    utils.seed_all(seed + CURRENT_EPOCH)


def worker_init_train(_worker_id: int):
    seed_from_epoch(TRAIN_SEED)


def worker_init_val(_worker_id: int):
    utils.seed_all(VAL_SEED)


def print_metrics(metrics: list):
    input_sisdr = np.array([x['input_si_sdr'] for x in metrics])
    sisdr = np.array([x['si_sdr'] for x in metrics])

    print("Average Input SI-SDR: {:03f}, Average Output SI-SDR: {:03f}, Average SI-SDRi: {:03f}".format(np.mean(input_sisdr), np.mean(sisdr), np.mean(sisdr - input_sisdr)))


def train(args: argparse.Namespace):
    """
    Resolve the network to be trained
    """
    # Fix random seeds
    utils.seed_all(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

    # Load experiment description early so CUDA/alloc knobs can apply before tensors.
    with open(args.config, 'rb') as f:
        params = json.load(f)

    # Windows-safe allocator knobs (expandable_segments is ignored on Windows).
    alloc_conf = params.get(
        "cuda_alloc_conf",
        "garbage_collection_threshold:0.8,max_split_size_mb:256",
    )
    if alloc_conf and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = str(alloc_conf)
        print(f"PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}")

    cudnn_benchmark = bool(params.get("cudnn_benchmark", False)) or bool(
        getattr(args, "use_nondeterministic_cudnn", False)
    )
    if torch.cuda.is_available():
        if cudnn_benchmark:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            print("cuDNN benchmark=True (non-deterministic)")
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Initialize datasets
    data_train = utils.import_attr(params['train_dataset'])(**params['train_data_args'], split='train')
    data_val = utils.import_attr(params['val_dataset'])(**params['val_data_args'], split='val')

    # Set up the device and workers
    use_cuda = True
    device = torch.device('cuda' if use_cuda else 'cpu')
    print("Using device {}".format('cuda' if use_cuda else 'cpu'))

    gpu_mem_fraction = params.get("gpu_mem_fraction", None)
    if gpu_mem_fraction is not None and device.type == "cuda":
        frac = float(gpu_mem_fraction)
        torch.cuda.set_per_process_memory_fraction(frac)
        print(f"GPU memory fraction capped at {frac:.2f}")

    # Set multiprocessing params
    # Windows spawn can't pickle nested lambdas; use module-level worker_init_fns.
    global TRAIN_SEED
    TRAIN_SEED = args.seed
    num_workers = min(multiprocessing.cpu_count(), params['num_workers'])
    kwargs = {
        'num_workers': num_workers,
        'worker_init_fn': worker_init_train,
        'pin_memory': True
    } if use_cuda else {}
    if use_cuda and num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = int(params.get('prefetch_factor', 4))

    # Set up data loaders
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=params['batch_size'],
                                               shuffle=True,
                                               **kwargs)
   
    val_kwargs = dict(kwargs)
    if len(val_kwargs) > 0:
        val_kwargs['worker_init_fn'] = worker_init_val
    test_loader = torch.utils.data.DataLoader(data_val,
                                              batch_size=params['eval_batch_size'],
                                              **val_kwargs)

    # Initialize HL module
    hl_module = utils.import_attr(params['pl_module'])(**params['pl_module_args'])
    hl_module.model.to(device)

    use_amp = bool(params.get("use_amp", False)) or bool(getattr(args, "amp", False))
    amp_dtype_name = str(params.get("amp_dtype", "bf16")).lower()
    if amp_dtype_name in ("bf16", "bfloat16"):
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    # GradScaler is only needed for fp16; bf16 on Ada does not use it.
    scaler_enabled = use_amp and device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    if use_amp:
        print(
            f"Using AMP dtype={amp_dtype_name} "
            f"(GradScaler enabled={scaler.is_enabled()})"
        )

    grad_accum_steps = max(1, int(params.get("grad_accum_steps", 1)))
    log_mem_every = int(params.get("log_mem_every", 0))
    max_steps = int(getattr(args, "max_steps", 0) or params.get("max_steps", 0) or 0)
    if grad_accum_steps > 1:
        print(f"Gradient accumulation steps: {grad_accum_steps}")
    if max_steps > 0:
        print(f"Max train steps this run: {max_steps}")
    
    # Get run name from run dir
    run_name = os.path.basename(args.run_dir.rstrip('/'))
    checkpoints_dir = os.path.join(args.run_dir, 'checkpoints')

    # Set up checkpoints
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # Copy json
    if not os.path.exists(os.path.join(args.run_dir, 'config.json')):
        shutil.copyfile(args.config, os.path.join(args.run_dir, 'config.json'))

    # Check if a model state path exists for this model, if it does, load it
    best_path = os.path.join(checkpoints_dir, 'best.pt')
    state_path = os.path.join(checkpoints_dir, 'last.pt')
    if os.path.exists(state_path):
        hl_module.load_state(state_path)

    start_epoch = hl_module.epoch
    
    if "project_name" in params.keys():
        project_name = params["project_name"]
    else:
        project_name = "AcousticBubble"
    # Initialize wandb (optional)
    wandb_run = _NoOpWandbRun()
    if not getattr(args, "no_wandb", False):
        if wandb is None:
            print("[wandb] disabled (package not installed)")
        else:
            try:
                wandb_run = wandb.init(
                    project=project_name,
                    name=run_name,
                    notes="Example of a note",
                    tags=["speech", "audio", "embedded-systems"],
                )
            except Exception as e:
                print(f"[wandb] disabled ({type(e).__name__}: {e})")
                wandb_run = _NoOpWandbRun()

    # Training loop
    try:        
        # Go over remaining epochs
        for epoch in range(start_epoch, params['epochs']):
            global CURRENT_EPOCH, VAL_SEED
            CURRENT_EPOCH = epoch
            seed_from_epoch(args.seed)

            hl_module.on_epoch_start()

            current_lr = hl_module.get_current_lr()
            print("CURRENT learning rate: {:0.08f}".format(current_lr))

            print("[TRAINING]")
            
            # Run testing step
            
            t1 = time.time()
            train_loss = train_epoch(
                hl_module,
                train_loader,
                device,
                scaler=scaler,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                grad_accum_steps=grad_accum_steps,
                log_mem_every=log_mem_every,
                max_steps=max_steps,
            )
            t2 = time.time()
            print(f"Train epoch time: {t2 - t1:02f}s")

            print("\nTrain set: Average Loss: {:.4f}\n".format(train_loss))

            print()

            # Probe / short runs: skip validation + checkpoint write.
            if max_steps > 0:
                if device.type == "cuda":
                    peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
                    print(
                        f"[mem] peak_alloc={peak_alloc:.2f}GiB "
                        f"peak_reserved={peak_reserved:.2f}GiB"
                    )
                print(f"max_steps={max_steps} reached; skipping val/checkpoint.")
                break

            # Fix seed for all validation passes 
            # (needed since localization models will choose some random points, so we fix those)
            utils.seed_all(VAL_SEED)

            # Run testing step

            print("[TESTING]")
            
            test_loss = test_epoch(hl_module, test_loader, device)
            
            print("\nTest set: Average Loss: {:.4f}\n".format(test_loss))
                    
            # # Add and save losses as pkl file
            # train_losses.append(train_loss)
            # val_losses.append(test_loss)
                    
            # # Save model params, optimizer, scheduler, losses
            # torch.save(model.module.state_dict(), os.path.join(checkpoints_dir, experiment_name + "_{}.pt".format(epoch)))
            # state = {
            #     'epoch': epoch,
            #     'optimizer': optimizer.state_dict(),
            #     'lr_sched': scheduler,
            #     'train_losses': train_losses,
            #     'val_losses': val_losses,
            # }
            # torch.save(state, state_path)
            hl_module.on_epoch_end(best_path, wandb_run)
            hl_module.dump_state(state_path)

            print()
            print("=" * 25, "FINISHED EPOCH", epoch, "=" * 25)
            print()

    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as _:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Experiment Params
    parser.add_argument('--config', type=str,
                        help='Path to experiment config')

    parser.add_argument('--run_dir', type=str,
                        help='Path to experiment directory')

    # Randomization Params
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--use_nondeterministic_cudnn',
                        action='store_true',
                        help="If using cuda, chooses whether or not to use \
                                non-deterministic cudDNN algorithms. Training will be\
                                faster, but the final results may differ slighty.")
    
    # wandb params
    parser.add_argument('--project_name',
                        type=str,
                        default='AcousticBubble',
                        help='Project name that shows up on wandb')
    parser.add_argument('--no_wandb',
                        action='store_true',
                        help='Disable Weights & Biases logging (offline/no key).')
    parser.add_argument('--amp',
                        action='store_true',
                        help='Enable CUDA automatic mixed precision (also set use_amp in config).')
    parser.add_argument('--max_steps',
                        type=int,
                        default=0,
                        help='If >0, run only this many train steps then exit (memory/speed probe).')
    train(parser.parse_args())
