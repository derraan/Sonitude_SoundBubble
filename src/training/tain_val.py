"""
The main training script for training on synthetic data
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import tqdm


def to_device(batch, device):
    if type(batch) == torch.Tensor:
        return batch.to(device)
    elif type(batch) == dict:
        for k in batch:
            batch[k] = to_device(batch[k], device)
        return batch
    elif type(batch) in [list, tuple]:
        batch = [to_device(x, device) for x in batch]
        return batch
    else:
        return batch

def test_epoch(hl_module, test_loader, device) -> float:
    """
    Evaluate the network.
    """
    hl_module.eval()
    
    test_loss = 0
    num_elements = 0

    num_batches = len(test_loader)
    pbar = tqdm.tqdm(total=num_batches)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = to_device(batch, device)
            
            loss, B = hl_module.validation_step(batch, batch_idx)
            #print(loss.item(), B)
            test_loss += (loss.item() * B)
            num_elements += B

            pbar.set_postfix(loss='%.05f'%(loss.item()) )
            pbar.update()

        return test_loss / num_elements

def train_epoch(
    hl_module,
    train_loader,
    device,
    scaler=None,
    use_amp: bool = False,
    amp_dtype=None,
    grad_accum_steps: int = 1,
    log_mem_every: int = 0,
    max_steps: int = 0,
) -> float:
    """
    Train a single epoch.
    """    
    # Set the model to training.
    hl_module.train()
    
    # Training loop
    train_loss = 0
    num_elements = 0

    num_batches = len(train_loader)
    if max_steps and max_steps > 0:
        num_batches = min(num_batches, int(max_steps))
    pbar = tqdm.tqdm(total=num_batches)
    amp_enabled = bool(use_amp) and (device.type == "cuda" if hasattr(device, "type") else str(device).startswith("cuda"))
    if amp_dtype is None:
        amp_dtype = torch.float16
    grad_accum_steps = max(1, int(grad_accum_steps))
    use_scaler = amp_enabled and scaler is not None and scaler.is_enabled()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    hl_module.optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, batch in enumerate(train_loader):
        if max_steps and batch_idx >= max_steps:
            break

        batch = to_device(batch, device)

        # Forward pass (+ optional AMP)
        autocast_kwargs = {"device_type": "cuda", "enabled": amp_enabled}
        if amp_enabled:
            autocast_kwargs["dtype"] = amp_dtype
        with torch.amp.autocast(**autocast_kwargs):
            loss, B = hl_module.training_step(batch, batch_idx)

        # Non-finite loss: skip step so AMP/edge-aux NaNs cannot poison weights.
        if not torch.isfinite(loss).all():
            hl_module.optimizer.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.update()
            pbar.set_postfix(loss='nan/skip')
            pbar.update()
            continue

        # Scale for gradient accumulation (effective batch stays constant).
        loss = loss / grad_accum_steps

        # Backpropagation
        if use_scaler:
            scaler.scale(loss).backward(retain_graph=False)
        else:
            loss.backward(retain_graph=False)

        step_now = ((batch_idx + 1) % grad_accum_steps == 0) or (
            (max_steps > 0 and (batch_idx + 1) >= max_steps)
            or (batch_idx + 1) == len(train_loader)
        )

        if step_now:
            if use_scaler:
                scaler.unscale_(hl_module.optimizer)
                if hl_module.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(hl_module.model.parameters(), hl_module.grad_clip)
                scaler.step(hl_module.optimizer)
                scaler.update()
            else:
                if hl_module.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(hl_module.model.parameters(), hl_module.grad_clip)
                hl_module.optimizer.step()
            hl_module.optimizer.zero_grad(set_to_none=True)

        # Save losses (report unscaled micro-batch loss for readability)
        loss_item = float(loss.detach().item() * grad_accum_steps)
        train_loss += (loss_item * B)
        num_elements += B
        postfix = {'loss': '%.05f' % loss_item}
        if log_mem_every and device.type == "cuda" and ((batch_idx + 1) % log_mem_every == 0):
            alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
            postfix['mem'] = '%.2f/%.2fGiB' % (alloc_gb, reserved_gb)
            print(
                f"[mem] step={batch_idx + 1} "
                f"max_alloc={alloc_gb:.2f}GiB max_reserved={reserved_gb:.2f}GiB"
            )
        pbar.set_postfix(**postfix)
        pbar.update()

    if num_elements == 0:
        return 0.0
    return train_loss / num_elements
