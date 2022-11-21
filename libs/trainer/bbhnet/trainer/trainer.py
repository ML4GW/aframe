import logging
import os
import time
from typing import Callable, Iterable, Optional, Tuple

import h5py
import numpy as np
import torch


def train_for_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    train_dataset: Iterable[Tuple[np.ndarray, np.ndarray]],
    validator: Optional[Callable] = None,
    profiler: Optional[torch.profiler.profile] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    scheduler: Optional[Callable] = None,
):
    """Run a single epoch of training"""

    train_loss = 0
    samples_seen = 0
    start_time = time.time()
    model.train()

    for samples, targets in train_dataset:
        optimizer.zero_grad(set_to_none=True)  # reset gradient

        # do forward step in mixed precision
        # hard code false for now
        with torch.autocast("cuda", enabled=scaler is not None):
            predictions = model(samples)
            loss = criterion(predictions, targets)
        train_loss += loss.item()
        samples_seen += len(samples)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if profiler is not None:
            profiler.step()
        if scheduler is not None:
            scheduler.step()

    if profiler is not None:
        profiler.stop()

    end_time = time.time()
    duration = end_time - start_time
    throughput = samples_seen / duration
    train_loss /= samples_seen

    logging.info(
        "Duration {:0.2f}s, Throughput {:0.1f} samples/s".format(
            duration, throughput
        )
    )
    # Evaluate performance on validation set if given
    if validator is not None:
        model.eval()
        return validator(model, train_loss)
    return False


def train(
    architecture: Callable,
    outdir: str,
    # data params
    train_dataset: Iterable[Tuple[np.ndarray, np.ndarray]],
    validator: Optional[Callable] = None,
    preprocessor: Optional[torch.nn.Module] = None,
    # optimization params
    max_epochs: int = 40,
    init_weights: Optional[str] = None,
    lr: float = 1e-3,
    min_lr: float = 1e-5,
    decay_steps: int = 10000,
    weight_decay: float = 0.0,
    # misc params
    device: Optional[str] = None,
    use_amp: bool = False,
    profile: bool = False,
) -> None:
    """Train BBHnet model on in-memory data
    Args:
        architecture:
            A callable which takes as its only input the number
            of ifos, and returns an initialized torch
            Module
        outdir:
            Location to save training artifacts like optimized
            weights, preprocessing objects, and visualizations
        train_dataset:
            An Iterable of (X, y) pairs where X is a batch of training
            data and y is the corresponding targets
        valid_dataset:
            An Iterable of (X, y) pairs where X is a batch of training
            data and y is the corresponding targets
        max_epochs:
            Maximum number of epochs over which to train.
        init_weights:
            Path to weights with which to initialize network. If
            left as `None`, network will be randomly initialized.
            If `init_weights` is a directory, it will be assumed
            that this directory contains a file called `weights.pt`.
        lr:
            Learning rate to use during training.
        min_lr:
            Minimum learning rate to decay to throughout training.
        decay_steps:
            The number of steps over which to decay from lr to min_lr.
        weight_decay:
            Amount of regularization to apply during training.
        early_stop:
            Number of epochs without improvement in validation
            loss before training terminates altogether. Ignored
            if `valid_data is None`.
        device:
            Indicating which device (i.e. cpu or gpu) to run on. Use
            `"cuda"` to use the default GPU available, or `"cuda:{i}`"`,
            where `i` is a valid GPU index on your machine, to specify
            a specific GPU (alternatively, consider setting the environment
            variable `CUDA_VISIBLE_DEVICES=${i}` and using just `"cuda"`
            here).
        profile:
            Whether to generate a tensorboard profile of the
            training step on the first epoch. This will make
            this first epoch slower.
    """

    device = device or "cpu"
    os.makedirs(outdir, exist_ok=True)

    X, y = next(iter(train_dataset))
    if preprocessor is not None:
        X = preprocessor(X)
    with h5py.File(os.path.join(outdir, "batch.h5"), "w") as f:
        f["X"] = X.cpu().numpy()
        f["y"] = y.cpu().numpy()

    logging.info(f"Device: {device}")
    # Creating model, loss function, optimizer and lr scheduler
    logging.info("Building and initializing model")

    # hard coded since we haven't generalized to multiple ifos
    # pull request to generalize dataloader is a WIP
    num_ifos = 2

    model = architecture(num_ifos)
    model.to(device)

    # if we passed a module for preprocessing,
    # include it in the model so that the weights
    # get exported along with everything else
    if preprocessor is not None:
        preprocessor.to(device)
        model = torch.nn.Sequential(preprocessor, model)

    if init_weights is not None:
        # allow us to easily point to the best weights
        # from another run of this same function
        if os.path.isdir(init_weights):
            init_weights = os.path.join(init_weights, "weights.pt")

        logging.debug(
            f"Initializing model weights from checkpoint '{init_weights}'"
        )
        model.load_state_dict(torch.load(init_weights))

    logging.info(model)
    logging.info("Initializing loss and optimizer")

    # TODO: Allow different loss functions or optimizers to be passed?
    criterion = torch.nn.functional.binary_cross_entropy_with_logits
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=decay_steps, eta_min=min_lr
    )

    # start training
    torch.backends.cudnn.benchmark = True

    # start training
    scaler = None
    if use_amp and device.startswith("cuda"):
        scaler = torch.cuda.amp.GradScaler()
    elif use_amp:
        logging.warning("'use_amp' flag set but no cuda device, ignoring")

    logging.info("Beginning training loop")
    for epoch in range(max_epochs):
        if epoch == 0 and profile:
            profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=10),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    os.path.join(outdir, "profile")
                ),
            )
            profiler.start()
        else:
            profiler = None

        logging.info(f"=== Epoch {epoch + 1}/{max_epochs} ===")
        stop = train_for_one_epoch(
            model,
            optimizer,
            criterion,
            train_dataset,
            validator,
            profiler,
            scaler,
            lr_scheduler,
        )
        if stop:
            break
