from ray import tune

space = {
    "model.learning_rate": tune.loguniform(1e-4, 1e-1),
    "model.pct_lr_ramp": tune.uniform(0.1, 0.8),
    "data.kernel_length": tune.uniform(1.5, 2.5),
    "data.psd_length": tune.uniform(8, 10),
    "data.swap_prob": tune.uniform(0.0, 0.15),
    "data.mute_prob": tune.uniform(0.0, 0.15),
    "data.waveform_prob": tune.uniform(0.2, 0.7),
}
