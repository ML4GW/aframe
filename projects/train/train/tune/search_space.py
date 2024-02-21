from ray import tune

space = {
    "model.learning_rate": tune.loguniform(1e-4, 1e-2),
    "model.pct_lr_ramp": tune.uniform(0.05, 0.5),
    "data.swap_prob": tune.uniform(0.0, 0.15),
    "data.mute_prob": tune.uniform(0.0, 0.15),
    "data.waveform_prob": tune.uniform(0.2, 0.4),
}
