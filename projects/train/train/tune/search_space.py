from ray import tune

space = {
    "model.learning_rate": tune.loguniform(1e-4, 1e-1),
    "model.pct_lr_ramp": tune.uniform(0, 1),
}
