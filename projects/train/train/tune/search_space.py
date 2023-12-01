from ray import tune

space = {
    "model.learning_rate": tune.loguniform(1e-4, 1e-1),
    "data.swap_frac": tune.uniform(0, 0.2),
    "data.mute_frac": tune.uniform(0, 0.2),
}
