"""Separate semantic objectives, episode budgets and prediction horizons."""


def episode_limit(config):
    return int(config["max_steps"] if config.get("family") == "single_goal" else config["depth"])


def renderer_version(config):
    return config.get("renderer", "clean_v2" if config.get("family") == "single_goal" else "legacy_v1")


def storage_estimate(config):
    """Dense RGB + FP16 latents, before checkpoints, environments and caches."""
    frames = episode_limit(config["bank"])+1
    episodes = sum(n*config["collection"]["train_repeats" if s == "train" else "eval_repeats"]
                   for s, n in config["bank"]["tasks"].items())
    rgb = episodes*frames*128*128*3
    latent = episodes*frames*4*16*16*2
    return {"episodes": episodes, "rgb_gib": rgb/2**30, "latent_gib": latent/2**30,
            "data_gib": (rgb+latent)/2**30}


def check_window_coverage(manifest, horizons):
    for split, info in manifest["splits"].items():
        for horizon in horizons:
            for key in ("complete_windows_by_horizon", "expert_complete_windows_by_horizon"):
                if info["diversity"][key].get(str(horizon),0) < 1:
                    raise ValueError(f"{split}: no real {key} for n={horizon}; collect longer/diverse episodes, not padding")
