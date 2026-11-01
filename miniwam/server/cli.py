from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from .common import ROOT, completed, read_json, require_server, runtime, stage, verify_files, write_json
from .progress import log, logged_stage
from .protocol import check_window_coverage, episode_limit, renderer_version, storage_estimate


def validate_config(c):
    bank = c["bank"]
    depth = episode_limit(bank)
    navigation = bank.get("family") == "single_goal"
    if navigation:
        moves = bank.get("travel_moves", [])
        if not moves or any(type(m) is not int or not 1 <= m <= 4 for m in moves) or depth < max(moves)+3:
            raise ValueError("single_goal needs travel_moves in [1,4] and max_steps >= max(travel_moves)+3")
    elif depth not in (1,2,3):
        raise ValueError("Legacy curriculum depth must be 1,2,3")
    if not c["horizons"] or any(type(n) is not int or not 1 <= n <= depth for n in c["horizons"]):
        raise ValueError("Prediction horizons must use real primitive transitions within the episode budget")
    if renderer_version(bank) not in ("legacy_v1", "clean_v2"):
        raise ValueError("Invalid renderer version")
    if set(bank["tasks"]) != {"train","val","test"} or min(bank["tasks"].values()) < 3:
        raise ValueError("At least 3 tasks in each of train/val/test")
    planner = c["model"].get("planner", {})
    if set(planner)-{"population","rounds","elites","chunk_size"}:
        raise ValueError("Unknown planner setting")
    if (planner.get("population",64) < 6 or planner.get("rounds",3) < 1 or
            not 1 <= planner.get("elites",8) <= planner.get("population",64) or planner.get("chunk_size",12) < 1):
        raise ValueError("Invalid planner budget")
    for section, keys in (("ppo", ("updates", "num_envs", "rollout_steps", "epochs", "minibatch_size", "eval_every")),
                          ("model", ("updates", "batch_size", "eval_every", "checkpoint_every", "patience", "val_windows", "ddim_steps"))):
        if min(c[section][k] for k in keys) < 1:
            raise ValueError(f"{section} counts must be positive")
    if not 2 <= c["model"]["ddim_steps"] <= 1000:
        raise ValueError("ddim_steps must be in [2,1000]")
    if min(c["collection"]["train_repeats"], c["collection"]["eval_repeats"]) < 7:
        raise ValueError("Need >=7 collection repeats")
    if c["model"].get("motion_weight",0.) < 0:
        raise ValueError("motion_weight must be nonnegative")
    if not 0 <= c["collection"]["epsilon"] <= 1 or not c["seeds"]:
        raise ValueError("Invalid collection epsilon or empty model seeds")


def doctor(device, assets=False):
    import torch
    import jax
    from .model import NanoWAM, autocast
    for vendor in ("nanowm", "xland_minigrid"):
        verify_files(ROOT/"vendor"/vendor, read_json(ROOT/"vendor"/vendor/"SOURCE.json")["files_sha256"])
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 GPU required by the server recipe")
    model = NanoWAM("joint", 1, width=64, layers=2, heads=4, action_width=32).to(device)
    current = torch.zeros(2, 4, 16, 16, device=device)
    with autocast(device):
        loss, _ = model.loss(current, torch.zeros(2,1,dtype=torch.long,device=device), current[:,None],
                             torch.ones(2,1,dtype=torch.bool,device=device), torch.ones(2,1,dtype=torch.bool,device=device))
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError("NanoWM GPU smoke failed")
    report = {**runtime(), "jax_devices": list(map(str, jax.devices())),
              "cuda_backbone_forward_backward": True, "disk_free_gib": shutil.disk_usage(Path.home()).free/2**30,
              "note": "NFS free space is not the personal 100GB quota; inspect du -sh ~ separately"}
    if assets:
        from .xland import fetch_benchmark
        from .vae import load_vae, encode, decode, vae_identity
        import numpy as np
        data = fetch_benchmark(ROOT/"data")
        report["trivial_1m_rows"] = len(data["goals"])
        del data
        vae = load_vae(device)
        latent = encode(vae, np.zeros((1,128,128,3),np.uint8), device)
        report["vae"] = vae_identity()
        report["vae_latent_shape"] = list(latent.shape)
        if tuple(decode(vae, latent).shape) != (1,3,128,128):
            raise RuntimeError("VAE shape check failed")
    write_json(ROOT/"logs/server-doctor.json", report)
    print(__import__("json").dumps(report, ensure_ascii=False, indent=2), flush=True)


def summarize(out):
    out = Path(out)
    config = read_json(out/"resolved_config.json")
    rows, entries, failures = [], [], []
    for mode in ("policy", "video", "joint"):
        for horizon in config["horizons"]:
            for model_seed in config["seeds"]:
                directory = out/f"models/{mode}_n{horizon}_seed{model_seed}"
                if not completed(directory, verify=False):
                    failures.append({"run": directory.name, "reason": "not completed"})
                    continue
                hashes = read_json(directory/"complete.json")["files_sha256"]
                verify_files(directory, {name: value for name, value in hashes.items() if name.endswith("evaluation.json")})
                evaluations = {s: read_json(directory/s/"evaluation.json") for s in ("train","val","test")}
                entry = {"mode": mode, "horizon": horizon, "seed": model_seed,
                         "sr": {s: v["closed_loop"]["sr"] for s,v in evaluations.items()}}
                entries.append(entry)
                control, metrics = evaluations["test"]["closed_loop"], evaluations["test"]["offline"]
                rows.append(f"| {mode} | {horizon} | {model_seed} | {entry['sr']['train']:.1%} | {entry['sr']['val']:.1%} | {control['sr']:.1%} | {control['successes']}/{control['episodes']} |")
                reasons = []
                if min(entry["sr"].values()) < 1.:
                    reasons.append("Inspect failures and by_goal_type; compare PPO and greedy vs sampled decoding")
                if mode != "policy":
                    if metrics["latent_mse"] >= metrics["persistence_latent_mse"]:
                        reasons.append("Dynamics does not beat copying the current frame")
                    if metrics["altered_action_latent_mse"] <= metrics["latent_mse"]:
                        reasons.append("Altered actions are no worse: inspect alignment and action conditioning")
                    cf = evaluations["test"].get("counterfactual", {})
                    if cf and cf["own_alternative_rgb_mse"] >= cf["wrong_pairing_rgb_mse"]:
                        reasons.append("Counterfactual predictions do not match their own real alternatives better than wrong pairings")
                if reasons:
                    failures.append({"run": directory.name, "checks": reasons,
                                     "evaluation": str((directory/"test/evaluation.json").relative_to(out))})
    expected = 3*len(config["horizons"])*len(config["seeds"])
    expert = read_json(out/"expert/evaluation.json") if (out/"expert/evaluation.json").exists() else {}
    controls_path = out/"latents/scorer_controls.json"
    if controls_path.exists():
        for label, control in read_json(controls_path)["controls"].items():
            if control["sr"] < 1.:
                failures.append({"stage": label, "sr": control["sr"],
                                 "checks": ["Visual task cost/localization fails even with true futures; inspect scorer/VAE before attributing failure to NanoWM"],
                                 "evaluation": "latents/scorer_controls.json"})
    perfect = len(entries) == expected and all(min(e["sr"].values()) == 1. for e in entries)
    perfect &= all(expert.get(s, {}).get("sr", 0.) == 1. for s in ("train","val","test"))
    summary = {"scope": "observable-goal controlled curriculum from trivial-1m, not the full hidden-goal benchmark",
               "episode_max_steps": episode_limit(config["bank"]),
               "task_family": config["bank"].get("family", "legacy_near_terminal"),
               "renderer": renderer_version(config["bank"]),
               "all_sr_100": bool(perfect), "models": entries,
               "expert_sr": {s: expert.get(s,{}).get("sr") for s in ("train","val","test")}, "diagnostics": failures}
    write_json(out/"summary.json", summary)
    write_json(out/"diagnosis.json", failures)
    (out/"summary.md").write_text("\n".join([
        "# Server experiment", "", summary["scope"], "", f"Task family: {summary['task_family']}",
        f"Episode primitive-transition budget: {summary['episode_max_steps']}; prediction horizons: {config['horizons']}",
        f"PPO SR: {summary['expert_sr']}", f"All finite-bank SR=100%: {perfect}", "",
        "| Mode | n | Seed | Train SR | Val SR | Test SR | Test successes |",
        "|---|---:|---:|---:|---:|---:|---:|", *rows, "",
        "Checkpoint selection uses validation only. Test is evaluated after training.",
        "Failure checks: diagnosis.json. Full traces and generated videos: models/*/{split}/.", ""]))
    print(f"Saved {out/'summary.md'}; all_sr_100={perfect}", flush=True)
    return summary


def pipeline(args, device):
    from .xland import make_banks
    from .expert import train_expert
    from .dataset import collect
    from .vae import cache_latents
    from .vae import load_vae
    from .controls import scorer_controls
    from .training import train
    config = read_json(args.config)
    validate_config(config)
    out = stage(args.out, {"pipeline": config}, args.resume)
    write_json(out/"resolved_config.json", config)
    write_json(out/"runtime.json", runtime())
    log("РАЗМЕР ДАННЫХ ДО ЗАПУСКА", **storage_estimate(config),
        note="dense RGB + latents only; additionally allow for checkpoints, environments and caches; personal quota 100GB")
    with logged_stage("1/6: подготовка и аудит банков XLand", max_steps=episode_limit(config["bank"]),
                      family=config["bank"].get("family", "legacy_near_terminal"), horizons=config["horizons"]):
        banks = make_banks(config["bank"], ROOT/"data", out/"banks", args.resume)
    with logged_stage("2/6: обучение PPO по состояниям"):
        expert = train_expert(banks, config["ppo"], out/"expert", device, args.resume)
    if not read_json(out/"expert/gate.json")["passed"]:
        summarize(out)
        raise RuntimeError(f"PPO train/val SR below 100%; inspect {out/'expert/evaluation.json'}. No WAM training started.")
    log("ПРОВЕРКА PPO ПРОЙДЕНА", train_val_sr="100%")
    with logged_stage("3/6: сбор и replay RGB-датасета"):
        dataset = collect(banks, expert, config["collection"], out/"dataset", device, args.resume)
        data_manifest = read_json(dataset/"manifest.json")
        check_window_coverage(data_manifest,config["horizons"])
        for split, info in data_manifest["splits"].items():
            log("ПОКРЫТИЕ ГОРИЗОНТОВ", split=split, **info["diversity"])
    with logged_stage("4/6: кодирование frozen VAE и проверка реконструкций"):
        cache = cache_latents(dataset, out/"latents", device, args.resume)
    with logged_stage("5/6: диагностический визуальный контроль"):
        if not (cache/"scorer_controls.json").exists():
            scorer_controls(banks, load_vae(device), device, cache)
        else:
            log("ПРОПУСК: визуальный контроль уже сохранен", path=cache/"scorer_controls.json")
    model_count = 3*len(config["horizons"])*len(config["seeds"])
    model_index = 0
    for mode in ("policy", "video", "joint"):
        for horizon in config["horizons"]:
            for model_seed in config["seeds"]:
                model_index += 1
                with logged_stage(f"6/6: модель {model_index}/{model_count}", mode=mode, n=horizon, seed=model_seed):
                    train(banks, dataset, cache, config["model"], mode, horizon, model_seed,
                          out/f"models/{mode}_n{horizon}_seed{model_seed}", device, args.resume)
                summarize(out)
    return summarize(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Server-only: XLand trivial-1m -> state PPO -> RGB -> NanoWAM")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("doctor")
    check.add_argument("--assets", action="store_true", help="Also download/verify trivial-1m and the frozen SD-VAE")
    run = sub.add_parser("pipeline")
    run.add_argument("--config", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--require-perfect", action="store_true")
    report = sub.add_parser("report")
    report.add_argument("--out", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--out", required=True)
    diagnose = sub.add_parser("diagnose-video", help="Frozen one-step video weights: all six futures and two task costs, validation only")
    diagnose.add_argument("--run", required=True)
    diagnose.add_argument("--out", required=True)
    diagnose.add_argument("--limit",type=int)
    diagnose.add_argument("--ddim-steps",type=int,default=20)
    diagnose.add_argument("--resume",action="store_true")
    ablation = sub.add_parser("train-model", help="Reuse collected data, optionally restrict training tasks for a learning curve")
    ablation.add_argument("--run", required=True)
    ablation.add_argument("--out", required=True)
    ablation.add_argument("--mode", choices=("policy","video","joint"), required=True)
    ablation.add_argument("--horizon", type=int, required=True)
    ablation.add_argument("--train-tasks", type=int, required=True)
    ablation.add_argument("--seed", type=int, default=3407)
    ablation.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "report":
        summarize(args.out)
        return 0
    device = require_server()
    if hasattr(args, "out"):
        destination = Path(args.out).expanduser().resolve()
        if not destination.is_relative_to(Path.home().resolve()):
            raise RuntimeError("Results must be inside your persistent server HOME")
        args.out = str(destination)
    if args.command == "doctor":
        with logged_stage("Doctor: CUDA, BF16, NanoWM и ресурсы"):
            doctor(device, args.assets)
    elif args.command == "audit":
        from .dataset import audit as audit_dataset
        from .xland import audit_banks
        print(audit_banks(Path(args.out)/"banks"))
        print(audit_dataset(Path(args.out)/"dataset", Path(args.out)/"banks"))
    elif args.command == "diagnose-video":
        from .diagnostics import diagnose_video
        diagnose_video(args.run,args.out,device,args.limit,args.ddim_steps,args.resume)
    elif args.command == "train-model":
        from .training import train
        run = Path(args.run)
        config = read_json(run/"resolved_config.json")
        validate_config(config)
        if args.horizon not in config["horizons"]:
            raise ValueError("Use a horizon declared in the source experiment")
        model_config = {**config["model"], "train_task_limit": args.train_tasks}
        train(run/"banks",run/"dataset",run/"latents",model_config,args.mode,args.horizon,
              args.seed,args.out,device,args.resume)
    else:
        result = pipeline(args, device)
        return 2 if args.require_perfect and not result["all_sr_100"] else 0
    return 0
