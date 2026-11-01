"""Sequential expert -> RGB dataset -> all objectives/horizons -> final test."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from .data import CollectionConfig, collect
from .env import EnvConfig, bank_manifest
from .evaluate import closed_loop, evaluate_model
from .model import MODES
from .ppo import PPOConfig, load_expert, train_expert
from .train import ModelConfig, train_model
from .utils import prepare_run, runtime, write_json


def resolve_config(raw):
    allowed = {"environment", "ppo", "collection", "model", "horizons", "seeds"}
    if set(raw)-allowed:
        raise ValueError(f"Unknown pipeline keys: {set(raw)-allowed}")
    if set(raw.get("model", {})) & {"mode", "horizon", "seed"}:
        raise ValueError("Set horizons/seeds at pipeline top level; all three modes always run")
    result = {"environment": asdict(EnvConfig(**raw.get("environment", {}))),
        "ppo": asdict(PPOConfig(**raw.get("ppo", {}))),
        "collection": asdict(CollectionConfig(**raw.get("collection", {}))),
        "model": asdict(ModelConfig(**raw.get("model", {}))),
        "horizons": raw.get("horizons", [1, 2, 3]), "seeds": raw.get("seeds", [3407])}
    for key in ("horizons", "seeds"):
        values = result[key]
        if not values or any(type(v) is not int or v < (1 if key == "horizons" else 0) for v in values):
            raise ValueError(f"{key} must be a nonempty list of valid integers")
        if len(set(values)) != len(values):
            raise ValueError(f"{key} contains duplicates")
    return result


def run_pipeline(config_path, out, device, resume=False):
    raw = json.loads(Path(config_path).read_text())
    config = resolve_config(raw)
    out = prepare_run(out, config, resume)
    write_json(out/"resolved_config.json", config)
    env_config = EnvConfig(**config["environment"])
    write_json(out/"task_banks.json", bank_manifest(env_config))
    expert_path = train_expert(env_config, PPOConfig(**config["ppo"]), out/"expert", device, resume)
    expert, _ = load_expert(expert_path, device)
    expert_results = {split: closed_loop(expert, env_config, split, device, state_policy=True)
                      for split in ("train", "val")}
    write_json(out/"expert"/"gate.json", expert_results)
    # Gate depends only on train/val. Test is reserved for the final report.
    if any(result["success_rate"] < 1 for result in expert_results.values()):
        raise RuntimeError("PPO expert has train/val SR < 100%; see expert/gate.json. Increase PPO budget in a NEW run.")
    dataset = collect(expert_path, CollectionConfig(**config["collection"]), out/"dataset", device, resume)
    # Train every model before looking at their test SR.
    jobs = []
    for seed in config["seeds"]:
        for horizon in config["horizons"]:
            for mode in MODES:
                model_config = ModelConfig(**{**config["model"], "mode": mode, "horizon": horizon, "seed": seed})
                directory = out/"models"/f"{mode}_n{horizon}_seed{seed}"
                checkpoint = train_model(dataset, model_config, directory, device, resume)
                jobs.append((model_config, directory, checkpoint))
    results = []
    for model_config, directory, checkpoint in jobs:
        evaluated = evaluate_model(checkpoint, dataset, "test", directory/"test", device)
        evaluated["seed"] = model_config.seed
        results.append(evaluated)
    expert_results["test"] = closed_loop(expert, env_config, "test", device, state_policy=True)
    random_result = closed_loop(None, env_config, "test", device)
    perfect = (expert_results["test"]["success_rate"] == 1
               and all(row["closed_loop"]["success_rate"] == 1 for row in results))
    summary = {"environment": asdict(env_config), "expert": expert_results, "random": random_result,
        "models": results, "all_test_sr_100": perfect, "runtime": runtime(device),
        "interpretation": "Exhaustive held-out goal positions on this finite board, not general 100% success."}
    write_json(out/"summary.json", summary)
    rows = ["# mini-WAM: результаты", "", f"Глубина: {env_config.depth} действий. Поле: {env_config.size}×{env_config.size}.", "",
        f"PPO test SR: {expert_results['test']['success_rate']:.1%}; random test SR: {random_result['success_rate']:.1%}.", "",
        "| Режим | n | Seed | Test SR | Успехи / задачи | Foreground accuracy |",
        "|---|---:|---:|---:|---:|---:|"]
    for result in results:
        evaluation = result["closed_loop"]
        accuracy = result["offline"].get("foreground_cell_accuracy")
        accuracy_text = "—" if accuracy is None else f"{accuracy:.3%}"
        rows.append(f"| {result['mode']} | {result['horizon']} | {result['seed']} | {evaluation['success_rate']:.1%} | "
                    f"{evaluation['successes']} / {evaluation['episodes']} | {accuracy_text} |")
    rows += ["", "SR измерен в настоящей среде, с новым наблюдением после каждого действия.",
             "Выбор checkpoint: только validation. Идентичные входные кадры между splits исключены.",
             "n больше длины эпизода не создаёт дополнительных targets: padding маскируется.", "",
             "Подробные ошибки, направления и траектории: models/*/test/evaluation.json.",
             "Данные и примеры GIF: dataset/. Предсказания: models/*/test/forecast*.gif.", ""]
    (out/"summary.md").write_text("\n".join(rows))
    print(f"Saved {out/'summary.md'}; all test SR=100%: {perfect}", flush=True)
    return summary
