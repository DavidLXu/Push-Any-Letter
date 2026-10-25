"""Online PPO trainer, reproducible evaluations, checkpoints and learning curves."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import argparse
import json
import time
from pathlib import Path


def atomic_json(path, value):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def append_json(path, value):
    with path.open("a") as f:
        f.write(json.dumps(value, allow_nan=False) + "\n")


def evaluate(model, font, batches, seed, out, step):
    import numpy as np
    from .env import LETTERS
    from .vec_env import GroupedVecEnv
    env = GroupedVecEnv(52, 13, font_path=font, curriculum=False)
    records = []
    try:
        for batch in range(batches):
            options = [{"letter": letter, "full": True} for letter in LETTERS]
            env.seed(seed + 52 * batch)
            env.set_options(options)
            obs = env.reset()
            reset_records = env.env_method("get_reset_record")
            finished = np.zeros(52, bool)
            while not finished.all():
                actions, _ = model.predict(obs, deterministic=True)
                obs, _, dones, infos = env.step(actions)
                for i in np.flatnonzero(dones & ~finished):
                    info = infos[i]
                    records.append(dict(case_id=batch*52+int(i), letter=LETTERS[i],
                        seed=seed+batch*52+int(i), success=bool(info["is_success"]),
                        coverage=info["coverage"], best_coverage=info["best_coverage"],
                        steps=info["steps"], **reset_records[i]))
                finished |= dones
    finally:
        env.close()
    records.sort(key=lambda x: x["case_id"])
    summary = dict(step=step, cases=len(records), successes=sum(x["success"] for x in records),
                   mean_coverage=float(np.mean([x["coverage"] for x in records])), seed=seed)
    summary["success_rate"] = summary["successes"] / summary["cases"]
    atomic_json(out / f"evaluation_{step:010d}_{len(records)}.json", dict(summary=summary, episodes=records))
    append_json(out / "evaluations.jsonl", summary)
    return summary


def main():
    import hashlib
    import platform
    import subprocess
    import traceback
    import numpy as np
    import torch
    import stable_baselines3 as sb3
    from collections import deque
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from .vec_env import GroupedVecEnv
    from .plot import plot_run
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--font", required=True)
    parser.add_argument("--num-envs", type=int, default=200)
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--steps", type=int, default=51200000)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2560)
    parser.add_argument("--eval-every", type=int, default=2560000)
    parser.add_argument("--checkpoint-every", type=int, default=1024000)
    parser.add_argument("--curriculum-steps", type=int, default=20000000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.num_envs * args.rollout % args.batch_size:
        parser.error("batch size must divide the complete vector rollout")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    # Exclusive ownership prevents accidentally launching a second learner in this run.
    import fcntl
    lock = (out / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / "config.json").exists():
        raise RuntimeError("Existing run directory: choose a new output; never overwrite a run")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.20, 0)
    source = {}
    root = Path(__file__).resolve().parents[1]
    for folder in ("rl", "push_any_letter"):
        for path in sorted((root/folder).glob("*.py")):
            source[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    config = dict(vars(args), algorithm="PPO", observation="state72", action="relative_waypoint_60px",
        gamma=0.999, gae_lambda=0.95, learning_rate=0.0003, n_epochs=4,
        clip_range=0.2, ent_coef=0.005, target_kl=0.03, hidden=[256,256],
        friction=1.4, damping=0.12, episode_limit=2400, success_coverage=0.95,
        seed_evaluation=314159, seed_final=271828, periodic_cases=52, final_cases=520,
        online_only=True, source_sha256=source, font_sha256=hashlib.sha256(Path(args.font).read_bytes()).hexdigest(),
        torch=torch.__version__, stable_baselines3=sb3.__version__, python=platform.python_version(),
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git_revision=os.environ.get("PUSH_LETTER_CODE_REVISION") or subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip(),
        reward=dict(success=20.0, time=-0.01, control=-0.001, coverage_potential=4.0,
                    centroid_potential=-2.0, contact_gap_potential=-0.5, length_scale_px=156.0),
        curriculum=dict(full_reset_probability=0.25, expansion_steps=args.curriculum_steps),
        physical_gpu_mask=os.environ.get("CUDA_VISIBLE_DEVICES"), cpu_logical_count=os.cpu_count(),
        gpu_name=torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None)
    atomic_json(out/"config.json", config)
    started = time.monotonic()
    env = None
    model = None
    last_step = 0
    class Record(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episodes = deque(maxlen=1000)
            self.last_eval = self.last_checkpoint = 0
            self.last_recorded = -1
        def _on_step(self):
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.episodes.append(dict(success=info["is_success"], full=info["full_start"],
                                              reward=info["episode"]["r"], length=info["episode"]["l"]))
            return True
        def _on_rollout_start(self):
            self.record()
            self.training_env.env_method("set_difficulty", min(1., self.num_timesteps/args.curriculum_steps))
        def record(self):
            nonlocal last_step
            step = int(self.num_timesteps)
            last_step = step
            if step == self.last_recorded:
                return
            self.last_recorded = step
            metrics = {k: float(v) for k, v in self.model.logger.name_to_value.items()
                       if k.startswith("train/") and isinstance(v, (float, int, np.number))}
            if not all(np.isfinite(v) for v in metrics.values()):
                raise FloatingPointError(f"Non-finite optimization metrics at step {step}")
            full = [x for x in self.episodes if x["full"]]
            summary = dict(step=step, elapsed_seconds=time.monotonic()-started,
                fps=step/max(1, time.monotonic()-started), difficulty=min(1,step/args.curriculum_steps),
                episodes_window=len(self.episodes), full_episodes_window=len(full),
                train_success=float(np.mean([x["success"] for x in self.episodes])) if self.episodes else None,
                train_full_success=float(np.mean([x["success"] for x in full])) if full else None,
                **metrics)
            append_json(out/"metrics.jsonl", summary)
            atomic_json(out/"status.json", dict(status="running", pid=os.getpid(), num_envs=args.num_envs,
                workers=args.workers, device=str(self.model.device), total_steps=args.steps, **summary))
            if step and (step-self.last_checkpoint >= args.checkpoint_every or self.last_checkpoint == 0):
                path = out / f"ppo_{step:010d}"
                self.model.save(path)
                self.last_checkpoint = step
            if not args.smoke and step-self.last_eval >= args.eval_every:
                atomic_json(out/"status.json", dict(status="evaluating", step=step, total_steps=args.steps))
                result = evaluate(self.model, args.font, 1, 314159, out, step)
                print("EVALUATION", json.dumps(result), flush=True)
                self.last_eval = step
                plot_run(out)
            if step:
                print("UPDATE", json.dumps(summary), flush=True)
        def _on_training_end(self):
            self.record()
    try:
        atomic_json(out/"status.json", dict(status="starting", pid=os.getpid(), num_envs=args.num_envs, workers=args.workers))
        env = GroupedVecEnv(args.num_envs, args.workers, font_path=args.font)
        model = sb3.PPO("MlpPolicy", env, device=args.device, seed=args.seed,
            n_steps=args.rollout, batch_size=args.batch_size, n_epochs=4,
            learning_rate=3e-4, gamma=.999, gae_lambda=.95, ent_coef=.005,
            clip_range=.2, target_kl=.03, policy_kwargs=dict(net_arch=dict(pi=[256,256], vf=[256,256])),
            verbose=1)
        model.set_logger(configure(str(out), ["csv", "tensorboard"]))
        if not args.smoke:
            atomic_json(out/"status.json", dict(status="baseline_evaluation", step=0, total_steps=args.steps))
            result = evaluate(model, args.font, 1, 314159, out, 0)
            print("BASELINE", json.dumps(result), flush=True)
        model.learn(total_timesteps=args.steps, callback=Record())
        model.save(out/"ppo_final")
        if not args.smoke:
            evaluate(model, args.font, 1, 314159, out, model.num_timesteps)
            evaluate(model, args.font, 10, 271828, out, model.num_timesteps)
        plot_run(out)
        atomic_json(out/"status.json", dict(status="complete", step=model.num_timesteps, total_steps=args.steps,
            elapsed_seconds=time.monotonic()-started))
    except BaseException:
        atomic_json(out/"status.json", dict(status="failed", step=last_step, traceback=traceback.format_exc()))
        raise
    finally:
        if env is not None:
            env.close()
        lock.close()


if __name__ == "__main__":
    main()
