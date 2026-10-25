# Online reinforcement learning

One goal-conditioned PPO policy controls all 52 ASCII letters. Training collects new transitions directly from the simulator; it does not load demonstrations or initialize from an imitation policy. Dataset labels such as V1, V2 and V3 therefore do not identify this experiment. Reproduction still requires a code revision, font hash, environment and reward settings, compute budget, and random seed.

This is an initial PPO baseline. A running training process is not evidence that all letters have been solved. The first run uses seed 42 and a budget of **51,200,000 environment transitions**, aggregated across **200 live environments**. It has 50 CPU worker processes, four environments per worker, and one GPU learner. Physics remains on CPU.

## First-run outcome

The initial run was stopped early on 26 October 2026 (UTC+8), with 28,339,200 transitions last logged. All 12 fixed 52-case monitoring evaluations, from initialization through 28,160,000 transitions, had zero successes. The separate 520-case final test was not run. The latest saved weights are from 27,673,600 transitions; checkpoints, logs and plots are retained.

A stationary-policy control over the 25% full / 75% curriculum reset mixture increased its undiscounted episode return from −27.04 to −19.55 as difficulty increased, without succeeding. This exposes a comparability problem in the training-return curve; it does not prove that the discounted PPO objective rewards moving away. Action standard deviation also contracted to about 0.06, and difficulty increased on a fixed schedule despite poor success. The settings below reproduce this unsuccessful baseline, not a validated solution.

## Task and state

Goals stay at (320, 320). At every reset the letter is sampled uniformly from A–Z/a–z and the goal angle continuously from [0, 2π). Full-distribution resets sample the object position across its feasible arena region, its angle uniformly, and a separated pusher position. The complete rotated glyph stays at least 48 pixels inside the inner wall; the pusher starts at least 20 pixels from the glyph surface and inside [44, 596]².

The 72-dimensional state contains a 52-way letter one-hot vector, pusher/object/goal positions (6), sine/cosine object and goal angles (4), pusher and object linear velocities plus object angular velocity (5), two relative position vectors (4), and current coverage (1). Positions and velocities are scaled. Velocities and coverage are privileged simulator state; this is not an image-policy experiment. Coordinates and dynamics are unchanged from `PushLetterEnv`.

The policy emits two values in [−1, 1]. They specify a waypoint up to 60 pixels from the current pusher position on each axis; the original force-limited servo clips the absolute target to the arena. Actions run at 20 Hz, physics at 200 Hz. Object contact friction is 1.4, pusher friction 0.6, space damping 0.12. An episode succeeds at 95% coverage and times out after 2,400 control steps.

## One reward for every letter

Let C be intersection area divided by glyph area, d the object-centroid distance to the goal divided by 156 pixels, and g the nonnegative pusher-to-glyph surface gap (after subtracting pusher radius), also divided by 156. Define:

```text
Φ(s) = 4 C(s) − 2 d(s) − 0.5 g(s)
r = 20 · success − 0.01 − 0.001 ||a||² + 0.999 Φ(s_next) − Φ(s)
```

At successful termination the absorbing next-state potential is zero. At a time-limit truncation it is the actual terminal state's potential; the vector environment preserves `terminal_observation` and `TimeLimit.truncated` so PPO bootstraps correctly. The movement penalty uses the clipped action actually sent to the environment. The shaping discount matches PPO's gamma (0.999).

The centroid term supplies direction before the glyph overlaps its target. Coverage guides orientation and final alignment. The contact-gap term helps the pusher approach the object. No raw angle penalty or letter-specific coefficients are used: geometrically equivalent orientations have the same reward. This also means the task judges visual shape alignment, not a hidden orientation label. Potential shaping telescopes under discounting; this property applies to the shaping term, not to a claim that the time and control costs leave the original sparse task unchanged.

## Exploration and evaluation

From the first update, 25% of resets use the full start distribution. The other 75% begin nearer the target with the pusher near a randomly sampled boundary location. Over the first 20 million transitions their position radius expands from 20 to 220 pixels, angle offset from ±15° to ±180°, and pusher clearance increases. Every curriculum level includes every letter and continuous goal angles. The mixture remains 25% full resets after expansion; curriculum episodes are not counted as held-out performance.

Evaluation uses full-distribution starts throughout, no curriculum, and deterministic policy actions. A fixed seed (314159) defines 52 monitoring tasks, one per letter, evaluated at initialization and every 2,560,000 transitions. At the end, 520 separately seeded tasks (seed 271828; ten per letter) are evaluated in addition to the monitoring set. Their starts and angles are generated independently from online training; RL has no finite training-angle grid defining a seen/unseen split. These are new-state evaluations on familiar letters, not generalization to new letters or fonts. These tasks are also different from the IL evaluation set, so the two aggregate scores are not a matched-case comparison.

Training success is logged over the latest 1,000 completed episodes, with the full-reset subset reported separately. The curriculum mixture and stochastic training actions differ from evaluation. PPO policy loss, value loss, approximate KL, entropy, and explained variance are recorded along with success curves. The 52-case monitoring estimate is coarse; final 520-case outcomes and per-letter records are retained. No early result is claimed here.

## Run

Python 3.10+ with a compatible PyTorch installation:

```bash
python -m pip install -e '.[rl]'
CUDA_VISIBLE_DEVICES=0 python -m rl.train \
  --font /path/to/Arial-Bold.ttf \
  --output runs/ppo_online_52letters_seed42 \
  --num-envs 200 --workers 50
```

Use a new output directory for each run; the trainer takes an exclusive lock and refuses to overwrite an existing configuration. To test the entire 200-environment pipeline for two updates:

```bash
CUDA_VISIBLE_DEVICES=0 python -m rl.train \
  --font /path/to/Arial-Bold.ttf --output runs/ppo_smoke_200 \
  --num-envs 200 --workers 50 --steps 51200 --smoke
```

A smaller CPU smoke run on macOS can use `--device cpu --num-envs 4 --workers 2 --rollout 32 --batch-size 64 --steps 256 --smoke`. The full run is intended for a many-core server. The first deployment uses PyTorch 2.7.0+cu128 and Stable-Baselines3 2.7.0 on an RTX 5090. GPU visibility should refer to an available device; the launcher does not reserve hardware from other users.

PPO uses separate two-layer 256-unit actor/critic networks, 128 steps per environment per rollout (25,600 transitions), minibatches of 2,560, four optimization epochs, learning rate 3e−4, clip range 0.2, GAE λ=0.95, entropy coefficient 0.005, and target KL 0.03. No learned dynamics model or planner is used.

Artifacts in the run directory:

- `config.json`: settings, library versions, font and source hashes.
- `status.json`, `metrics.jsonl`, `progress.csv`, TensorBoard events: status and optimization history.
- `ppo_*.zip`, `ppo_final.zip`: initial-update, periodic, and final policy checkpoints.
- `evaluations.jsonl`, `evaluation_*.json`: success summaries and each case's reset state/outcome.
- `curves.png`, `curves.pdf`: plots regenerated after monitoring evaluations and completion.

Tests cover all 52 glyphs' reset safety, seeded dynamics, potential telescoping and the absorbing-success boundary, grouped/serial rollout equivalence, and time-limit bootstrap observations.

Sources: [PPO, Schulman et al. (2017)](https://arxiv.org/abs/1707.06347), [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/v2.7.0/modules/ppo.html), [potential-based reward shaping, Ng et al. (1999)](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf).

## Save live reward and success plots

The learner continuously records shaped episode return in `progress.csv` and TensorBoard, and rolling success in `metrics.jsonl`. An independent CPU watcher can refresh persistent curves every 30 seconds while training continues:

```bash
python -m rl.live_curves --run runs/ppo_online_52letters_seed42 --interval 30
```

This watcher only reads training logs. It saves `live_curves/reward_success.png` and `.pdf`, plus `reward_history.csv` and `success_history.csv`. Raw learner logs retain the full history. Reward is the mean shaped return over the latest 100 completed episodes; training success uses the latest 1,000, with full-reset episodes separately labeled. Only completed independent evaluations are plotted. Curriculum progress can change training returns, so reward is not a substitute for held-out success. `curve_service_status.json` reports freshness or errors, a lock prevents duplicate watchers, and the watcher renders once more then exits when training completes or fails.
