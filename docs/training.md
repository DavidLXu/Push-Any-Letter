# Training and evaluation

Install the simulator on macOS for interaction and replay. The full comparison is designed for a Linux server with four NVIDIA GPUs. Use Python 3.10+; the reference runs use Python 3.12, PyTorch 2.8.0+cu128 and RTX 5090 GPUs. Install the CUDA build of PyTorch appropriate for your GPU before installing the train extra.

```bash
python -m pip install -e '.[train,test]'
python examples/download_dataset.py --output datasets/push-any-letter
python -m training.data datasets/push-any-letter datasets/state --font /path/to/Arial-Bold.ttf
```

Preparation verifies 6240 unique state/action sequences, groups of ten, friction, reset goals and validation flags. It writes compact memory-mapped state/action arrays and generates the fixed test cases. Run the downloader first to verify the full manifest and checksums. Keep data preparation outputs in their original location; generated manifests refer to local files.

## State policies

Each observation contains 62 features: six normalized position coordinates, sine and cosine of object/goal angles, and a 52-way letter identity. Policies receive the two most recent observations, repeating the first observation at episode start. Actions are normalized as `(xy - 320) / 292`.

| Policy | Training objective | Action horizon / executed actions |
| --- | --- | --- |
| Vanilla BC | MLP regression, MSE | 1 / 1 |
| Diffusion Policy | Conditional 1D U-Net, epsilon MSE, 100 noise steps, 10 DDIM sampling steps | 16 / 4 |
| Flow Matching | Same U-Net, linear Gaussian-to-action path, velocity MSE, 10 Euler steps | 16 / 4 |
| ACT-style | Conditional VAE transformer, L1 + 10 × KL, latent 32, width 128 | 16 / 4 |

ACT uses zero latent at inference and no temporal aggregation. All models use AdamW, peak learning rate 1e-4 with warmup/cosine decay, weight decay 1e-4, gradient clipping at 1, parameter EMA 0.995, batch 256, seed 42 and 100,000 optimizer updates. Loss scales differ across objectives and should not be ranked against each other.

Run each command in its own terminal or scheduler job:

```bash
CUDA_VISIBLE_DEVICES=0 python -m training.train --algo bc --data datasets/state --output runs/state/bc
CUDA_VISIBLE_DEVICES=1 python -m training.train --algo diffusion --data datasets/state --output runs/state/diffusion
CUDA_VISIBLE_DEVICES=2 python -m training.train --algo flow --data datasets/state --output runs/state/flow
CUDA_VISIBLE_DEVICES=3 python -m training.train --algo act --data datasets/state --output runs/state/act
```

Each process automatically evaluates its final EMA policy on all 520 test cases. `--device cpu` permits a small state-policy smoke run. Resume by repeating the identical command; changing configuration in an existing output directory is rejected. Checkpoint loading uses PyTorch pickle, so load only checkpoints you trust.

## Image policies

Image observations contain two 96 × 96 RGB frames and 58 auxiliary features per frame: pusher xy, goal xy/sin/cos and letter identity. Object pose and velocity are excluded from the policy input. An independently trained residual CNN with spatial softmax maps each image to 256 features, then feeds the same four policy heads. RGB is normalized to [0, 1]. There is no pretrained encoder or image augmentation.

```bash
python -m training.image_data datasets/state datasets/images --workers 8
CUDA_VISIBLE_DEVICES=0 python -m training.image_train --algo bc --data datasets/state --image-cache datasets/images --output runs/image/bc
CUDA_VISIBLE_DEVICES=1 python -m training.image_train --algo diffusion --data datasets/state --image-cache datasets/images --output runs/image/diffusion
CUDA_VISIBLE_DEVICES=2 python -m training.image_train --algo flow --data datasets/state --image-cache datasets/images --output runs/image/flow
CUDA_VISIBLE_DEVICES=3 python -m training.image_train --algo act --data datasets/state --image-cache datasets/images --output runs/image/act
```

The uncompressed memory-mapped image cache needs about **131 GB** (122 GiB), plus a 10 GiB free-space margin. Effective batch size remains 256 using four microbatches of 64. Start these after the state suite finishes if reusing the same GPUs.

## Loss and success curves

Loss is logged every 100 updates to `metrics.jsonl`. Immutable EMA snapshots are saved every 10,000 updates, plus the first available and final states. The reference state run enabled snapshots after a restart, so its earlier checkpoints are unavailable; report those curves starting at their first actual checkpoint.

After all four output directories exist, run the following alongside training on a four-GPU Linux server:

```bash
python -m training.curves --data datasets/state --runs runs/state
# Reuse the same case plan for the later image comparison:
cp runs/state/curve_plan.json runs/image/curve_plan.json
python -m training.curves --data datasets/state --runs runs/image --image
```

The curve service maps algorithms to physical GPU indices 0–3 and evaluates each snapshot in the background. It is separate from the four optimizer processes. It requires NVIDIA tools and is not a macOS plotting service. To only regenerate existing plots on any machine:

```bash
python -m training.curves --data datasets/state --runs runs/state --plot-only
```

`curves/` contains PNG/PDF loss plots, training/test success plots, and CSV/JSON measurements. Periodic evaluations use fixed, paired subsets: 52 training starts and 52 unseen-angle starts. The final `evaluation/summary.json` covers all 520 test cases; do not confuse this result with the 52-case monitoring curve. Success requires at least 95% intersection-over-object-area coverage within 2,400 control steps. Symmetric letters can satisfy this metric at equivalent orientations.

All 6,240 demonstrations belong to training. There is no validation split. The test generator uses seed 20261027 to sample ten angles at least 3° from the 30° training grid and one another; each letter/angle has one independently sampled start. The final 100,000-update checkpoint is fixed in advance. Results from one training seed do not quantify training-seed variation, and the benchmark does not test goal translation or font transfer.
