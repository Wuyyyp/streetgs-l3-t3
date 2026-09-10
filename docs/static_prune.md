# StreetGS static background contribution pruning

Default: disabled. Existing configs retain standard StreetGS training.

Add these fields to a scene YAML to train 50000 steps, prune every 5000 steps,
including the final prune at 25000, then keep optimizing without densification:

```yaml
train:
  iterations: 50000
  save_iterations: [1000, 10000, 25000, 50000]
  checkpoint_iterations: [1000, 25000, 50000]
  test_iterations: [1000, 10000, 25000, 50000]
optim:
  static_prune_enabled: true
  static_prune_interval: 5000
  static_prune_cdf_threshold: 0.99
  densify_until_iter: 25000
  position_lr_max_steps: 50000
```

Use the normal train.py entry point. Set static_prune_enabled: false to disable
only this extra contribution pruning; standard StreetGS opacity/size pruning
still follows the original densification settings.

The implementation reuses the tested GSAPro static-background-only algorithm.
It scores every training camera with the optional _ms rasterizer, selects
low-contribution Gaussians using a cumulative importance threshold, and calls
background.prune_points(). Dynamic objects and sky are not pruned by this step.
0.99 is an importance threshold, not a percentage of Gaussian count retained.

The cutoff is inclusive for contribution pruning and exclusive for ordinary
densification. At 25000 the final contribution prune runs before that iteration's
forward pass. Positions, scales, rotations, opacity and appearance continue to
optimize afterward. Resume starts after the loaded checkpoint iteration.

Each event is recorded in model_path/static_prune_events.jsonl, including
iteration, static counts before/after, number removed and elapsed time.

Files:
- train.py: scheduling hook
- lib/config/config.py: disabled-by-default settings
- lib/utils/static_prune.py: backend loading, schedule and pruning
- submodules/diff-gaussian-rasterization_ms: copied GSAPro source; compiled extensions are excluded from this repository. For another Python/PyTorch/CUDA environment,
  rebuild with that environment's python:
  python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization_ms
- tests/test_static_prune_switch.py: CUDA regression

From the repository root:
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python tests/test_static_prune_switch.py

Verified on the current host: disabled default, no _ms import while disabled,
5000..25000 inclusive schedule, no later events, invalid interval rejection,
actual 64-to-61 Gaussian pruning, unchanged dynamic test object, preserved Adam
moment shapes, and normal StreetGS render/backward/optimizer step after pruning.

Independent inclusive end: optim.static_prune_until_iter (default -1 preserves the legacy schedule). With interval=10000, until=45000, densify_until_iter=50000: prune at 10000/20000/30000/40000/45000, then continue densification until the existing <50000 boundary.
