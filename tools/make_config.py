"""Generate a seven-camera scene YAML without changing the original presets."""
import argparse
from pathlib import Path
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--iterations', type=int, default=100000)
    p.add_argument('--frames', type=int, default=101)
    p.add_argument('--densify-until', type=int, default=50000)
    p.add_argument('--prune-interval', type=int, default=0, help='0 disables contribution pruning')
    p.add_argument('--prune-until', type=int, default=45000)
    a = p.parse_args()
    if a.frames < 2 or not 0 <= a.densify_until <= a.iterations or a.prune_interval < 0:
        p.error('Invalid frame count, densification end or pruning interval')
    if a.prune_interval and not 0 < a.prune_until <= a.densify_until:
        p.error('Pruning end must be positive and no later than densification end')
    root = Path(__file__).resolve().parents[1]
    c = yaml.safe_load((root/'configs/example/l3_t3_7view_100k.yaml').read_text())
    c.update(source_path=str(a.source.resolve()), model_path=str(a.model.resolve()),
             record_dir=str(a.model.resolve()/'record'), exp_name=a.model.name,
             gpus=[a.gpu], data_device='cpu')
    c['data'].update(cameras=[0,1,2,3,4,9,10], selected_frames=[0,a.frames-1])
    c['train'].update(iterations=a.iterations, save_iterations=[a.iterations],
                      test_iterations=[a.iterations], checkpoint_iterations=[a.iterations])
    c['optim'].update(position_lr_max_steps=a.iterations, densify_until_iter=a.densify_until,
                      static_prune_enabled=bool(a.prune_interval),
                      static_prune_interval=a.prune_interval or 5000,
                      static_prune_until_iter=a.prune_until if a.prune_interval else -1,
                      static_prune_cdf_threshold=0.99)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        yaml.safe_dump(c, f, sort_keys=False)
    print(a.output)


if __name__ == '__main__':
    main()
