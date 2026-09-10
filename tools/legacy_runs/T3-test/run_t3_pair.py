#!/usr/bin/env python3
"""Convert or train a T3 before/after pair, with pruning checks on completion."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time
import cv2

ROOT = Path('/data/l3data-reconstruction-bingxing/tem-test/streetGS/T3-test')
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--clip', required=True)
ap.add_argument('--stage', choices=['convert', 'train'], required=True)
ap.add_argument('--resume', action='store_true', help='Resume existing incomplete training checkpoints.')
args = ap.parse_args()
started = time.time()
status_path = ROOT/'logs'/f'{args.clip}.{args.stage}.state.json'
state = dict(clip=args.clip, stage=args.stage, status='running', pid=os.getpid(),
             started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())


def save():
    tmp = status_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(status_path)


def ply_counts(path):
    counts = {}
    with path.open('rb') as f:
        for line in f:
            line = line.decode().strip()
            if line.startswith('element '):
                _, name, count = line.split()
                counts[name] = int(count)
            if line == 'end_header':
                return counts
    raise ValueError(f'Invalid PLY: {path}')


save()
try:
    if args.stage == 'train':
        assert (ROOT/'checks/training.ready.json').is_file(), 'T3 smoke validation required'
        before, after = [ROOT/v/args.clip for v in ['before', 'after']]
        for sample in [before, after]:
            assert (sample/'converted/conversion.complete.json').is_file()
        if not (before/'sky.complete').exists():
            state['variant'] = 'shared_sky_masks'
            save()
            da = Path('/data/mnt/yswang-wan22/code/Depth-Anything-3')
            record = json.loads((before/'input_manifest.json').read_text())
            env = dict(os.environ, PYTHONPATH=str(da/'src'), CUDA_VISIBLE_DEVICES=str(record['gpu']),
                       OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
            with (before/'sky.log').open('x') as log:
                subprocess.run([str(da/'env/bin/python'), str(da/'scripts/infer_sky.py'),
                    '--root', str(before/'converted'), '--out', str(before/'converted')],
                    env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            images = sorted((before/'converted/images').glob('*.jpg'))
            assert len(images) == 707
            for image in images:
                mask = cv2.imread(str(before/'converted/sky_mask'/image.name), cv2.IMREAD_GRAYSCALE)
                assert mask is not None and mask.shape == (900, 1600), image
            (before/'sky.complete').write_text('All 707 sky masks verified.\n')
        sky_link = after/'converted/sky_mask'
        if sky_link.is_symlink():
            assert sky_link.resolve() == (before/'converted/sky_mask').resolve()
        else:
            sky_link.symlink_to(os.path.relpath(before/'converted/sky_mask', after/'converted'))
        (after/'sky.complete').write_text('Shared sky masks with before; identical images.\n')
    for variant in ['before', 'after']:
        sample = ROOT/variant/args.clip
        record = json.loads((sample/'input_manifest.json').read_text())
        state['variant'] = variant
        save()
        if args.stage == 'convert':
            command = ['/usr/bin/python', '-u', str(ROOT/'scripts/convert_t3_streetgs.py'),
                '--restored-root', record['restored_root'], '--raw-root', record['raw_root'],
                '--output', str(sample/'converted'), '--variant', variant,
                '--frame-start', str(record['frame_start']), '--frame-count', '101', '--num-workers', '4']
            if variant == 'after':
                command += ['--shared-images', str(ROOT/'before'/args.clip/'converted/images')]
            env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            with (sample/'convert.log').open('x') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            assert (sample/'converted/conversion.complete.json').is_file()
        else:
            assert (ROOT/'checks/training.ready.json').is_file(), 'T3 smoke validation required'
            assert (sample/'converted/conversion.complete.json').is_file()
            if not (sample/'train.complete').exists():
                command = ['/usr/bin/python', str(ROOT/'scripts/run_streetgs_scene.py'), '--sample', str(sample)]
                if args.resume and list((sample/'model/trained_model').glob('iteration_*.pth')):
                    command.append('--resume')
                subprocess.run(command, check=True)
            model = sample/'model'
            events = [json.loads(line) for line in (model/'static_prune_events.jsonl').read_text().splitlines()]
            assert [e['iteration'] for e in events] == [2500, 5000, 7500, 10000, 12500]
            middle = ply_counts(model/'point_cloud/iteration_12500/point_cloud.ply')
            final = ply_counts(model/'point_cloud/iteration_25000/point_cloud.ply')
            assert middle == final, 'Point counts changed after final scheduled pruning'
            assert final['vertex_background'] == events[-1]['static_after']
            (sample/'pruning.validation.json').write_text(json.dumps(
                dict(schedule=[e['iteration'] for e in events], counts_12500=middle,
                     counts_25000=final, point_counts_locked=True), indent=2))
    state['status'] = 'complete'
except Exception as error:
    state.update(status='failed', error=repr(error))
    raise
finally:
    state.update(elapsed_seconds=time.time()-started,
                 finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save()
    print(json.dumps(state), flush=True)
