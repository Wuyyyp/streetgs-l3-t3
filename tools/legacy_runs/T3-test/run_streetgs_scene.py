#!/usr/bin/env python3
"""Train one validated StreetGS scene and record completion and elapsed time."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--sample', type=Path, required=True)
ap.add_argument('--resume', action='store_true', help='Resume an existing checkpoint and retain the previous log.')
args = ap.parse_args()
sample = args.sample.resolve()
record = json.loads((sample / 'input_manifest.json').read_text())
assert not (sample / 'train.complete').exists()
checkpoints = list((sample / 'model/trained_model').glob('iteration_*.pth'))
resume_iteration = max((int(p.stem.split('_')[-1]) for p in checkpoints), default=0)
if args.resume:
    assert resume_iteration > 0, 'No checkpoint to resume'
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    for name in ['train.log', 'state.json']:
        path = sample / name
        if path.exists():
            path.rename(sample / (name + '.before_resume_' + stamp))
    events_path = sample / 'model/static_prune_events.jsonl'
    if events_path.exists():
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        if any(event['iteration'] > resume_iteration for event in events):
            events_path.rename(events_path.with_name(events_path.name + '.before_resume_' + stamp))
            events_path.write_text(''.join(json.dumps(event)+'\n' for event in events
                                          if event['iteration'] <= resume_iteration))
else:
    assert not checkpoints, 'Expected fresh training; use --resume for existing checkpoints'
repo = Path('/data/l3_data_test/street_gaussians-main-local-v2')
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES=str(record['gpu']), PWD=str(repo),
           OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4')
started = time.time()
state = {'method': 'StreetGS', 'stage': 'training', 'gpu': record['gpu'],
         'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'launcher_pid': os.getpid(), 'iterations': 25000, 'resume_iteration': resume_iteration}

def save_state():
    temporary = sample / 'state.json.tmp'
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(sample / 'state.json')

save_state()
try:
    with (sample / 'train.log').open('x') as log:
        process = subprocess.Popen(['/usr/bin/python', str(repo / 'train.py'),
                                    '--config', str(sample / 'train.yaml')],
                                   cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
        state['train_pid'] = process.pid
        save_state()
        rc = process.wait()
        if rc:
            raise RuntimeError(f'Training exited with code {rc}; see train.log')
    for name in ['point_cloud/iteration_25000/point_cloud.ply',
                 'trained_model/iteration_25000.pth']:
        assert (sample / 'model' / name).stat().st_size > 0, name
    state['stage'] = 'complete'
    (sample / 'train.complete').write_text('Training succeeded; final PLY and checkpoint verified.\n')
except Exception as error:
    state.update(stage='failed', error=repr(error))
    raise
finally:
    state.update(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 elapsed_seconds=time.time()-started)
    save_state()
    print(json.dumps(state), flush=True)
