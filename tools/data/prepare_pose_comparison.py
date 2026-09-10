#!/usr/bin/env python3
"""Create a before/after ego-pose experiment, reusing identical non-pose inputs."""
import argparse
import copy
import json
from pathlib import Path
import pickle

import numpy as np
import yaml

BEFORE = 'ego2global_transformation_matrix_camera_reference_offset'
AFTER = BEFORE + '_optimized'


class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(module.replace('numpy._core', 'numpy.core'), name)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--clip', required=True)
    ap.add_argument('--gpu', type=int, required=True)
    args = ap.parse_args()
    after = args.base / 'after' / args.clip
    original = after / 'converted'
    manifest = json.loads((original / 'conversion.complete.json').read_text())
    assert manifest['pose_field'] == AFTER
    assert (after / 'sky.complete').is_file()
    with Path(manifest['source_pkl']).open('rb') as handle:
        infos = NumpyCompatUnpickler(handle).load()['infos']
    comparison = args.base / 'comparison'
    before = args.base / 'before' / args.clip
    before.mkdir(parents=True, exist_ok=False)
    comparison.mkdir(parents=True, exist_ok=True)
    converted = before / 'converted'
    converted.mkdir()
    shared = []
    for source in original.iterdir():
        if source.name in ('ego_pose', 'conversion.complete.json'):
            continue
        target = converted / source.name
        target.symlink_to(source, target_is_directory=source.is_dir())
        assert target.samefile(source)
        shared.append(source.name)
    (converted / 'ego_pose').mkdir()
    translation_deltas, rotation_deltas = [], []
    for mapping in manifest['frame_mapping']:
        frame = mapping['output_frame']
        info = infos[mapping['source_frame']]
        old, new = [np.asarray(info[key], dtype=np.float64) for key in (BEFORE, AFTER)]
        for pose in (old, new):
            assert pose.shape == (4, 4) and np.isfinite(pose).all()
            np.testing.assert_allclose(pose[3], [0, 0, 0, 1], atol=1e-8)
        np.testing.assert_allclose(np.loadtxt(original/'ego_pose'/f'{frame:06d}.txt'), new, atol=1e-8)
        np.savetxt(converted/'ego_pose'/f'{frame:06d}.txt', old)
        translation_deltas.append(float(np.linalg.norm(new[:3, 3]-old[:3, 3])))
        rotation_deltas.append(float(np.degrees(np.arccos(np.clip(
            (np.trace(new[:3, :3] @ old[:3, :3].T)-1)/2, -1, 1)))))
        for name, camera_id in manifest['cameras'].items():
            stem = f'{frame:06d}_{camera_id:02d}.txt'
            ext = np.asarray(info['sensors']['cams'][name]['extrinsic'])
            expected = old @ np.linalg.inv(ext)
            np.testing.assert_allclose(np.loadtxt(original/'ego_pose'/stem), new @ np.linalg.inv(ext), atol=1e-8)
            np.savetxt(converted/'ego_pose'/stem, expected)
            np.testing.assert_allclose(np.loadtxt(converted/'ego_pose'/stem) @ ext, old, atol=1e-8)
    config = yaml.safe_load((after/'train.yaml').read_text())
    old_config = copy.deepcopy(config)
    old_config.update(source_path=str(converted), model_path=str(before/'model'),
                      record_dir=str(before/'model/record'), gpus=[args.gpu],
                      exp_name=args.clip+'_before')
    allowed = {'source_path', 'model_path', 'record_dir', 'gpus', 'exp_name'}
    assert {k:v for k,v in config.items() if k not in allowed} == {
        k:v for k,v in old_config.items() if k not in allowed}
    (before/'train.yaml').write_text(yaml.safe_dump(old_config, sort_keys=False))
    manifest.update(pose_field=BEFORE, camera_pose='before_ego_to_world @ inverse(pkl_ego_to_camera)',
                    shared_non_pose_data=str(original), comparison='before versus after optimization')
    (converted/'conversion.complete.json').write_text(json.dumps(manifest, indent=2))
    (before/'sky.complete').write_text('Shared with after; identical images and masks.\n')
    report = {'clip':args.clip, 'before':str(before), 'after':str(after),
              'before_pose_field':BEFORE, 'after_pose_field':AFTER, 'gpu':args.gpu,
              'iterations':config['train']['iterations'], 'random_seed':0,
              'shared_inputs':sorted(shared), 'config_difference_keys':sorted(allowed),
              'ego_translation_delta_m':{'mean':float(np.mean(translation_deltas)), 'max':max(translation_deltas)},
              'ego_rotation_delta_degrees':{'mean':float(np.mean(rotation_deltas)), 'max':max(rotation_deltas)},
              'checked_frames':len(manifest['frame_mapping']),
              'checked_camera_poses':len(manifest['frame_mapping'])*len(manifest['cameras']),
              'evaluation_split':'training views; reconstruction fit, not held-out generalization'}
    report_dir = comparison / args.clip
    report_dir.mkdir()
    (report_dir/'pair.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
