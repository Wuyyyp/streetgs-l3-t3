#!/usr/bin/env python3
"""Validate and summarize all eight complete T3 evaluations."""
import json
from pathlib import Path
import statistics

root = Path('/data/l3data-reconstruction-bingxing/tem-test/streetGS/T3-test')
camera_names = {0:'左前', 1:'右前', 2:'后视', 3:'左后', 4:'右后', 9:'前视 FOV30', 10:'前视 FOV120'}
rows, camera_rows = [], []
for sample in sorted((root/'before').iterdir()):
    if not sample.is_dir():
        continue
    folder = root/'comparison'/sample.name
    assert json.loads((folder/'full_psnr.state.json').read_text())['stage'] == 'complete'
    data = {v:json.loads((folder/(v+'_full_psnr.json')).read_text()) for v in ['before','after']}
    expected = {f'{f:06d}_{c:02d}' for f in range(101) for c in camera_names}
    for variant, value in data.items():
        assert value['iteration'] == 25000 and value['image_count'] == 707
        assert value['mask'] == 'full frame, unmasked'
        assert len(value['per_view']) == 707 and {v['image'] for v in value['per_view']} == expected
        assert abs(statistics.mean(v['psnr'] for v in value['per_view'])-value['mean_psnr']) < 1e-10
    for name in expected:
        assert (root/'after'/sample.name/'converted/images'/(name+'.jpg')).resolve() == (
            sample/'converted/images'/(name+'.jpg')).resolve()
    before, after = data['before']['mean_psnr'], data['after']['mean_psnr']
    label = sample.name[11:19].replace('_', ':')
    rows.append(dict(clip=sample.name, label=label, images_per_variant=707,
                     before_psnr=before, after_psnr=after, delta_psnr=after-before))
    for camera, camera_name in camera_names.items():
        a, b = [data[v]['per_camera'][str(camera)] for v in ['before','after']]
        assert a['image_count'] == b['image_count'] == 101
        camera_rows.append(dict(clip=sample.name, label=label, camera=camera, camera_name=camera_name,
            images_per_variant=101, before_psnr=a['mean_psnr'], after_psnr=b['mean_psnr'],
            delta_psnr=b['mean_psnr']-a['mean_psnr']))
assert len(rows) == 4 and len(camera_rows) == 28
overall = {v+'_psnr':statistics.mean(r[v+'_psnr'] for r in rows) for v in ['before','after']}
overall['delta_psnr'] = overall['after_psnr']-overall['before_psnr']
result = dict(iteration=25000, images_per_variant=2828, total_evaluations=5656,
    aggregation='Arithmetic mean of per-image full-frame PSNR in dB',
    split='original training views', mask='full frame, unmasked',
    overall=overall, clips=rows, per_camera=camera_rows)
out = root/'comparison'
(out/'full_psnr_summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
lines = ['# T3 StreetGS 全量 PSNR', '',
    '25,000 步模型；每组全部 101 帧 × 7 视角，共 707 张图。每张图计算 PSNR 后取算术均值，单位 dB。',
    '使用完整图像，不额外应用训练时的车身遮挡 mask；直接比较浮点渲染结果与训练图像。', '',
    '| Clip | 修复前 | 修复后 | 修复后 − 修复前 |', '|---|---:|---:|---:|']
for row in rows + [dict(label='四个 clip 全量均值', **overall)]:
    lines.append(f"| {row['label']} | {row['before_psnr']:.4f} | {row['after_psnr']:.4f} | {row['delta_psnr']:+.4f} |")
lines += ['', '每个版本总计 2,828 张图，修复前后共完成 5,656 次评估。属于训练视角的重建拟合指标。', '',
    '## 分视角结果', '', '| Clip | 视角 | 图像数/版本 | 修复前 | 修复后 | 差值 |', '|---|---|---:|---:|---:|---:|']
for row in camera_rows:
    lines.append(f"| {row['label']} | {row['camera_name']} | 101 | {row['before_psnr']:.4f} | {row['after_psnr']:.4f} | {row['delta_psnr']:+.4f} |")
(out/'full_psnr_summary.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(overall=overall, clips=rows), indent=2, ensure_ascii=False))
