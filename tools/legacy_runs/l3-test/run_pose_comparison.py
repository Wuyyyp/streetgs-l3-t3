#!/usr/bin/env python3
"""Train the before-pose model, then render and compare both completed runs."""
import argparse
import filecmp
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path('/data/l3_data_test/street_gaussians-main-local-v2')
PYTHON = '/usr/bin/python'


def run(command, log):
    print('RUN', ' '.join(map(str, command)), flush=True)
    with log.open('w') as handle:
        subprocess.run(list(map(str, command)), cwd=str(REPO), stdout=handle,
                       stderr=subprocess.STDOUT, check=True)


def summarize(pair, directory, iteration):
    # Use the same PSNR/SSIM implementation as StreetGS, streaming one view.
    sys.path.insert(0, str(REPO))
    sys.argv = ['pose_comparison', '--config', str(Path(pair['before'])/'train.yaml'),
                'mode', 'evaluate', 'gpus', str([pair['gpu']])]
    import numpy as np
    from PIL import Image
    import torch
    import torchvision.transforms.functional as tf
    from lib.utils.loss_utils import psnr, ssim

    roots = {key:Path(pair[key])/'model/train'/f'ours_{iteration}' for key in ('before','after')}
    images = sorted((Path(pair['after'])/'converted/images').glob('*.jpg'))
    views = []
    with torch.no_grad():
        for image in images:
            name = image.stem
            gt_paths = {key:path/(name+'_gt.png') for key,path in roots.items()}
            assert filecmp.cmp(str(gt_paths['before']),str(gt_paths['after']),shallow=False), name
            gt = tf.to_tensor(Image.open(gt_paths['after']).convert('RGB')).cuda()
            view = {'image':name,'camera':int(name.split('_')[-1])}
            for key,path in roots.items():
                rgb = tf.to_tensor(Image.open(path/(name+'_rgb.png')).convert('RGB')).cuda()
                assert rgb.shape == gt.shape
                view[key] = {'PSNR':float(psnr(rgb,gt).item()),'SSIM':float(ssim(rgb,gt).item())}
            views.append(view)
    def average(selected):
        result = {key:{metric:float(np.mean([v[key][metric] for v in selected]))
                       for metric in ('PSNR','SSIM')} for key in roots}
        result['after_minus_before'] = {metric:result['after'][metric]-result['before'][metric]
                                         for metric in ('PSNR','SSIM')}
        return result
    result = {'clip':pair['clip'],'iteration':iteration,'view_count':len(views),
              'split':'training views','mask':'full frame, unmasked',
              'note':'Measures reconstruction fit, not held-out generalization.',
              'overall':average(views),
              'per_camera':{str(cam):average([v for v in views if v['camera']==cam])
                            for cam in sorted({v['camera'] for v in views})}}
    (directory/'metrics.json').write_text(json.dumps(result,indent=2))
    (directory/'per_view.json').write_text(json.dumps(views,indent=2))
    rows = ['<tr><th>Camera</th><th>Before PSNR</th><th>After PSNR</th>'
            '<th>Before SSIM</th><th>After SSIM</th></tr>']
    for camera,values in [('all',result['overall'])]+list(result['per_camera'].items()):
        rows.append('<tr><td>'+camera+'</td>'+''.join(
            f'<td>{values[key][metric]:.4f}</td>'
            for metric in ('PSNR','SSIM') for key in ('before','after'))+'</tr>')
    frames = sorted({int(v['image'][:6]) for v in views})
    chosen = {frames[0],frames[len(frames)//2],frames[-1]}
    sections = []
    for view in views:
        name=view['image']
        if int(name[:6]) not in chosen:
            continue
        figures=[]
        for label,path in [('Ground truth',roots['after']/(name+'_gt.png')),
                           ('Before',roots['before']/(name+'_rgb.png')),
                           ('After',roots['after']/(name+'_rgb.png'))]:
            src=html.escape(os.path.relpath(path,directory),quote=True)
            figures.append(f'<figure><figcaption>{label}</figcaption><img loading="lazy" src="{src}"></figure>')
        sections.append(f'<h3>{html.escape(name)}</h3><div class="views">'+''.join(figures)+'</div>')
    page='<!doctype html><meta charset="utf-8"><title>Pose comparison</title><style>'\
         'body{font:16px system-ui;margin:24px}table{border-collapse:collapse}'\
         'td,th{padding:8px;border:1px solid #bbb}.views{display:flex;gap:12px}'\
         'figure{margin:0;flex:1;min-width:0}img{width:100%}</style>'
    page+=f'<h1>{html.escape(pair["clip"])}: Before / After</h1>'
    page+=f'<p>Iteration {iteration}; {len(views)} training views; full-frame metrics. Higher PSNR/SSIM is better.</p>'
    page+='<table>'+''.join(rows)+'</table>'+''.join(sections)
    (directory/'comparison.html').write_text(page)
    print(json.dumps(result,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pair',type=Path,required=True)
    ap.add_argument('--iteration',type=int,default=25000)
    ap.add_argument('--evaluate-only',action='store_true')
    args=ap.parse_args()
    args.pair=args.pair.resolve()
    pair=json.loads(args.pair.read_text())
    directory=args.pair.parent
    os.chdir(REPO)
    os.environ.update(PWD=str(REPO),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
    before,after=[Path(pair[key]) for key in ('before','after')]
    state=directory/'pipeline_state.json'
    def stage(value):
        state.write_text(json.dumps({'stage':value,'updated_unix':time.time()},indent=2))
        print('STAGE',value,flush=True)
    try:
        if not args.evaluate_only:
            if not (before/'train.complete').exists():
                stage('training_before')
                run([PYTHON,REPO/'train.py','--config',before/'train.yaml'],before/'train.log')
                assert (before/'model/trained_model'/f'iteration_{args.iteration}.pth').is_file()
                (before/'train.complete').write_text('Training succeeded.\n')
            stage('waiting_for_after_training')
            launcher=json.loads((after/'launch.json').read_text())
            while not (after/'train.complete').exists():
                proc=Path('/proc')/str(launcher['pid'])
                if not proc.exists() or not (proc/'cmdline').read_bytes():
                    raise RuntimeError('After training exited without train.complete; inspect after/train.log')
                time.sleep(30)
        for key,sample in [('before',before),('after',after)]:
            assert (sample/'model/trained_model'/f'iteration_{args.iteration}.pth').is_file()
            stage('rendering_'+key)
            run([PYTHON,REPO/'render.py','--config',sample/'train.yaml','mode','evaluate',
                 'loaded_iter',str(args.iteration),'gpus',str([pair['gpu']]),
                 'eval.skip_test','True'],directory/(key+'_render.log'))
        stage('comparing')
        summarize(pair,directory,args.iteration)
        stage('complete')
        (directory/'comparison.complete').write_text('Both runs rendered and compared.\n')
    except Exception as error:
        state.write_text(json.dumps({'stage':'failed','error':repr(error),'updated_unix':time.time()},indent=2))
        raise


if __name__ == '__main__':
    main()
