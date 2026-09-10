#!/usr/bin/env python3
"""Render selected original-pose StreetGS models and concatenate 0,10,1 videos."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]


def run(command, log, env):
    with log.open('w') as handle:
        subprocess.run(list(map(str, command)), stdout=handle, stderr=subprocess.STDOUT,
                       cwd=REPO, env=env, check=True)


def render_one(record, work, encode_only=False):
    directory = work / record['sample']
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(record['gpu']), PWD=str(REPO),
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4')
    output = Path(record['video'])
    if output.exists() and not encode_only:
        raise FileExistsError(output)
    command = [sys.executable, REPO/'render_shifted_v2.py',
               '--source_path', record['source'], '--model_path', record['model'],
               '--output_dir', record['render_output'], '--loaded_iter', '25000',
               '--shift_vector', '0', '0', '0', '--max_pixels', '0', '--images_only']
    assert record['shift_vector'] == [0, 0, 0]
    if not encode_only:
        print('RENDER_START',record['sample'],'GPU',record['gpu'],flush=True)
        run(command,directory/'render.log',env)
    images = Path(record['render_output'])/'images'
    start,end = record['frames']
    for frame in range(start,end+1):
        for camera in (0,10,1):
            if not (images/f'{frame:06d}_{camera:02d}.jpg').is_file():
                raise FileNotFoundError(f'frame={frame}, camera={camera}')
    expected = record['count'] * 4
    assert len(list(images.glob('*.jpg'))) == expected
    # Match the acquisition rate (~10 Hz), avoiding the old YAML's 24 fps default.
    fps = round(record['fps']) if abs(record['fps']-round(record['fps'])) < 0.01 else record['fps']
    command = ['ffmpeg','-hide_banner','-loglevel','warning','-nostdin','-n',
               '-filter_complex_threads','1']
    for camera in (0,10,1):
        command += ['-framerate',str(fps),'-start_number',str(start),
                    '-i',str(images/f'%06d_{camera:02d}.jpg')]
    # Original front-camera exclusion used by train.py: bottom from 1500/2160.
    # Common height preserves all three aspect ratios, without stretching/cropping.
    # JPEG uses full-range YUV. Convert to video range before drawing black,
    # otherwise FFmpeg's Y=16 black is interpreted as dark gray in full range.
    scale = 'scale=-2:720:in_range=full:out_range=tv,format=yuv420p'
    filters = (f'[0:v]{scale},setsar=1[left];'
               f'[1:v]{scale},drawbox=x=0:y=ih*1500/2160:w=iw:h=ih:color=black:t=fill,'
               'setsar=1[center];'
               f'[2:v]{scale},setsar=1[right];'
               '[left][center][right]hstack=inputs=3[out]')
    partial = output.with_suffix('.corrected.mp4' if encode_only else '.partial.mp4')
    command += ['-filter_complex',filters,'-map','[out]','-frames:v',str(record['count']),
                '-an','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p',
                '-threads','4','-color_range','tv','-movflags','+faststart',str(partial)]
    run(command,directory/'encode.log',env)
    probe = json.loads(subprocess.check_output([
        'ffprobe','-v','error','-select_streams','v:0','-count_frames',
        '-show_entries','stream=codec_name,width,height,nb_read_frames,avg_frame_rate,duration',
        '-of','json',str(partial)],env=env))
    stream = probe['streams'][0]
    assert int(stream['nb_read_frames']) == record['count'], stream
    assert (stream['width'],stream['height']) == (3440,720), stream
    assert abs(float(stream['duration'])-record['count']/fps) < 0.1, stream
    run(['ffmpeg','-v','error','-nostdin','-threads','2','-i',partial,'-f','null','-'],
        directory/'decode_check.log',env)
    if (directory/'decode_check.log').stat().st_size:
        raise RuntimeError('Video decode reported errors')
    partial.replace(output)
    result = dict(record, encoded_fps=fps, probe=stream,
                  hood_mask='front camera only: y >= height * 1500/2160, filled black',
                  layout='left front (0), front FOV120 (10), right front (1)')
    (directory/'complete.json').write_text(json.dumps(result,indent=2))
    print('VIDEO_COMPLETE',str(output),flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--encode-only',action='store_true')
    args=ap.parse_args()
    records=json.loads(args.manifest.read_text())
    selected=[record for record in records if record['gpu']==args.gpu]
    assert selected
    for record in selected:
        render_one(record,args.manifest.parent,args.encode_only)


if __name__ == '__main__':
    main()
