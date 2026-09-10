#!/usr/bin/env python3
# Run on the Pod: check seven-camera data and the existing eleven-camera sample.
from pathlib import Path
import ast,glob,os,json,numpy as np
repo=Path('/data/l3_data_test/street_gaussians-main-local-v2')
base=Path('/data/l3data-reconstruction-bingxing/tem-test/streetGS/l3-test')
source=(repo/'lib/utils/m18_utils.py').read_text()
tree=ast.parse(source)
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_camera_info')
ns={'np':np,'os':os,'glob':glob.glob}
exec(compile(ast.Module(body=[function],type_ignores=[]),str(repo/'lib/utils/m18_utils.py'),'exec'),ns)
paths=[base/'checks/smoke_f0_2/converted',Path('/data/l3_data_test/L3-reconstruction/reconstruction_300_segments/sample_000_clip_M18-2_07_20251128074803_DF_seg000_f0_100_left/converted_0based')]
report=[]
for path in paths:
 intrinsics,extrinsics,egos,c2ws=ns['load_camera_info'](str(path))
 files=sorted((path/'intrinsics').glob('*.txt'))
 mapping={f.stem:i for i,f in enumerate(files)}
 cams=sorted({int(f.stem.split('_')[1]) for f in files})
 assert len(intrinsics)==len(extrinsics)==len(c2ws)==len(egos)*len(cams)
 if len(cams)==11:
  assert all(mapping[f.stem]==int(f.stem[:6])*11+int(f.stem[-2:]) for f in files)
 center=np.mean([np.loadtxt(p)[:3,3] for p in sorted((path/'ego_pose').glob('*.txt')) if '_' not in p.stem],axis=0)
 for f in files:
  expected=np.loadtxt(path/'ego_pose'/f.name)
  expected[:3,3]-=center
  np.testing.assert_allclose(c2ws[mapping[f.stem]],expected,atol=1e-8)
 report.append({'path':str(path),'frames':len(egos),'cameras':cams,'calibration_files':len(files),'passed':True})
print(json.dumps(report,indent=2))
(base/'checks/reader_validation.json').write_text(json.dumps(report,indent=2))
