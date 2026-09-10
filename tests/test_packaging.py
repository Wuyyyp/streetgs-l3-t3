"""CPU packaging checks: entry points, portable config, timestamp units, pruning."""
import ast
import subprocess
import sys
import tempfile
from pathlib import Path
import yaml

root = Path(__file__).resolve().parents[1]
for folder in ['lib', 'tools', 'tests']:
    for p in (root/folder).rglob('*.py'):
        ast.parse(p.read_text(), filename=str(p))
for p in root.glob('*.py'):
    ast.parse(p.read_text(), filename=str(p))
for folder in ['submodules/diff-gaussian-rasterization',
               'submodules/diff-gaussian-rasterization_ms', 'FaithFusion-main/diff']:
    assert (root/folder/'setup.py').is_file()
    assert (root/folder/'third_party/glm/glm/glm.hpp').is_file()
assert (root/'tools/data/M18proc/M18_converter_xirang_pkl_parallel_v2.py').is_file()
# Run the exact pure timestamp-matching helper without heavy conversion imports.
p = root/'tools/data/convert_t3_streetgs.py'
tree = ast.parse(p.read_text())
f = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'image_path')
scope = {'Path': Path}
exec(compile(ast.Module(body=[f], type_ignores=[]), str(p), 'exec'), scope)
cam = {'data_path':'/camera/1700000000000123.jpg', 'timestamp':1700000000000123}
assert str(scope['image_path'](Path('/raw'), 'left_front_camera', cam)) == '/raw/camera/left_front_camera/1700000000000123000.jpg'
try:
    scope['image_path'](Path('/raw'), 'left_front_camera', dict(cam, timestamp=1))
except AssertionError:
    pass
else:
    raise AssertionError('Mismatched image timestamps accepted')
with tempfile.TemporaryDirectory() as d:
    outputs = []
    for interval in [0,10000]:
        out = Path(d)/f'{interval}.yaml'
        subprocess.run([sys.executable,str(root/'tools/make_config.py'),
                        '--source',d,'--model',d+'/model','--output',str(out),
                        '--prune-interval',str(interval)],check=True)
        outputs.append(yaml.safe_load(out.read_text()))
    a,b = outputs
    assert a['data']['cameras'] == b['data']['cameras'] == [0,1,2,3,4,9,10]
    assert b['optim']['static_prune_until_iter'] == 45000
    assert b['optim']['densify_until_iter'] == 50000
    assert b['train']['iterations'] == 100000
    for c in outputs:
        for key in ['static_prune_enabled','static_prune_interval','static_prune_until_iter']:
            c['optim'].pop(key)
    assert a == b
subprocess.run([sys.executable,str(root/'tests/test_static_prune_schedule.py')],check=True)
print('Packaging checks passed.')
