"""Small CUDA regression for optional static contribution pruning."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
temporary = tempfile.TemporaryDirectory(prefix="streetgs-prune-test-")
sys.argv = ["prune-test", "--config", str(root/"configs/example/L3_front.yaml"),
            "source_path", temporary.name, "gpus", "[-1]", "model_path", str(Path(temporary.name)/"model"), "record_dir", str(Path(temporary.name)/"record")]
import numpy as np
import torch
from lib.config import cfg
Path(cfg.model_path).mkdir(parents=True, exist_ok=True)
from lib.utils.static_prune import scheduled_prune_iterations, prune_static_background
from lib.models.gaussian_model_bkgd import GaussianModelBkgd
from lib.utils.graphics_utils import BasicPointCloud
from lib.utils.general_utils import inverse_sigmoid
from lib.utils.camera_utils import make_rasterizer

assert not cfg.optim.static_prune_enabled
assert scheduled_prune_iterations(cfg.optim, 50000) == set()
assert "diff_gaussian_rasterization_ms" not in sys.modules
cfg.optim.static_prune_enabled = True
cfg.optim.densify_until_iter = 25000
assert sorted(scheduled_prune_iterations(cfg.optim, 50000)) == [5000,10000,15000,20000,25000]
assert 30000 not in scheduled_prune_iterations(cfg.optim, 50000)
cfg.optim.static_prune_interval = 0
try:
    scheduled_prune_iterations(cfg.optim, 50000)
    raise AssertionError("invalid interval accepted")
except ValueError:
    pass
cfg.optim.static_prune_interval = 5000
torch.manual_seed(7)
x,y=np.meshgrid(np.linspace(-.32,.32,8,dtype=np.float32),np.linspace(-.32,.32,8,dtype=np.float32))
xyz=np.stack([x.ravel(),y.ravel(),np.full(64,.8,dtype=np.float32)],axis=1)
colors=np.full((64,3),.5,dtype=np.float32)
b=GaussianModelBkgd()
b.create_from_pcd(BasicPointCloud(xyz,colors,np.zeros_like(xyz)),1.)
b._opacity=torch.nn.Parameter(inverse_sigmoid(torch.linspace(.01,.95,64,device="cuda").unsqueeze(1)))
b.training_setup()
# Populate Adam moments, then verify they survive pruning and a further step.
loss=sum(g["params"][0].sum() for g in b.optimizer.param_groups)*1e-6
loss.backward(); b.update_optimizer()
camera=SimpleNamespace(image_height=96,image_width=96,FoVx=np.pi/2,FoVy=np.pi/2,
 world_view_transform=torch.eye(4,device="cuda"),full_proj_transform=torch.eye(4,device="cuda"),
 camera_center=torch.zeros(3,device="cuda"))
actor=SimpleNamespace(get_xyz=torch.randn(5,3,device="cuda"))
actor_before=actor.get_xyz.clone()
scene=SimpleNamespace(background=b,obj_list=["actor"],actor=actor)
event=prune_static_background(25000,scene,[camera]*3)
assert 0<event["static_after"]<64
assert torch.equal(actor.get_xyz,actor_before)
n=event["static_after"]
for group in b.optimizer.param_groups:
    param=group["params"][0]
    assert param.shape[0]==n
    assert b.optimizer.state[param]["exp_avg"].shape==param.shape
assert b.xyz_gradient_accum.shape[0]==b.denom.shape[0]==b.max_radii2D.shape[0]==n
rasterizer=make_rasterizer(camera,b.active_sh_degree)
result=rasterizer(means3D=b.get_xyz,means2D=torch.zeros_like(b.get_xyz,requires_grad=True),
 shs=b.get_features,colors_precomp=None,opacities=b.get_opacity,scales=b.get_scaling,
 rotations=b.get_rotation,cov3D_precomp=None,semantics=b.get_semantic)
assert torch.isfinite(result[0]).all()
result[0].mean().backward(); b.update_optimizer()
assert torch.isfinite(b.get_xyz).all()
print("PASS: disabled default; no optional backend loaded when off; 25000 inclusive; post-cutoff disabled; CUDA prune; dynamic points unchanged; Adam states; native render/backward after prune.")
print(event)
temporary.cleanup()
