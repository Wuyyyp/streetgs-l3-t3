"""DA3 sky masks. Run in the separately installed Depth Anything 3 environment."""
import argparse
import glob
import os
import cv2
import numpy as np
import torch
import tqdm
from depth_anything_3.api import DepthAnything3
from safetensors.torch import load_file

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--weights', required=True, help='DA3MONO-LARGE model.safetensors')
args = parser.parse_args()

if __name__ == '__main__':
    os.makedirs(os.path.join(args.out, 'sky_mask'), exist_ok=True)
    model = DepthAnything3('da3mono-large')
    model.load_state_dict(load_file(args.weights), strict=False)
    model.eval()
    model = model.to(device=torch.device('cuda'))
    test_imgs = glob.glob(os.path.join(args.root, 'images', '*.png'))
    test_imgs.sort()
    if len(test_imgs) == 0:
        test_imgs = glob.glob(os.path.join(args.root, 'images', '*.jpg'))
        test_imgs.sort()

    # example_path = "/home/yusen/Downloads/clip_M18/downsample_M18-2_07_20251230162345_DF/glomap_workspace/images/9"
    # images = sorted(glob.glob(os.path.join(example_path, "*.jpg")))[:2]
    for img_f in tqdm.tqdm(test_imgs):
        img = cv2.imread(img_f,-1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raw_h, raw_w, _ = img.shape
        ds_h, ds_w = raw_h // 2, raw_w // 2
        img_ds = cv2.resize(img, dsize=(ds_w, ds_h), interpolation=cv2.INTER_LINEAR)
        prediction = model.inference(
            [img_ds], process_res=int(max(ds_h, ds_w))
        )

        skymask = np.asarray(prediction.sky[0] * 255, dtype=np.uint8)
        mask_resize = cv2.resize(skymask, dsize=(raw_w, raw_h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(args.out, 'sky_mask', os.path.basename(img_f)), mask_resize)
