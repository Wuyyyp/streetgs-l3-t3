import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--segpath', type=str, default='')
parser.add_argument('--out', type=str, default='/home/yusen/Downloads/code/street_gaussians-main/script/waymo/waymo_splits/waymo_split.txt')
parser.add_argument('--list', type=str, default='/home/yusen/Downloads/code/street_gaussians-main/script/waymo/waymo_splits/segment_list_train.txt')
args = parser.parse_args()

if __name__ == '__main__':
    proc_segs = os.listdir(args.segpath)
    seglist = open(args.list).read().splitlines()
    with open(args.out, 'w') as f:
        f.write('# scene_id, seg_name, start_timestep, end_timestep, scene_type')
        #176,seg139090,0,-1,high-speed-dfrd
        for seg in proc_segs:
            f.write('\n')
            segname = seg.split('.')[0]
            search_id = seglist.index(segname)
            f.write(f'{search_id},seg{segname[8:14]},0,-1,high-speed-dfrd')
