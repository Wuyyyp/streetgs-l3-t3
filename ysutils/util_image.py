import cv2
import glob
import os
def rescaleImages(folder,output,scale=2):
    os.makedirs(output,exist_ok=True)
    imgsList = os.listdir(folder)
    for im in imgsList:
        img = cv2.imread(os.path.join(folder,im))
        img = cv2.resize(img, (img.shape[1]//scale, img.shape[0]//scale))
        cv2.imwrite(os.path.join(output,os.path.basename(im)),img)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/home/yswang/Downloads/mill19_meganerf/building-pixsfm/train/rgbs')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--out', type=str, default='/home/yswang/Downloads/mill19_meganerf/building-pixsfm/train/images')
    args = parser.parse_args()

    rescaleImages(args.input,args.out,args.scale)