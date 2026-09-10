import struct
import numpy as np
import torch
import torch.nn as nn
def read_dmb(file):
    f = open(file,'rb')
    data = f.read()
    type,h,w,nb = struct.unpack('iiii',data[:16])
    datasize = h*w*nb
    z = [struct.unpack('f',data[(16+i*4):(16+4*(i+1))]) for i in range(datasize)]
    img = np.asarray(z).reshape(h,w,nb)
    return img


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points

def depth2pointcloud(h,w,intrin,extrin,depth,mask=None,pcd_downsample=2):
    points_w = []
    bproj = BackprojectDepth(1, h, w)

    d = torch.from_numpy(depth)
    points_c = bproj(d.unsqueeze(dim=0).unsqueeze(dim=0).float(),
                     torch.from_numpy(np.linalg.pinv(intrin)).unsqueeze(dim=0).float())
    if mask is not None:
        points_c = points_c[:,:,np.asarray(mask.reshape(-1),dtype=np.bool_)]
    if pcd_downsample > 1:
        points_c = points_c[:,:,::pcd_downsample]
    return np.asarray(np.matmul(np.linalg.pinv(extrin), points_c[0])[:3, :])

