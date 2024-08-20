#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from gaussiansplatting.utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov, fov2focal,getWorld2View2_tensor,getWorld2View_tensor
from threestudio.utils.ops import get_cam_info_gaussian

class Camera(nn.Module):
    def __init__(self, c2w, FoVy, height, width,
                 trans=torch.tensor([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()
        FoVx = focal2fov(fov2focal(FoVy, height), width)
        # FoVx = focal2fov(fov2focal(FoVy, width), height)


        R = c2w[:3, :3]
        T = c2w[:3, 3]

        self.R = R.float()
        self.T = T.float()
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_height =height
        self.image_width = width

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")


        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans.float()
        self.scale = scale

        # w2c, proj, cam_p = get_cam_info_gaussian(
        #     c2w=c2w, fovx=FoVy, fovy=FoVy, znear=0.1, zfar=self.zfar
        # )

        self.world_view_transform = getWorld2View2_tensor(R, T).transpose(0, 1).float().cuda()
        #self.world_view_transform = w2c.float()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).float().cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0).float()
        #self.full_proj_transform = proj.float()
        self.camera_center = self.world_view_transform.inverse()[3, :3].float()
        #self.camera_center = cam_p.float()
        # print('self.camera_center',self.camera_center)
    
    def __repr__(self):
        return (f"Camera(FoVx={self.FoVx}, FoVy={self.FoVy}, camera_center={self.camera_center}, "
                f"image_height={self.image_height}, image_width={self.image_width}, "
                f"world_view_transform={self.world_view_transform}, "
                f"full_proj_transform={self.full_proj_transform})")

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

