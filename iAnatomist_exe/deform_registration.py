import glob
import os
import numpy as np
import torch
import SimpleITK as sitk
from skimage import transform
import time
import matplotlib.pyplot as plt
from natsort import natsorted
from d_TransMorph_diff import TransMorphDiff_MutiTask, Bilinear
from d_TransMorph_diff import CONFIGS as CONFIGS_TM


def deformReg(img):
    '''
        非线性配准程序，利用深度学习网络对线性配准好的图像进行非线性配准
        输入：线性配准后的图像
        输出：非线性配准后的图像
    '''
    config = CONFIGS_TM['TransMorphDiff']
    model = TransMorphDiff_MutiTask(config).cuda()
    # torch.serialization.add_safe_globals([model])
    model.load_state_dict(torch.load('./cache/checkpoints/checkpoint_deform.pth'))
    model.eval()

    reg_model_nn = Bilinear(zero_boundary=True, mode='nearest').cuda()
    for param in reg_model_nn.parameters():
        param.requires_grad = False
        param.volatile = True
    reg_model_high = Bilinear(zero_boundary=True, mode='bilinear')
    for param in reg_model_high.parameters():
        param.requires_grad = False
        param.volatile = True

    y = sitk.GetArrayFromImage(sitk.ReadImage('./cache/deform/atlas.tif'))
    x, y, scale = _img_process(img, y)
    x_def, flow, disp_field, seg, _, _ = model((x.cuda().float(), y.cuda().float()))

    '''
    这里主要是把非线性形变场进行上采样到原图大小，然后应用到原图上
    '''

    # x_high = img
    flow_high = transform.resize(flow.detach().cpu().numpy()[0, :, :, :, :].transpose(1,2,3,0),
                         output_shape=img.shape, order=1, anti_aliasing=True, preserve_range=True)

    # flow_high_x = transform.resize(flow.detach().cpu().numpy()[0, 0, :, :, :],
    #                              output_shape=img.shape, order=1, anti_aliasing=True, preserve_range=True)
    # flow_high_y = transform.resize(flow.detach().cpu().numpy()[0, 1, :, :, :],
    #                                output_shape=img.shape, order=1, anti_aliasing=True, preserve_range=True)
    # flow_high_z = transform.resize(flow.detach().cpu().numpy()[0, 2, :, :, :],
    #                                output_shape=img.shape, order=1, anti_aliasing=True, preserve_range=True)
    # flow_high = np.stack([flow_high_x, flow_high_y, flow_high_z], axis=3)

    # sitk.WriteImage(sitk.GetImageFromArray(
    #     flow_high), './cache/deform/deform_transform.nii.gz')
    def_out = reg_model_high(torch.from_numpy(img[None,None,...]).float(),
                             torch.from_numpy(flow_high.transpose(3,0,1,2)[None,...]))
    # sitk.WriteImage(sitk.GetImageFromArray(
    #     def_out.cpu().numpy()[0, 0, :, :, :]), './cache/deform/moving_deform.nii.gz')

    return def_out.cpu().numpy()[0, 0, :, :, :], flow_high


def _img_process(img1, img2):
    img_shape = img1.shape
    img1 = transform.resize(img1, output_shape=(128, 128, 128),
                           order=0, anti_aliasing=False, preserve_range=True)
    img1, img2 = img1[None,None, ...], img2[None,None, ...]
    img1 = np.ascontiguousarray(img1)
    img2 = np.ascontiguousarray(img2)
    img1, img2= torch.from_numpy(img1), torch.from_numpy(img2)

    return img1.cuda(), img2.cuda(), img_shape

