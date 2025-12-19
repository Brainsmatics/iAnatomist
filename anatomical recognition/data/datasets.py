import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import transform
import numpy as np


class fMOSTBrainInferDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        # path = self.paths[index]
        x_path = glob.glob(self.paths+"/image/*")[index]
        y_path = self.paths.replace('moving', 'fixed').replace('test', 'fixed') + '/image/atlas.tif'
        # y_path = 'G:/Registration/TransMorph/TransMorph/dataset/fixed/image/atlas.tif'
        x_seg_path = x_path
        y_seg_path = self.paths.replace('moving', 'fixed').replace('test', 'fixed') + '/seg/atlas.tif'
        # y_seg_path = 'G:/Registration/TransMorph/TransMorph/dataset/fixed/seg/atlas.tif'
        x = sitk.GetArrayFromImage(sitk.ReadImage(x_path))
        y = sitk.GetArrayFromImage(sitk.ReadImage(y_path))
        x_seg = sitk.GetArrayFromImage(sitk.ReadImage(x_seg_path))
        y_seg = sitk.GetArrayFromImage(sitk.ReadImage(y_seg_path))
        # x = transform.resize(x, output_shape=(128, 128, 128),
        #                        order=1, anti_aliasing=True, preserve_range=True)
        # y = transform.resize(y, output_shape=(128, 128, 128),
        #                         order=0, anti_aliasing=False, preserve_range=True)

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)

        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg, x_path

    def __len__(self):
        return len(glob.glob(self.paths + "/image/*"))


class fMOSTBrainDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        if len(glob.glob(self.paths + "/image/*.nii.gz")) > 13:
            x_path = glob.glob(self.paths + "/image/*.nii.gz")[index]
        else:
            x_path = glob.glob(self.paths + "/image/*.nii.gz")[index]

        if "CH1" in x_path:
            x_label = torch.from_numpy(np.array([1]))
        else:
            x_label = torch.from_numpy(np.array([0]))
        y_label = torch.from_numpy(np.array([5]))
        x_seg_path = glob.glob(self.paths + "/seg/*.nii.gz")[index]
        y_path = self.paths.replace('moving', 'fixed').replace('val', 'fixed') + '/image/atlas.tif'
        y_seg_path = self.paths.replace('moving', 'fixed').replace('val', 'fixed') + '/seg/atlas.tif'
        # y_path = 'G:/Registration/TransMorph/TransMorph/dataset/fixed/image/atlas.tif'
        # y_seg_path = 'G:/Registration/TransMorph/TransMorph/dataset/fixed/seg/atlas.tif'

        disp_path = glob.glob(self.paths.replace('val', 'moving') + '/disp/*.nii.gz')[index]
        x_edge = sitk.GetArrayFromImage(sitk.SobelEdgeDetection(sitk.ReadImage(x_path)))
        y_edge = sitk.GetArrayFromImage(sitk.SobelEdgeDetection(sitk.ReadImage(y_path)))
        x = sitk.GetArrayFromImage(sitk.ReadImage(x_path))
        y = sitk.GetArrayFromImage(sitk.ReadImage(y_path))
        x_seg = sitk.GetArrayFromImage(sitk.ReadImage(x_seg_path))
        y_seg = sitk.GetArrayFromImage(sitk.ReadImage(y_seg_path))
        disp = sitk.GetArrayFromImage(sitk.ReadImage(disp_path))
        # x =
        # y = transform.resize(y, output_shape=(image_size, image_size, image_size),
        #                      order=0, anti_aliasing=False, preserve_range=True)
        # x_seg = transform.resize(x_seg, output_shape=(image_size, image_size, image_size),
        #                      order=0, anti_aliasing=False, preserve_range=True)
        # y_seg = transform.resize(y_seg, output_shape=(image_size, image_size, image_size),
        #                      order=0, anti_aliasing=False, preserve_range=True)

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])

        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x_edge = np.ascontiguousarray(x_edge)
        y_edge = np.ascontiguousarray(y_edge)

        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        x_edge = torch.from_numpy(x_edge)
        y_edge = torch.from_numpy(y_edge)
        # set 1 for the displacements with large offset
        disp[np.abs(disp) > 10] = 1
        disp = torch.from_numpy(disp)
        # convert the displacement in itk to the displacement in torch
        cord = torch.stack(torch.meshgrid(torch.linspace(-64, 64, 128) / 64, torch.linspace(-64, 64, 128) / 64,
                                               torch.linspace(-64, 64, 128) / 64)).permute(1, 2, 3, 0)
        disp_x = (cord + torch.flip(disp,dims=[3])).permute(3,0,1,2)
        disp_y = (cord - torch.flip(disp,dims=[3])).permute(3,0,1,2)
        return x, y, x_seg, y_seg, disp_x, disp_y, x_edge, y_edge, x_label, y_label, x_path

    def __len__(self):
        if len(glob.glob(self.paths + "/image/*.nii.gz")) > 13:
            return len(glob.glob(self.paths + "/image/*.nii.gz"))
        else:
            return len(glob.glob(self.paths + "/image/*.nii.gz"))



