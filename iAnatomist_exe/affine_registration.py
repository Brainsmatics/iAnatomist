import os
import numpy as np
import SimpleITK as sitk
from torch.distributed.fsdp.fully_sharded_data_parallel import FLAT_PARAM

from a_vit_seg_modeling import VisionTransformer3d as ViT_seg
from a_vit_seg_configs import get_r50_b16_config
import torch
from skimage import transform
import ants

# 利用antspy进行仿射变换
def affineReg(img):
    '''
        仿射变换函数，调用模型并预测出外轮廓，然后利用ANTs进行配准，中间文件会被保存
        输入：待配准图像
        输出：配准后的图像
    '''
    # save img to affine dir
    # sitk.WriteImage(sitk.GetImageFromArray(img), './cache/affine/moving.nii.gz')

    # load model parameters
    config_vit = get_r50_b16_config()
    model = ViT_seg(config_vit, img_size=(128,128,128), num_classes=2).cuda()
    model.load_state_dict(torch.load('./cache/checkpoints/checkpoint_affine.pth', map_location='cuda'))
    model.eval()

    img_resize, old_shape = _image_process(img)
    img_resize = img_resize.cuda()
    outputs = model(img_resize)
    mask_pred = outputs.argmax(dim=1).detach().cpu().numpy()

    # save the predicted outline
    if old_shape[0] > 700:
        # moving_outline = transform.resize(mask_pred[0, ...], output_shape=(int(old_shape[0]*0.4), int(old_shape[1]*0.4), int(old_shape[2]*0.4),),
        #                                   order=0, anti_aliasing=False, preserve_range=True)
        # img = transform.resize(img, output_shape=(int(old_shape[0]*0.4), int(old_shape[1]*0.4), int(old_shape[2]*0.4),),
        #                        order=0, anti_aliasing=False, preserve_range=True)
        # moving_outline = ants.from_numpy(moving_outline.transpose(2, 1, 0) * 10)
        moving_outline = transform.resize(mask_pred[0, ...], output_shape=(
        int(old_shape[0]), int(old_shape[1]), int(old_shape[2]),),
                                          order=0, anti_aliasing=False, preserve_range=True)
        img = transform.resize(img, output_shape=(
        int(old_shape[0]), int(old_shape[1]), int(old_shape[2]),),
                               order=0, anti_aliasing=False, preserve_range=True)
        moving_outline = ants.from_numpy(moving_outline.transpose(2, 1, 0) * 10)
        '''
        判断图片的大小，执行不同的线性配准程序，因为不同的大小需要不同的参数
        '''
        fixed_outline = ants.image_read('./cache/affine/annotation.tif')
        mytx = ants.registration(ants.resample_image(fixed_outline, (2.5,2.5,2.5), use_voxels=False, interp_type=0),
                                 ants.resample_image(moving_outline, (2.5,2.5,2.5), use_voxels=False, interp_type=0),
                                 type_of_transform='AffineFast')
        moving_image = ants.apply_transforms(fixed_outline, ants.from_numpy(img.transpose(2, 1, 0)),
                                             mytx['fwdtransforms'])
    else:
        moving_outline = transform.resize(mask_pred[0,...], output_shape=old_shape,
                               order=0, anti_aliasing=False, preserve_range=True)
        moving_outline = ants.from_numpy(moving_outline.transpose(2,1,0)*10)
        '''
            判断图片的大小，执行不同的线性配准程序，因为不同的大小需要不同的参数
            '''
        fixed_outline = ants.image_read('./cache/affine/annotation_25.tif')
        mytx = ants.registration(fixed_outline, moving_outline, type_of_transform='AffineFast')
        moving_image = ants.apply_transforms(fixed_outline, ants.from_numpy(img.transpose(2, 1, 0)),
                                             mytx['fwdtransforms'])

    # read the registration result and return
    return moving_image.numpy().transpose(2,1,0), mytx['fwdtransforms'][0]

# 利用ants进行仿射变换
# def affineReg(img):
#     '''
#         仿射变换函数，调用模型并预测出外轮廓，然后利用ANTs进行配准，中间文件会被保存
#         输入：待配准图像
#         输出：配准后的图像
#     '''
#     # save img to affine dir
#     sitk.WriteImage(sitk.GetImageFromArray(img), './cache/affine/moving.nii.gz')
#
#     # load model parameters
#     config_vit = get_r50_b16_config()
#     model = ViT_seg(config_vit, img_size=(128,128,128), num_classes=2).cuda()
#     model.load_state_dict(torch.load('./cache/checkpoints/checkpoint_affine.pth', map_location='cuda'))
#     model.eval()
#
#     img, old_shape = _image_process(img)
#     img = img.cuda()
#     outputs = model(img)
#     mask_pred = outputs.argmax(dim=1).detach().cpu().numpy()
#
#     # save the predicted outline
#     outline = transform.resize(mask_pred[0,...], output_shape=old_shape,
#                            order=0, anti_aliasing=False, preserve_range=True)
#     sitk.WriteImage(sitk.GetImageFromArray(outline),
#                     os.path.join('./cache/affine/', 'moving_outline.nii.gz'))
#
#     '''
#     判断图片的大小，执行不同的线性配准程序，因为不同的大小需要不同的参数
#     '''
#     if old_shape[0] > 800:
#         # run the affine registration script
#         os.chdir('./cache/affine/')
#         os.system('sh affine_registration_10um.sh')
#         os.chdir('../..')
#     else:
#         # run the affine registration script
#         os.chdir('./cache/affine/')
#         os.system('sh affine_registration.sh')
#         os.chdir('../..')
#
#     # read the registration result and return
#     return sitk.GetArrayFromImage(sitk.ReadImage('./cache/affine/moving_affine.nii.gz'))


def _image_process(img):
    """
    preprocess the image and resize
    """
    img_shape = img.shape
    img[img > 255] = 255
    img = img / 255.0
    img = transform.resize(img, output_shape=(128, 128, 128),
                           order=0, anti_aliasing=False, preserve_range=True)
    return torch.from_numpy(img[None,None, ...]).float(), img_shape
