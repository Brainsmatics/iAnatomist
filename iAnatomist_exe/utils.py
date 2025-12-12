import SimpleITK as sitk
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
# import nibabel as nib


def transform3d_construct(mode, angle, factor_x, factor_y):
    """
    Construct a 3d transform base on the draw line
    :param angle:确定线的方向
    :param factor_x:确定了需要形变范围的长
    :param factor_y:确定了需要形变范围的宽
    :param rho:
    :return:
    """
    # construct basic transform
    # 确定了需要形变范围的深度
    factor_z = int(0.8*factor_x)
    transform_array = np.zeros((factor_x, factor_y, factor_z))
    x, y, z = np.meshgrid(np.arange(factor_x//2), np.arange(factor_y//2), np.arange(factor_z//2))
    # # g = np.exp(-((x - factor_x/2) * (x - factor_x/2)/2 + (y - factor_y/2) * (y - factor_y/2)/2 + (z - factor_z/2) * (z - factor_z/2)/2) / (9 * rho * rho * rho))
    # xt = np.log10(factor_x / 4 - abs(x - factor_x / 4) + 1)
    # yt = np.log10(factor_y / 4 - abs(y - factor_y / 4) + 1)
    # zt = np.log10(factor_z / 4 - abs(z - factor_z / 4) + 1)
    xt = (factor_x / 4 - abs(x - factor_x / 4))/factor_x*4
    yt = (factor_y / 4 - abs(y - factor_y / 4)) / factor_y*4
    zt = (factor_z / 4 - abs(z - factor_z / 4)) / factor_z*4
    # 创造一个由线中心向外线性减弱的调整形变场
    g = np.sqrt(xt * yt * zt)
    transform_array[factor_x//4+x, factor_y//4+y, factor_z//4+z] = g

    # rotation of basic transform
    image = sitk.GetImageFromArray(transform_array)
    euler = sitk.Euler3DTransform()
    if mode == 0:
        euler.SetRotation(0, 0, angle)
    elif mode == 1:
        euler.SetRotation(0, angle, 0)
    elif mode == 2:
        euler.SetRotation(angle, 0, 0)
    euler.SetCenter((image.GetSize()[0]/2, image.GetSize()[1]/2, image.GetSize()[2]/2))
    new_transform = sitk.Resample(image, euler, sitk.sitkNearestNeighbor)
    new_transform = sitk.GetArrayFromImage(new_transform)

    return new_transform


# def transform3d_modify(transform_old, mode, center_point, angle, gain, size):
#     """
#
#     :param transform_old:
#     :param center_point:
#     :param angle:
#     :param gain:
#     :param rho:
#     :return:
#     """
#     size = size*20
#     add_transform = transform3d_construct(mode, angle, size*2, size*2)
#     transform_new = np.zeros((transform_old.shape[0]+size*2, transform_old.shape[1]+size*2, transform_old.shape[2]+size*2, 3))
#     transform_new[size:-size, size:-size, size:-size] = transform_old
#     if mode == 0:
#         transform_new[center_point[0] - add_transform.shape[0] // 2+size:center_point[0] + add_transform.shape[0] // 2+size,
#                       center_point[1] - add_transform.shape[1] // 2+size:center_point[1] + add_transform.shape[1] // 2+size,
#                       center_point[2] - add_transform.shape[2] // 2+size:center_point[2] + add_transform.shape[2] // 2+size, 0] \
#             += gain * add_transform * -np.cos(angle)
#         transform_new[center_point[0] - add_transform.shape[0] // 2+size:center_point[0] + add_transform.shape[0] // 2+size,
#                       center_point[1] - add_transform.shape[1] // 2+size:center_point[1] + add_transform.shape[1] // 2+size,
#                       center_point[2] - add_transform.shape[2] // 2+size:center_point[2] + add_transform.shape[2] // 2+size, 1] \
#             += gain * add_transform * -np.sin(angle)
#     elif mode == 1:
#         transform_new[
#         center_point[0] - add_transform.shape[0] // 2 + size:center_point[0] + add_transform.shape[0] // 2 + size,
#         center_point[1] - add_transform.shape[1] // 2 + size:center_point[1] + add_transform.shape[1] // 2 + size,
#         center_point[2] - add_transform.shape[2] // 2 + size:center_point[2] + add_transform.shape[2] // 2 + size, 0] \
#             += gain * add_transform * -np.cos(angle)
#         transform_new[
#         center_point[0] - add_transform.shape[0] // 2 + size:center_point[0] + add_transform.shape[0] // 2 + size,
#         center_point[1] - add_transform.shape[1] // 2 + size:center_point[1] + add_transform.shape[1] // 2 + size,
#         center_point[2] - add_transform.shape[2] // 2 + size:center_point[2] + add_transform.shape[2] // 2 + size, 2] \
#             += gain * add_transform * -np.sin(angle)
#     elif mode == 2:
#         transform_new[
#         center_point[0] - add_transform.shape[0] // 2 + size:center_point[0] + add_transform.shape[0] // 2 + size,
#         center_point[1] - add_transform.shape[1] // 2 + size:center_point[1] + add_transform.shape[1] // 2 + size,
#         center_point[2] - add_transform.shape[2] // 2 + size:center_point[2] + add_transform.shape[2] // 2 + size, 1] \
#             += gain * add_transform * -np.cos(angle)
#         transform_new[
#         center_point[0] - add_transform.shape[0] // 2 + size:center_point[0] + add_transform.shape[0] // 2 + size,
#         center_point[1] - add_transform.shape[1] // 2 + size:center_point[1] + add_transform.shape[1] // 2 + size,
#         center_point[2] - add_transform.shape[2] // 2 + size:center_point[2] + add_transform.shape[2] // 2 + size, 2] \
#             += gain * add_transform * -np.sin(angle)
#     transform_old = transform_new[size:-size, size:-size, size:-size]
#
#     return transform_old


# a = sitk.GetArrayFromImage(sitk.ReadImage("linear_mask2_251Warp.nii.gz"))
# new_transform = transform3d_modify(a, (z, y, x), 0.5, 100,2)
# sitk.WriteImage(sitk.GetImageFromArray(new_transform), "new_warp.nii.gz")
# sitk.WriteImage(sitk.ReadImage("linear_mask2_251Warp.nii.gz"), "new_warp.nii.gz")

def transform3d_modify(img_old, transform_old, mode, center_point, angle, gain, size):
    """

    :param transform_old:
    :param center_point:
    :param angle:
    :param gain:
    :param rho:
    :return:
    """
    size = size*20
    add_transform = transform3d_construct(mode, angle, size*2, size*2)
    img_new = np.zeros((img_old.shape[0]+size*2, img_old.shape[1]+size*2, img_old.shape[2]+size*2))
    img_new[size:-size, size:-size, size:-size] = img_old
    transform_new = np.zeros(
        (transform_old.shape[0] + size * 2, transform_old.shape[1] + size * 2, transform_old.shape[2] + size * 2, 3))
    transform_new[size:-size, size:-size, size:-size] = transform_old
    # 冠状面
    if mode == 0:
        transform = np.stack([gain * add_transform * -np.cos(angle),
                              gain * add_transform * -np.sin(angle),
                              np.zeros_like(add_transform)], axis=3)
        img_crop = img_new[center_point[0] - add_transform.shape[0] // 2+size:center_point[0] + add_transform.shape[0] // 2+size,
                      center_point[1] - add_transform.shape[1] // 2+size:center_point[1] + add_transform.shape[1] // 2+size,
                      center_point[2] - add_transform.shape[2] // 2+size:center_point[2] + add_transform.shape[2] // 2+size]
        trans_img = sitk.Resample(sitk.GetImageFromArray(img_crop),
                                  sitk.DisplacementFieldTransform(sitk.GetImageFromArray(transform)),
                                  sitk.sitkNearestNeighbor)
    # 水平面
    elif mode == 1:
        transform = np.stack([gain * add_transform * -np.sin(angle),
                              np.zeros_like(add_transform),
                              gain * add_transform * -np.cos(angle)], axis=3)
        img_crop = img_new[center_point[0] - add_transform.shape[0] // 2 + size:center_point[0] + add_transform.shape[
            0] // 2 + size,
                   center_point[1] - add_transform.shape[1] // 2 + size:center_point[1] + add_transform.shape[
                       1] // 2 + size,
                   center_point[2] - add_transform.shape[2] // 2 + size:center_point[2] + add_transform.shape[
                       2] // 2 + size]
        trans_img = sitk.Resample(sitk.GetImageFromArray(img_crop),
                                  sitk.DisplacementFieldTransform(sitk.GetImageFromArray(transform)),
                                  sitk.sitkNearestNeighbor)
    # 矢状面
    elif mode == 2:
        transform = np.stack([np.zeros_like(add_transform),
                              gain * add_transform * -np.sin(angle),
                              gain * add_transform * -np.cos(angle)], axis=3)
        img_crop = img_new[center_point[0] - add_transform.shape[0] // 2 + size:center_point[0] + add_transform.shape[
            0] // 2 + size,
                   center_point[1] - add_transform.shape[1] // 2 + size:center_point[1] + add_transform.shape[
                       1] // 2 + size,
                   center_point[2] - add_transform.shape[2] // 2 + size:center_point[2] + add_transform.shape[
                       2] // 2 + size]
        trans_img = sitk.Resample(sitk.GetImageFromArray(img_crop),
                                  sitk.DisplacementFieldTransform(sitk.GetImageFromArray(transform)),
                                  sitk.sitkNearestNeighbor)
    img_new[center_point[0] - add_transform.shape[0] // 2 + size:center_point[0] + add_transform.shape[
            0] // 2 + size,
                   center_point[1] - add_transform.shape[1] // 2 + size:center_point[1] + add_transform.shape[
                       1] // 2 + size,
                   center_point[2] - add_transform.shape[2] // 2 + size:center_point[2] + add_transform.shape[
                       2] // 2 + size] = sitk.GetArrayFromImage(trans_img)
    transform_new[
        center_point[0] - add_transform.shape[0] // 2 + size:center_point[0] + add_transform.shape[0] // 2 + size,
        center_point[1] - add_transform.shape[1] // 2 + size:center_point[1] + add_transform.shape[1] // 2 + size,
        center_point[2] - add_transform.shape[2] // 2 + size:center_point[2] + add_transform.shape[2] // 2 + size, :] \
            += transform
    transform_new = transform_new[size:-size, size:-size, size:-size]
    img_old = img_new[size:-size, size:-size, size:-size]

    return img_old, transform_new