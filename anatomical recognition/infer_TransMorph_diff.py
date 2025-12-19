import glob
import os, losses, utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
import time
from functools import partial
import SimpleITK as sitk
from torchvision import transforms
from skimage import transform
import matplotlib.pyplot as plt
from multiprocessing import Pool
from natsort import natsorted
from models.TransMorph_diff import TransMorphDiff_MutiTask, Bilinear
from models.TransMorph_diff import CONFIGS as CONFIGS_TM

def warp(path, flow, i):
    print(time.time())
    x_high = sitk.GetArrayFromImage(sitk.ReadImage(path[i].replace('image', 'image_high')))
    flow_high = transform.resize(flow.detach().cpu().numpy()[i, :, :, :, :].transpose(1, 2, 3, 0),
                                 output_shape=x_high.shape, order=1, anti_aliasing=True,
                                 preserve_range=True)
    reg_model_high = Bilinear(zero_boundary=True, mode='bilinear')
    for param in reg_model_high.parameters():
        param.requires_grad = False
        param.volatile = True
    def_out = reg_model_high(torch.from_numpy(x_high[None, None, ...]).float(),
                             torch.from_numpy(flow_high.transpose(3, 0, 1, 2)[None, ...]))
    sitk.WriteImage(sitk.GetImageFromArray(
        def_out.cpu().numpy()[0, 0, :, :, :]), path[0].replace('image', 'image_warp_high'))
    print(time.time())

def main():
    print(time.time())
    model_idx = -1
    model_folder = 'new_TransMorphDiff+nodisp/'
    model_dir = 'experiments/' + model_folder

    config = CONFIGS_TM['TransMorphDiff']
    model = TransMorphDiff_MutiTask(config)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    # del best_model['linear_stack.4.weight']
    # del best_model['linear_stack.4.bias']
    model.load_state_dict(best_model)
    model.cuda()
    reg_model_nn = Bilinear(zero_boundary=True, mode='nearest').cuda()
    for param in reg_model_nn.parameters():
        param.requires_grad = False
        param.volatile = True
    reg_model_high = Bilinear(zero_boundary=True, mode='bilinear')
    for param in reg_model_high.parameters():
        param.requires_grad = False
        param.volatile = True
    test_composed = transforms.Compose([
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = datasets.JHUBrainInferDataset('./dataset/MRI_test/noMRI', transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=3, pin_memory=True, drop_last=True)
    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    k = 0
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            path = data[4]
            data = [t.cuda() for t in data[:4]]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]


            # save warped images and masks
            x_def, flow, disp_field, seg, _, _ = model((x, y))
            def_out = reg_model_nn(x_seg.cuda().float(), flow.cuda())

            sitk.WriteImage(sitk.GetImageFromArray(
                def_out.detach().cpu().numpy()[0, 0, :, :, :]), path[0].replace('image', 'mask_warp'))
            # sitk.WriteImage(sitk.GetImageFromArray(
            #     x_def.detach().cpu().numpy()[0, 0, :, :, :]), path[0].replace('image', 'image_warp'))

            # save warped image with high resolution
            # with Pool(3) as pool:  # 创建包含 4 个进程的进程池
            #     pool.starmap(warp, [(path, flow, i) for i in range(3)])
            x_high = sitk.GetArrayFromImage(sitk.ReadImage(path[0].replace('image', 'image_high')))
            x_seg_high = sitk.GetArrayFromImage(sitk.ReadImage(path[0].replace('image', 'mask_high')))
            flow_high = transform.resize(flow.detach().cpu().numpy()[0, :, :, :, :].transpose(1,2,3,0),
                                 output_shape=x_high.shape, order=1, anti_aliasing=True, preserve_range=True)
            # sitk.WriteImage(sitk.GetImageFromArray(
            #     flow_high), path[0].replace('image', 'disp_high'))
            def_out = reg_model_high(torch.from_numpy(x_high[None,None,...]).float(),
                                     torch.from_numpy(flow_high.transpose(3,0,1,2)[None,...]))
            def_out_seg = reg_model_nn(torch.from_numpy(x_seg_high[None, None, ...]).float(),
                                     torch.from_numpy(flow_high.transpose(3, 0, 1, 2)[None, ...]))
            sitk.WriteImage(sitk.GetImageFromArray(
                def_out.cpu().numpy()[0, 0, :, :, :]), path[0].replace('image', 'image_warp_high'))
            sitk.WriteImage(sitk.GetImageFromArray(
                def_out_seg.cpu().numpy()[0, 0, :, :, :]), path[0].replace('image', 'mask_warp_high'))

            # 10um 结果
            # x_high = sitk.GetArrayFromImage(sitk.ReadImage(path[0].replace('image', 'image_high10')))
            # # flow_high = transform.resize(flow.detach().cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0),
            # #                              output_shape=x_high.shape, order=1, anti_aliasing=True, preserve_range=True)
            # flow_high_x = transform.resize(flow.detach().cpu().numpy()[0, 0, :, :, :],
            #                              output_shape=x_high.shape, order=1, anti_aliasing=True, preserve_range=True)
            # flow_high_y = transform.resize(flow.detach().cpu().numpy()[0, 1, :, :, :],
            #                                output_shape=x_high.shape, order=1, anti_aliasing=True, preserve_range=True)
            # flow_high_z = transform.resize(flow.detach().cpu().numpy()[0, 2, :, :, :],
            #                                output_shape=x_high.shape, order=1, anti_aliasing=True, preserve_range=True)
            # flow_high = np.stack([flow_high_x, flow_high_y, flow_high_z], axis=3)
            # def_out = reg_model_high(torch.from_numpy(x_high[None, None, ...]).float(),
            #                          torch.from_numpy(flow_high.transpose(3, 0, 1, 2)[None, ...]))
            # sitk.WriteImage(sitk.GetImageFromArray(
            #     def_out.cpu().numpy()[0, 0, :, :, :]), path[0].replace('image', 'image_warp_high10'))

            # tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            # jac_det = utils.jacobian_determinant_vxm(disp_field.detach().cpu().numpy()[0, :, :, :, :])
            # # line = utils.dice_val_substruct(def_out.long(), y_seg.long(), stdy_idx)
            # # line = line #+','+str(np.sum(jac_det <= 0)/np.prod(tar.shape))
            # # csv_writter(line, 'experiments/' + model_folder[:-1])
            # eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            # print('det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(tar.shape)))

            # dsc_trans = utils.dice_val(def_out.long(), y_seg.long(), 46)
            # dsc_raw = utils.dice_val(x_seg.long(), y_seg.long(), 46)
            # print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(),dsc_raw.item()))
            # eval_dsc_def.update(dsc_trans.item(), x.size(0))
            # eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            # stdy_idx += 1

            # # flip moving and fixed images
            # x_def, flow, disp_field = model((y, x))
            # def_out = reg_model_nn(y_seg.cuda().float(), flow.cuda())
            # tar = x.detach().cpu().numpy()[0, 0, :, :, :]
            #
            # jac_det = utils.jacobian_determinant_vxm(disp_field.detach().cpu().numpy()[0, :, :, :, :])
            # line = utils.dice_val_substruct(def_out.long(), x_seg.long(), stdy_idx)
            # line = line #+ ',' + str(np.sum(jac_det < 0) / np.prod(tar.shape))
            # out = def_out.detach().cpu().numpy()[0, 0, :, :, :]
            # print('det < 0: {}'.format(np.sum(jac_det <= 0)/np.prod(tar.shape)))
            # csv_writter(line, 'experiments/' + model_folder[:-1])
            # eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            #
            # dsc_trans = utils.dice_val(def_out.long(), x_seg.long(), 46)
            # dsc_raw = utils.dice_val(y_seg.long(), x_seg.long(), 46)
            # print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(), dsc_raw.item()))
            # eval_dsc_def.update(dsc_trans.item(), x.size(0))
            # eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            # stdy_idx += 1

        print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std))
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()
