from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import SimpleITK as sitk
import torch
from torchvision import transforms
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from natsort import natsorted
from models.iAnatomist import iAnatomist, Bilinear
from models.iAnatomist_config import CONFIGS as CONFIGS_TM

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class WeightedLoss(torch.nn.Module):
    """
    Class that implements automatically weighed loss from:
    https://arxiv.org/pdf/1705.07115.pdf
    NOTE:
    Don't forget to give these params to the optimiser:
    optim.SGD(model.parameters() + criterion.parameters(), optim_args).
    """

    def __init__(self, n_reg_losses=0, n_cls_losses=0):

        super(WeightedLoss, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.
                                   is_available() else "cpu")
        self.reg_coeffs = []
        self.cls_coeffs = []
        for i in range(n_reg_losses):
            init_value = random.random()  # Any init value will do.
            param = torch.nn.Parameter(torch.tensor(init_value))
            name = "reg_param_" + str(i)
            self.register_parameter(name, param)
            self.reg_coeffs.append(param)
        for i in range(n_cls_losses):
            init_value = random.random()
            param = torch.nn.Parameter(torch.tensor(init_value))
            name = "cls_param_" + str(i)
            self.register_parameter(name, param)
            self.cls_coeffs.append(param)

    def forward(self, reg_losses=[], cls_losses=[]):
        '''Forward pass

        Keyword Arguments:
            reg_losses {list} -- List of tensors of regression
            (Tested with smooth L1 and L2) losses (default: {[]})
            cls_losses {list} -- List of tensors of classification
            (tested with BCE) losses (default: {[]})

        Returns:
            torch.Tensor -- 0-dimensional tensor with final loss.
        '''

        assert len(reg_losses) == len(
            self.
            reg_coeffs), "Loss mismatch, check how many reg_losses are passed"
        assert len(cls_losses) == len(
            self.
            cls_coeffs), "Loss mismatch, check how many cls_losses are passed"
        net_loss = torch.zeros(1).to(self.device)
        for i, reg_loss in enumerate(reg_losses):
            net_loss += 0.5 * torch.exp(-self.reg_coeffs[i]) * reg_loss
            net_loss += 0.5 * self.reg_coeffs[i]
        for i, cls_loss in enumerate(cls_losses):
            net_loss += torch.exp(-self.cls_coeffs[i]) * cls_loss
            net_loss += 0.5 * self.cls_coeffs[i]
        return net_loss


class DiceLoss(torch.nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def main():
    batch_size = 2
    save_dir = 'iAnatomist/'
    if not os.path.exists('experiments/'+save_dir):
        os.makedirs('experiments/'+save_dir)
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)
    sys.stdout = Logger('logs/' + save_dir)
    lr = 0.00004
    epoch_start = 0
    max_epoch = 2000
    cont_training = True

    '''
    Initialize spatial transformation function
    '''
    reg_model = Bilinear(zero_boundary=True, mode='nearest').cuda()
    for param in reg_model.parameters():
        param.requires_grad = False
        param.volatile = True
    reg_model_bilin = Bilinear(zero_boundary=True, mode='bilinear').cuda()
    for param in reg_model_bilin.parameters():
        param.requires_grad = False
        param.volatile = True

    '''
    Initialize model
    '''
    config = CONFIGS_TM['iAnatomist']
    model = iAnatomist(config)
    model.cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        epoch_start = 10
        model_dir = 'experiments/'+save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-1]))
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])
    val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16)),
                                       ])

    train_set = datasets.fMOSTBrainDataset('./dataset/moving', transforms=train_composed)
    val_set = datasets.fMOSTBrainDataset('./dataset/val', transforms=val_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=3, pin_memory=True, drop_last=True)

    MIND_criterion = losses.MIND_loss()
    CE_criterion = torch.nn.CrossEntropyLoss()
    Dice_criterion = DiceLoss(13)
    label_criterion = torch.nn.CrossEntropyLoss()
    MSE_criterion = torch.nn.MSELoss()
    Grad_criterion = losses.Grad3d(penalty='l2')
    # awl = AutomaticWeightedLoss(4)
    awl = WeightedLoss(n_reg_losses=2, n_cls_losses=2)

    # optimizer = optim.Adam([
    #     {'params': model.parameters(), "lr": updated_lr},
    #     {'params': awl.parameters(), 'lr': 5e-2, 'weight_decay': 0}
    # ],  amsgrad=True)
    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    best_dsc = 0
    best_seg = 10
    flag = 0
    writer = SummaryWriter(log_dir='logs/' + save_dir)
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            loss_sim_iter = 0
            loss_reg_iter = 0
            loss_seg_iter = 0
            loss_disp_iter = 0
            loss_cons_iter = 0
            loss_label_iter = 0
            loss_disp_seg_iter = 0
            loss_edge_iter = 0
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            path = data[-1]
            data = [t.cuda() for t in data[:-1]]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]
            disp = data[4]
            disp_trans = data[5]
            x_edge = data[6]
            y_edge = data[7]
            x_label = data[8]
            y_label = data[9]
            output = model((x, y))
            loss_sim = MIND_criterion(output[0], y)
            loss_sim_iter += loss_sim
            loss_disp = MSE_criterion(output[1], disp.float())
            loss_disp_iter += loss_disp
            loss_disp_seg = CE_criterion(
                reg_model_bilin(F.one_hot(x_seg.long(), 13).float().squeeze(dim=1).permute(0, 4, 1, 2, 3), output[1]),
                torch.squeeze(y_seg.long(), dim=1))
            loss_cons = Dice_criterion(reg_model_bilin(output[3], output[1]),
                torch.squeeze(y_seg.long(), dim=1))
            loss_cons_iter += loss_cons
            loss_disp_seg_iter += loss_disp_seg
            loss_seg = CE_criterion(output[3], torch.squeeze(x_seg.long(), dim=1))
            loss_seg_iter += loss_seg
            loss_edge = MSE_criterion(output[4]/60000.0, x_edge.float()/60000.0)
            loss_edge_iter += loss_edge
            loss_label = label_criterion(output[5], torch.squeeze(x_label.long()))

            loss_label_iter += loss_label
            loss_reg = Grad_criterion(output[1], y)
            loss_reg_iter += loss_reg
            cord = torch.stack(torch.meshgrid(torch.linspace(-64, 64, 128) / 64, torch.linspace(-64, 64, 128) / 64,
                                              torch.linspace(-64, 64, 128) / 64))[None,...].cuda()
            loss_ble = MSE_criterion(output[1], cord)
            # loss =  + 0.001 * loss_reg + 0.2*loss_disp_seg + 100*loss_sim
            loss = loss_sim + loss_reg + 0.2 * loss_disp_seg + 0.2 * loss_cons + 0.2 * loss_edge + \
                   0.1 * loss_label + 0.1 * loss_seg + 100*loss_ble
            # loss = awl([500*loss_disp, loss_edge], [loss_seg, loss_label]) + 0.01*loss_disp_seg + 0.01*loss_cons + 0.00001*loss_reg

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_all.update(loss.item(), y.numel())

            del output
            output = model((y, x))
            loss_sim = MIND_criterion(output[0], y)
            loss_sim_iter += loss_sim
            loss_disp = MSE_criterion(output[1], disp_trans.float())
            loss_disp_iter += loss_disp
            loss_disp_seg = CE_criterion(
                reg_model_bilin(F.one_hot(y_seg.long(), 13).float().squeeze(dim=1).permute(0, 4, 1, 2, 3), output[1]),
                torch.squeeze(x_seg.long(), dim=1))
            loss_disp_seg_iter += loss_disp_seg
            loss_cons = Dice_criterion(reg_model_bilin(output[3], output[1]),
                                       torch.squeeze(x_seg.long(), dim=1))
            loss_cons_iter += loss_cons
            loss_seg = CE_criterion(output[3], torch.squeeze(y_seg.long(), dim=1))
            loss_seg_iter += loss_seg
            loss_reg = Grad_criterion(output[1], x)
            loss_reg_iter += loss_reg
            loss_edge = MSE_criterion(output[4] / 60000.0, y_edge.float() / 60000.0)
            loss_edge_iter += loss_edge
            loss_label = label_criterion(output[5], torch.squeeze(y_label.long()))
            loss_label_iter += loss_label
            loss_ble = MSE_criterion(output[1], cord)
            loss = loss_sim + loss_reg + 0.2 * loss_disp_seg + 0.2 * loss_cons + 0.2 * loss_edge + \
                   0.1 * loss_label + 0.1 * loss_seg + 100 * loss_ble
            # loss = 0.001 * loss_reg + 0.2 * loss_disp_seg + 100*loss_sim
            # loss = awl([500*loss_disp, loss_edge], [loss_seg, loss_label]) + 0.01*loss_disp_seg + 0.01*loss_cons + 0.00001*loss_reg
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_all.update(loss.item(), y.numel())
            del output

            print('Iter {} of {} loss {:.4f}, Img sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader),
                     loss.item(), loss_sim_iter.item()/2, loss_reg_iter.item()/2))

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        '''
        Validation
        '''
        eval_seg = utils.AverageMeter()
        eval_dsc = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data[:-1]]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]
                grid_img = mk_grid_img(8, 1, config.img_size)
                # _, flow, _, _ = model((x, y), infer=True)
                # def_out = reg_model(x_seg.float(), flow)
                # def_grid = reg_model_bilin(grid_img.float(), flow)
                # dsc = utils.dice_val(def_out.long(), y_seg.long(), 13)
                _, flow, _, seg_out,_,_ = model((x, y), infer=True)
                def_out = reg_model(x_seg.float(), flow)
                def_grid = reg_model_bilin(grid_img.float(), flow)
                # dsc_cal = losses.DiceLossMultiClass(13)
                # dsc = dsc_cal(seg_out, torch.squeeze(x_seg.long(), dim=1))
                dice = DiceLoss(13)
                ce = dice(seg_out, torch.squeeze(x_seg.long(), dim=1))
                dsc = utils.dice_val(def_out.long(), y_seg.long(), 13)
                eval_dsc.update(dsc.item(), x.size(0))
                eval_seg.update(ce.item(), x.size(0))
                print('ce: ' + str(eval_seg.avg))
                print('dsc: ' + str(eval_dsc.avg))
        if best_dsc < eval_dsc.avg:
            # flag += 1
            # if flag % 3 == 1:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_dsc': best_dsc,
                'optimizer': optimizer.state_dict(),
            }, save_dir='experiments/'+save_dir, filename='dsc{:.6f}_{:.6f}.pth.tar'.format(eval_dsc.avg, eval_seg.avg))
            best_dsc = max(eval_dsc.avg, best_dsc)
            best_seg = min(eval_seg.avg, best_seg)
        writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
        plt.switch_backend('agg')
        pred_fig = comput_fig(def_out)
        grid_fig = comput_fig(def_grid)
        x_fig = comput_fig(x_seg)
        tar_fig = comput_fig(y_seg)
        writer.add_figure('Grid', grid_fig, epoch)
        plt.close(grid_fig)
        writer.add_figure('input', x_fig, epoch)
        plt.close(x_fig)
        writer.add_figure('ground truth', tar_fig, epoch)
        plt.close(tar_fig)
        writer.add_figure('prediction', pred_fig, epoch)
        plt.close(pred_fig)
        loss_all.reset()
    writer.close()

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=5):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

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
