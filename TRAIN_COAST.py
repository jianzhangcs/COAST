# -*- coding: utf-8 -*-
import os
from datetime import datetime
import cv2
import glob
import torch
import platform
import numpy as np
from tqdm import tqdm
from time import time
import torch.nn as nn
import scipy.io as sio
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from skimage.measure import compare_ssim as ssim
from utils import RandomDataset, imread_CS_py, img2col_py, col2im_CS_py, psnr, write_data

parser = ArgumentParser(description='ISTA-Net-plus')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=800, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=20, help='phase number of ISTA-Net-plus')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')

parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')

parser.add_argument('--test_name', type=str, default='Set11', help='name of test set')
parser.add_argument('--save_cycle', type=int, default=10, help='epoch number of each test cycle')
parser.add_argument('--model_name', type=str, default='COAST', help='log directory')

args = parser.parse_args()
start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
gpu_list = args.gpu_list
model_name = args.model_name
test_name = args.test_name
save_cycle = args.save_cycle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_dir = os.path.join(args.data_dir, test_name)
filepaths = glob.glob(test_dir + '/*.tif')
ImgNum = len(filepaths)
ratio_dict = {1: 10, 4: 43, 10: 109, 20: 218, 25: 272, 30: 327, 40: 436, 50: 545}
n_output = 1089
nrtrain = 88912  # number of training blocks
batch_size = 64
total_phi_num = 50
rand_num = 25

train_cs_ratio_set = [10, 20, 30, 40, 50]

Phi_all = {}
for cs_ratio in train_cs_ratio_set:
    size_after_compress = ratio_dict[cs_ratio]
    Phi_all[cs_ratio] = np.zeros((int(rand_num * 1), size_after_compress, 1089))
    Phi_name = './%s/phi_sampling_%d_%dx%d.npy' % (args.matrix_dir, total_phi_num, size_after_compress, 1089)
    Phi_data = np.load(Phi_name)
    for k in range(rand_num):
        Phi_all[cs_ratio][k, :, :] = Phi_data[k, :, :]

Phi = {}
for cs_ratio in train_cs_ratio_set:
    Phi[cs_ratio] = torch.from_numpy(Phi_all[cs_ratio]).type(torch.FloatTensor)
    Phi[cs_ratio] = Phi[cs_ratio].to(device)

Training_data_Name = 'Training_Data.mat'
Training_data = sio.loadmat('./%s/%s' % (args.data_dir, Training_data_Name))
Training_labels = Training_data['labels']


class CPMB(nn.Module):
    '''Residual block with scale control
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, res_scale_linear, nf=32):
        super(CPMB, self).__init__()

        conv_bias = True
        scale_bias = True
        cond_dim = 2

        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=conv_bias)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=conv_bias)
        self.res_scale = res_scale_linear
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        cond = x[1]
        content = x[0]
        cond = cond[:, 0:1]
        cond_repeat = cond.repeat((content.shape[0], 1))
        out = self.act(self.conv1(content))
        out = self.conv2(out)
        res_scale = self.res_scale(cond_repeat)
        alpha1 = res_scale.view(-1, 32, 1, 1)
        out1 = out * alpha1
        return content + out1, cond


class BasicBlock(torch.nn.Module):
    def __init__(self, res_scale_linear):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))

        self.head_conv = nn.Conv2d(1, 32, 3, 1, 1, bias=True)
        self.ResidualBlocks = nn.Sequential(
            CPMB(res_scale_linear=res_scale_linear, nf=32),
            CPMB(res_scale_linear=res_scale_linear, nf=32),
            CPMB(res_scale_linear=res_scale_linear, nf=32)
        )
        self.tail_conv = nn.Conv2d(32, 1, 3, 1, 1, bias=True)

    def forward(self, x, PhiTPhi, PhiTb, cond, block_size):
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x_input = x.view(-1, 1, block_size, block_size)

        x_mid = self.head_conv(x_input)
        x_mid, cond = self.ResidualBlocks([x_mid, cond])
        x_mid = self.tail_conv(x_mid)
        x_pred = x_input + x_mid

        x_pred = x_pred.view(-1, block_size * block_size)

        return x_pred



class COAST(torch.nn.Module):
    def __init__(self, LayerNo):
        super(COAST, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        nf = 32
        scale_bias = True
        res_scale_linear = nn.Linear(1, nf, bias=scale_bias)

        for i in range(LayerNo):
            onelayer.append(BasicBlock(res_scale_linear=res_scale_linear))

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, x, Phi, block_size=33):

        Phix = x[0]
        cond = x[1]

        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)
        x = PhiTb.clone()

        for i in range(self.LayerNo):
            x = self.fcs[i](x, PhiTPhi, PhiTb, cond, block_size)

        x_final = x

        return x_final


model = COAST(layer_num)
model = nn.DataParallel(model)
model = model.to(device)

print_flag = 1

if print_flag:
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

if (platform.system() == "Windows"):
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size,
                             num_workers=0, shuffle=True)
else:
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size,
                             num_workers=4, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "./%s/%s_layer_%d_group_%d_ratio_all_lr_%.5f" % (
    args.model_dir, model_name, layer_num, group_num, learning_rate)

log_file_name = "./%s/%s_Log_COAST_layer_%d_group_%d_lr_%.4f.txt" % (
    args.log_dir, model_name, layer_num, group_num, learning_rate)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if start_epoch > 0:
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (pre_model_dir, start_epoch)))


def get_cond(cs_ratio, sigma, cond_type):
    para_noise = sigma / 5.0
    if cond_type == 'org_ratio':
        para_cs = cs_ratio / 100.0
    else:
        para_cs = cs_ratio * 2.0 / 100.0

    para_cs_np = np.array([para_cs])
    para_cs = torch.from_numpy(para_cs_np).type(torch.FloatTensor)
    para_cs = para_cs.to(device)

    para_noise_np = np.array([para_noise])
    para_noise = torch.from_numpy(para_noise_np).type(torch.FloatTensor)

    para_noise = para_noise.to(device)
    para_cs = para_cs.view(1, 1)
    para_noise = para_noise.view(1, 1)
    para = torch.cat((para_cs, para_noise), 1)

    return para


# Training loop
for epoch_i in range(start_epoch + 1, end_epoch + 1):

    print('\n', '-' * 15, 'Epoch: [%d/%d]' % (epoch_i, end_epoch), '-' * 15)
    total_iter_num = np.ceil(nrtrain / batch_size)

    for _, data in tqdm(enumerate(rand_loader), total=total_iter_num):
        batch_x = data
        batch_x = batch_x.to(device)
        rand_Phi_index = np.random.randint(rand_num * 1)

        rand_cs_ratio = np.random.choice(train_cs_ratio_set)
        cur_Phi = Phi[rand_cs_ratio][rand_Phi_index]
        Phix = torch.mm(batch_x, torch.transpose(cur_Phi, 0, 1))

        x_input = [Phix, get_cond(rand_cs_ratio, 0.0, 'org_ratio')]
        x_output = model(x_input, cur_Phi)

        loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))
        loss_all = loss_discrepancy

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

    output_data = str(datetime.now()) + " [%d/%d] Total loss: %.4f, discrepancy loss: %.4f\n" % (
        epoch_i, end_epoch, loss_all.item(), loss_discrepancy.item())
    print(output_data)

    write_data(log_file_name, output_data)

    if epoch_i % save_cycle == 0:
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters
