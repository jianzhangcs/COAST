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
import math
from utils import RandomDataset, imread_CS_py, img2col_py, col2im_CS_py, psnr, write_data

parser = ArgumentParser(description='COAST')

parser.add_argument('--test_epoch', type=int, default=810, help='epoch number of testing')
parser.add_argument('--layer_num', type=int, default=20, help='phase number of COAST')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate of model')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=10, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')

parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')

parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--test_name', type=str, default='Set11', help='name of test set')
parser.add_argument('--test_cycle', type=int, default=10, help='epoch number of each test cycle')
parser.add_argument('--blocksize', type=int, default=33, help='epoch number of each test cycle')
parser.add_argument('--model_name', type=str, default='COAST', help='log directory')


args = parser.parse_args()

test_epoch = args.test_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list
model_name=args.model_name

test_name = args.test_name
test_cycle = args.test_cycle

test_dir = os.path.join(args.data_dir, test_name)
filepaths = glob.glob(test_dir + '/*.tif')

result_dir = os.path.join(args.result_dir, test_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

ImgNum = len(filepaths)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ratio_dict = {1: 10, 4: 43, 10: 109, 20: 218, 25: 272, 30: 327, 40: 436, 50: 545}


n_input = ratio_dict[cs_ratio]
n_output = args.blocksize*args.blocksize


Phi_input = None
total_phi_num = 50
rand_num = 25

test_cs_ratio_set = [args.cs_ratio]



Phi_all = {}
for cs_ratio in test_cs_ratio_set:
    patch_size = args.blocksize * args.blocksize  
    size_after_compress = int(np.ceil(cs_ratio * patch_size / 100))  
    Phi_all[cs_ratio] = np.zeros((int(rand_num * 1), size_after_compress, patch_size))

    Phi_name = './%s/phi_sampling_%d_%dx%d.npy' % (args.matrix_dir, total_phi_num, size_after_compress, patch_size)
    Phi_data = np.load(Phi_name)

   
    for k in range(rand_num):
        Phi_all[cs_ratio][k, :, :] = Phi_data[k, :, :]

        

class CPMB(nn.Module):
    def __init__(self, res_scale_linear, nf=32):
        super(CPMB, self).__init__()

        conv_bias = True
        scale_bias = True
        map_dim = 64
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



    def forward(self, x, PhiTPhi, PhiTb, cond):
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x_input = x.view(-1, 1, args.blocksize, args.blocksize)
        block_num = int(math.sqrt(x_input.shape[0]))
        x_input = x_input.contiguous().view(block_num, block_num, args.blocksize, args.blocksize).permute(0, 2, 1, 3)
        x_input = x_input.contiguous().view(1, 1, int(block_num*args.blocksize), int(block_num*args.blocksize))
        
        x_mid = self.head_conv(x_input)
        x_mid, cond = self.ResidualBlocks([x_mid, cond])
        x_mid = self.tail_conv(x_mid)
        
        x_pred = x_input + x_mid
        x_pred = x_pred.contiguous().view(block_num, args.blocksize, block_num, args.blocksize).permute(0, 2, 1, 3)
        x_pred = x_pred.contiguous().view(-1, args.blocksize*args.blocksize)

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

    def forward(self, x, Phi):

        Phix = x[0]  
        cond = x[1]

        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)

        x = PhiTb.clone()

        for i in range(self.LayerNo):
            x = self.fcs[i](x, PhiTPhi, PhiTb, cond)

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



optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "./%s/%s_layer_%d_group_%d_ratio_all_lr_%.5f" % (args.model_dir, model_name, layer_num, group_num, learning_rate)

log_file_name = "./%s/%s_Log_layer_%d_group_%d_ratio_%d_lr_%.5f.txt" % (args.log_dir, model_name, layer_num, group_num, cs_ratio, learning_rate)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, test_epoch)))



Phi = {}
for cs_ratio in test_cs_ratio_set:
    Phi[cs_ratio] = torch.from_numpy(Phi_all[cs_ratio]).type(torch.FloatTensor)
    Phi[cs_ratio] = Phi[cs_ratio].to(device)
cur_Phi = None  

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


def test_model(epoch_num, cs_ratio, sigma, model_name):
    PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
    SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
    COST_TIME_All = np.zeros([1, ImgNum], dtype=np.float32)

    rand_Phi_index = 0
    cur_Phi = Phi[cs_ratio][rand_Phi_index]
    print("(Test)CS reconstruction start, using Phi[%d][%d] to test" % (cs_ratio, rand_Phi_index))

    with torch.no_grad():
        for img_no in tqdm(range(ImgNum)):
            imgName = filepaths[img_no]

            Img = cv2.imread(imgName, 1)

            Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
            Img_rec_yuv = Img_yuv.copy()

            Iorg_y = Img_yuv[:, :, 0]

            [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y,args.blocksize)
            Icol = Ipad / 255.0

            block_num = int(row_new // args.blocksize)

            Img_output = Icol

            start_time = time()

            batch_x = torch.from_numpy(Img_output)
            batch_x = batch_x.type(torch.FloatTensor)
            batch_x = batch_x.to(device)
            batch_x = batch_x.contiguous().view(block_num, args.blocksize, block_num, args.blocksize).permute(0, 2, 1, 3).contiguous().view(-1,
                                                                                                                    args.blocksize*args.blocksize)

            
            Phix = torch.mm(batch_x, torch.transpose(cur_Phi, 0, 1))

            x_input = [Phix, get_cond(cs_ratio, sigma, 'org_ratio')]
            x_output = model(x_input, cur_Phi)

            end_time = time()

            Prediction_value = x_output.contiguous().view(block_num, block_num, args.blocksize, args.blocksize).permute(0, 2, 1,
                                                                                                3).contiguous().view(
                int(block_num * args.blocksize), int(block_num * args.blocksize))
            Prediction_value = Prediction_value.cpu().data.numpy()
            X_rec = np.clip(Prediction_value, 0, 1)[:row, :col]
            rec_PSNR = psnr(X_rec * 255, Iorg.astype(np.float64))
            rec_SSIM = ssim(X_rec * 255, Iorg.astype(np.float64), data_range=255)


            Img_rec_yuv[:, :, 0] = X_rec * 255

            im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
            im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)

            resultName = imgName.replace(args.data_dir, args.result_dir)
            cv2.imwrite("%s_%s_layer_%d_ratio_%d_sigma_%d_lr_%.5f_epoch_%d_PSNR_%.2f_SSIM_%.4f.png" % (
                resultName, model_name, layer_num, cs_ratio, sigma, learning_rate, epoch_num, rec_PSNR, rec_SSIM),
                        im_rec_rgb)

            del x_output

            PSNR_All[0, img_no] = rec_PSNR
            SSIM_All[0, img_no] = rec_SSIM
            COST_TIME_All[0, img_no] = end_time - start_time

    print_data = str(datetime.now()) + " CS ratio is %d, avg PSNR/SSIM for %s is %.2f/%.4f, epoch number of model is %d, avg cost time is %.4f second(s)\n" % (cs_ratio, args.test_name, np.mean(PSNR_All), np.mean(SSIM_All), epoch_num, np.mean(COST_TIME_All))
    print(print_data)

    output_file_name = "./%s/%s_PSNR_SSIM_Results_layer_%d_group_%d_ratio_%d_sigma_%d_lr_%.5f.txt" % (args.log_dir, model_name, layer_num, group_num, cs_ratio, sigma, learning_rate)

    output_data = "%d, %.2f, %.4f, %.4f\n" % (epoch_num, np.mean(PSNR_All), np.mean(SSIM_All), np.mean(COST_TIME_All))

    write_data(output_file_name, output_data)  


test_model(test_epoch, args.cs_ratio, 0.0, model_name)
