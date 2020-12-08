import numpy as np
import torch
from model import *
import scipy.io as sio
import os
from easydict import EasyDict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def test(opt):

    temp_net = CNN_Temp(32, opt.dropout, opt.hidden).to(opt.device)

    net_path = os.path.join(opt.net_dir, opt.net_name)
    temp_net.load_state_dict(torch.load(net_path, map_location=opt.device))

    if not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)
    if not os.path.exists(opt.output_fig_folder):
        os.makedirs(opt.output_fig_folder)

    temp_net.eval()

    num = opt.num

    for n in range(num):

        data = sio.loadmat(os.path.join(opt.input_folder, str(n + 1) + '.mat'))
        temp_test = data['test_temp']

        results_temp = np.zeros((temp_test.shape[0], temp_test.shape[1]))

        for i in range(temp_test.shape[0]):
            temp = torch.from_numpy(temp_test[i, :, :]).float().to(opt.device)            
            temp_result = temp_net(temp)

            temp_result = torch.sigmoid(temp_result)

            results_temp[i, :] = temp_result.cpu().squeeze().detach().numpy()
    
        #fig_temp = os.path.join(opt.output_fig_folder, 'temp_'+str(n+1)+'.png')
        #plt.imshow(results_temp.transpose(), cmap='viridis', interpolation='nearest')
        #plt.colorbar()
        #plt.savefig(fig_temp)
        #plt.clf()

        path_temp = os.path.join(opt.output_folder, 'temp_' + str(n + 1) + '.mat')

        sio.savemat(path_temp, {'results': results_temp})
        print(n)


if __name__ == '__main__':
    opt = EasyDict()

    opt.dropout = 0
    opt.hidden = 256

    opt.net_dir = './model'
    opt.net_name = 'temp_net.w'

    opt.num = 781

    opt.output_folder = './results/Heatmaps_temp'
    opt.output_fig_folder = './results/figs_temp'
    opt.input_folder = './testdata/Test_set_temp'
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test(opt)
