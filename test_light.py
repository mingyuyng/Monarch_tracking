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

    light_net = CNN_Light(480, 128, opt.num_layer, 128, opt.dropout).to(opt.device)

    net_path = os.path.join(opt.net_dir, opt.net_name)
    light_net.load_state_dict(torch.load(net_path, map_location=opt.device))

    if not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)

    if not os.path.exists(opt.output_fig_folder):
        os.makedirs(opt.output_fig_folder)
    
    light_net.eval()

    num = opt.num

    for n in range(num):

        data = sio.loadmat(os.path.join(opt.input_folder, str(n + 1) + '.mat'))
        light_test = data['test_light']

        results_light = np.zeros((light_test.shape[0], light_test.shape[1]))

        for i in range(light_test.shape[0]):
            light = torch.from_numpy(light_test[i, :, :]).float().unsqueeze(1).to(opt.device)

            light_result = light_net(light)

            light_result = torch.sigmoid(light_result)

            results_light[i, :] = light_result.cpu().squeeze().detach().numpy()
        
        #fig_light = os.path.join(opt.output_fig_folder, 'light_'+str(n+1)+'.png')
       # plt.imshow(results_light.transpose(), cmap='viridis', interpolation='nearest')
        #plt.colorbar()
        #plt.savefig(fig_light)
       # plt.clf()
        
        path_light = os.path.join(opt.output_folder, 'light_' + str(n + 1) + '.mat')

        sio.savemat(path_light, {'results': results_light})
        print(n)


if __name__ == '__main__':
    opt = EasyDict()

    opt.dropout = 0.25
    opt.num_layer = 3

    opt.net_dir = './model'
    opt.net_name = 'light_net.w'

    opt.num = 781

    opt.output_folder = './results/Heatmaps_light'
    opt.output_fig_folder = './results/figs_light'
    opt.input_folder = './testdata/Test_set_light'
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test(opt)
