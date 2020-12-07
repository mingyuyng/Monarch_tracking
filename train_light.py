import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.random import rand, randn
from dataloader import *
from model import *
from torch.optim.lr_scheduler import StepLR
from easydict import EasyDict


def test_network(net, dataloader, device):
    net.eval()
    target = []
    prediction = []
    for i, (frame) in enumerate(dataloader):
        x = frame['intensity'].float().to(device)
        y = frame['label'].float().to(device)
        pred = net(x.unsqueeze(1))
        pred_data = torch.sigmoid(pred).squeeze()
        decision = (pred_data > 0.5).float()
        target.append(y.squeeze().detach().cpu().numpy())
        prediction.append(decision.detach().cpu().numpy())
    target = np.hstack(target)
    prediction = np.hstack(prediction)
    TP = np.sum(np.logical_and(target == 1, prediction == 1))
    T = np.sum(target == 1)
    P = np.sum(prediction == 1)
    correct = np.sum(abs(target - prediction) == 0)
    Number = target.shape[0]
    net.train()
    return TP, T, P, correct, Number


def train_network(train_loader, valid_loader, net, optimizer, scheduler, criterion, num_epoch, log_path, net_path, device, print_interval=1):

    for epoch in range(num_epoch):

        for i, (frame) in enumerate(train_loader):

            x = frame['intensity'].float().to(device)
            y = frame['label'].float().to(device)

            pred = net(x.unsqueeze(1))
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        if epoch % print_interval == 0:

            TP, T, P, correct, Number = test_network(net, valid_loader, device)

            recall = TP / T
            precision = TP / P
            accuracy = correct / Number

            F1 = 2 * recall * precision / (recall + precision)
            F2 = 5 * recall * precision / (recall + 4 * precision)
            F0 = (1 + 0.25**2) * recall * precision / (recall + 0.25**2 * precision)

            message = f'Epoch: {epoch}, loss: {loss.item():.3f}, test acc: {accuracy:.3f}, Recall: {recall:.3f}, Precision: {precision:.3f}, F0.5: {F0:.3f}, F1: {F1:.3f}, F2: {F2:.3f}'

            print(message)

            with open(log_path, "a") as log_file:
                log_file.write('%s/n' % message)

    print('Training finished!')
    torch.save(net.state_dict(), net_path)


def main(opt):
    dataset_train = dataloader_light(opt.train_data)
    dataset_valid = dataloader_light(opt.test_data)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batchsize, shuffle=True, num_workers=1, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=opt.batchsize, shuffle=False, num_workers=1, pin_memory=True)

    net = CNN_Light(480, 128, opt.num_layer, 128, opt.dropout).to(opt.device)
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    scheduler = StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)

    # class 0 : class 1  =  3 : 1
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3]).to(opt.device))

    if not os.path.exists(opt.save_log_dir):
        os.makedirs(opt.save_log_dir)

    log_path = os.path.join(opt.save_log_dir, opt.save_log_name)

    if not os.path.exists(opt.save_net_dir):
        os.makedirs(opt.save_net_dir)

    net_path = os.path.join(opt.save_net_dir, opt.save_net_name)

    message = f'Start training Neural Network with Dropout: {opt.dropout}, CNN Layer: {opt.num_layer}'
    print(message)

    train_network(train_loader, valid_loader, net, optimizer, scheduler, criterion, opt.epoch, log_path, net_path, opt.device, opt.print_interval)


if __name__ == '__main__':
    opt = EasyDict()

    opt.train_data = 'dataset/Light_train_8.mat'
    opt.test_data = 'dataset/Light_valid_8.mat'

    opt.batchsize = 400
    opt.lr = 1e-3
    opt.epoch = 40
    opt.num_layer = 3
    opt.dropout = 0

    opt.stepsize = 30
    opt.gamma = 0.1

    opt.save_log_dir = './logs'
    opt.save_log_name = 'train_light_log.txt'

    opt.save_net_dir = './model'
    opt.save_net_name = 'light_net.w'

    opt.print_interval = 1

    opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    main(opt)
