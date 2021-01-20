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


def train_network(train_data, train_labels, batch_size, valid_loader, net, optimizer, scheduler, criterion, num_iter, log_path, net_path, device, print_interval=1):
    
    index_true = np.where(train_labels==1)[0]
    index_false = np.where(train_labels==0)[0]

    for it in range(num_iter):
        
        batch_true_index = np.random.choice(len(index_true), batch_size//2)
        batch_false_index = np.random.choice(len(index_false), batch_size//2)

        batch_true_data = torch.from_numpy(train_data[index_true[batch_true_index]]).float().to(device)
        batch_false_data = torch.from_numpy(train_data[index_false[batch_false_index]]).float().to(device)
        batch_data = torch.cat((batch_true_data, batch_false_data), 0)
        batch_label = torch.cat((torch.ones(batch_size//2), torch.zeros(batch_size//2)), 0).to(device)

        pred = net(batch_data.unsqueeze(1))
        loss = criterion(pred, batch_label.unsqueeze(1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        scheduler.step()

        if it % print_interval == 0:

            TP, T, P, correct, Number = test_network(net, valid_loader, device)

            recall = TP / T
            precision = TP / P
            accuracy = correct / Number

            F1 = 2 * recall * precision / (recall + precision)
            F2 = 5 * recall * precision / (recall + 4 * precision)
            F0 = (1 + 0.25**2) * recall * precision / (recall + 0.25**2 * precision)

            message = f'Iteration: {it}, loss: {loss.item():.3f}, test acc: {accuracy:.3f}, Recall: {recall:.3f}, Precision: {precision:.3f}, F0.5: {F0:.3f}, F1: {F1:.3f}, F2: {F2:.3f}'

            print(message)

            with open(log_path, "a") as log_file:
                log_file.write('%s/n' % message)

    print('Training finished!')
    torch.save(net.state_dict(), net_path)


def main(opt):

    # Prepare the training data
    train_mat_data = sio.loadmat(opt.train_data)
    valid_mat_data = sio.loadmat(opt.test_data)
    train_data = train_mat_data['data']
    train_labels = train_mat_data['label']

    dataset_valid = dataloader_light(opt.test_data)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=opt.batchsize, shuffle=False, num_workers=1, pin_memory=True)

    net = CNN_Light(480, 128, opt.num_layer, 128, opt.dropout).to(opt.device)
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    scheduler = StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)

    # class 0 : class 1  =  3 : 1
    criterion = nn.BCEWithLogitsLoss()

    if not os.path.exists(opt.save_log_dir):
        os.makedirs(opt.save_log_dir)

    log_path = os.path.join(opt.save_log_dir, opt.save_log_name)

    if not os.path.exists(opt.save_net_dir):
        os.makedirs(opt.save_net_dir)

    net_path = os.path.join(opt.save_net_dir, opt.save_net_name)

    message = f'Start training Neural Network with Dropout: {opt.dropout}, CNN Layer: {opt.num_layer}'
    print(message)

    train_network(train_data, train_labels, opt.batchsize, valid_loader, net, optimizer, scheduler, criterion, opt.iteration, log_path, net_path, opt.device, opt.print_interval)


if __name__ == '__main__':
    opt = EasyDict()

    opt.train_data = 'dataset/Light_train_8.mat'
    opt.test_data = 'dataset/Light_valid_8.mat'

    opt.batchsize = 400
    opt.lr = 1e-3
    opt.iteration = 5000
    opt.num_layer = 3
    opt.dropout = 0

    opt.stepsize = 4000
    opt.gamma = 0.1

    opt.save_log_dir = './logs'
    opt.save_log_name = 'train_light_log.txt'

    opt.save_net_dir = './model'
    opt.save_net_name = 'light_net.w'

    opt.print_interval = 100

    opt.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    main(opt)
