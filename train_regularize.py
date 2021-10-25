import argparse
import os
from collections import deque

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
from torchvision import transforms as transforms

import models
import topology
from utils import progress_bar

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def get_weights(net):
    w = []
    for p in net.parameters():
        w.append(p.view(-1))
    return torch.cat(w)


def main():
    parser = argparse.ArgumentParser(description="Regularization of Topology Dimensions")
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs tp train for')
    parser.add_argument('--trainBatchSize', default=100, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    parser.add_argument('--lam', default=0.5, type=float)
    parser.add_argument('--model', default='lenet', type=str)
    parser.add_argument('--between_update', default=400, type=int)
    parser.add_argument('--updates', default=50, type=int)
    parser.add_argument('--save_dir', default='results/', type=str)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    solver = Solver(args)
    solver.run()


class Solver(object):
    def __init__(self, config):
        self.model = config.model
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None
        self.lam = config.lam
        self.weights = deque([])
        self.between_update = config.between_update
        self.updates = config.updates
        self.save_dir = config.save_dir
        self.seed = config.seed

        self.train_accs = []
        self.test_accs = []

    def load_data(self):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root='~/data/', train=True, transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True)
        test_set = torchvision.datasets.CIFAR10(root='~/data/', train=False, transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        self.model = getattr(models, self.model)().to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train(self):
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            self.weights.append(get_weights(self.model))
            if len(self.weights) > 100:
                self.weights.popleft()
                top_loss = topology.calculate_ph_dim_gpu(torch.stack(list(self.weights)), min_points=10, max_points=100, point_jump=10)
            else:
                top_loss = 0
            self.top_dims.append(top_loss) 

            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target) + self.lam * top_loss
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)

            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))

            self.train_accs.append(100. * train_correct / total)
            if self.lam != 0:
                self.weights[-1].detach_()
        return train_loss, train_correct / total

    def test(self):
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))

                self.test_accs.append(100. * test_correct / total)

        return test_loss, test_correct / total

    def save(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        torch.save(self.train_accs, os.path.join(self.save_dir, 'train.pt'))
        torch.save(self.test_accs, os.path.join(self.save_dir, 'test.pt'))

    def run(self):
        self.load_data()
        self.load_model()
        self.accuracy = 0
        for epoch in range(1, self.epochs + 1):
            train_result = self.train()
            torch.save(train_result, os.path.join(self.save_dir, 'train.pt'))
            test_result = self.test()
            torch.save(test_result, os.path.join(self.save_dir, 'test.pt'))
            self.accuracy = max(self.accuracy, test_result[1])
            if epoch == self.epochs:
                self.save()


if __name__ == '__main__':
    main()
