import torchvision
import torch
import argparse
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from resnet import *
import torch.optim as optim
from utils import progress_bar
import time
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

parser=argparse.ArgumentParser()
parser.add_argument('--use-cuda',help='use cuda',type=str,default="TRUE")
parser.add_argument('--num-of-workers',help='num of workers',type=int,default=0)
parser.add_argument('--optimizer',default='sgd')
parser.add_argument('--batch-normalize',type=str,default="TRUE")
args=parser.parse_args()

if __name__ == '__main__':    

    if args.use_cuda.upper()=="TRUE":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device='cpu'
    torchdevice=torch.device(device)
    print("device:",torchdevice)

    num_of_workers=args.num_of_workers
    print("num of workers: ",num_of_workers)
    optimizer_name=args.optimizer
    print("num of workers: ",optimizer_name)
    batch_normalize=args.batch_normalize.upper()
    print("batch normalize: ",batch_normalize)


    transform=transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform =transform)
    test_data=torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform =transform)

    kwargs = {'pin_memory': True} if device=='cuda' else {}
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=128,
                                            shuffle=True,num_workers=num_of_workers,**kwargs)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=100,
                                            shuffle=True,num_workers=num_of_workers,**kwargs)
    if batch_normalize=='TRUE':
        net = ResNet18()
    else:
        net=ResNet18_NoBN()

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    if optimizer_name=='sgd':
        optimizer = optim.SGD(net.parameters(), lr=0.1,
                            momentum=0.9, weight_decay=5e-4)
    if optimizer_name=='sgd-nesterov':
        optimizer = optim.SGD(net.parameters(), lr=0.1,
                            momentum=0.9, weight_decay=5e-4,nesterov=True)
    if optimizer_name=='adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=0.1,
                            weight_decay=5e-4)
    if optimizer_name=='adadelta':
        optimizer = optim.Adadelta(net.parameters(), lr=0.1,
                            weight_decay=5e-4)
    if optimizer_name=='adam':
        optimizer = optim.Adam(net.parameters(), lr=0.1,
                            weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    def train_model(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        dataloading_time=0
        training_time=0
        dataloading_start=time.perf_counter()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            dataloading_end=time.perf_counter()
            dataloading_time+=(dataloading_end-dataloading_start)
            
            training_start=time.perf_counter()
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            training_end=time.perf_counter()
            training_time+=(training_end-training_start)
            
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            """
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            """
            

            dataloading_start=time.perf_counter()
        total_time=dataloading_time+training_time
        print("data loading time for this epoch: ",dataloading_time,"training time for this epoch: ",training_time,"total time for this epoch: ",total_time)
        print('Loss: %.3f | Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        return dataloading_time,training_time,total_time
        
        
    def test_model(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    dataloadtime_time_all_epoch=0
    training_time_all_epoch=0
    total_time_all_epoch=0
    for epoch in range(start_epoch, start_epoch+5):
        dataloading_time,training_time,total_time=train_model(epoch)
        dataloadtime_time_all_epoch+=dataloading_time
        training_time_all_epoch+=training_time
        total_time_all_epoch+=total_time
        #test_model(epoch)
        scheduler.step()
        
    print("data loading time for all epochs: ",dataloadtime_time_all_epoch/5,"training time for all epochs: ",training_time_all_epoch/5,"total time for all epochs: ",total_time_all_epoch/5)




