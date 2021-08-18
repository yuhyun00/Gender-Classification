import argparse
from time import gmtime, strftime
import os
import numpy as np
import matplotlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from batch_manager import classification

import numpy as np

from arch.resnet import resnet14, resnet18, resnet34, resnet50, resnet101, resnet152
from arch.vgg import vgg11_m4, vgg11_m4_bn
import train
import val
import test
import wandb
import random

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)

# 해당 모듈이 임포트된 경우가 아닌 인터프리터에서 직접 실행된 경우에 if문 이하의 코드를 실행
if __name__ == '__main__':
    wandb.init(project="Gender classification", reinit=True)
    wandb.run.name = 'vgg11_m4'
    wandb.run.save()

    # parsing:어떤 데이터를 원하는 모양으로 만들어내는 것
    # 인자값을 받을 수 있는 인스턴스 
    parser = argparse.ArgumentParser()
    # 입력 받을 인자값 등록
    parser.add_argument('--arch', type=str, default='vgg11_m4', choices=[vgg11_m4])
    parser.add_argument('--lr_base', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=512) #18:512 50:128
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr_drop_epochs', type=int, default=[10, 15], nargs='+')
    parser.add_argument('--lr_drop_rate', type=float, default=0.1)
    # 입력 받은 인자값을 args에 저장
    args = parser.parse_args()
    wandb.config.update(args)

    # define model
    #startswich:문자열이 특정 문자로 시작하는 지 여부를 알려줌
    if args.arch.startswith('resnet'):
        if args.arch == 'resnet18':
            model = resnet18(num_classes=2)
        elif args.arch == 'resnet14':
            model = resnet14(num_classes=2)
        elif args.arch == 'resnet34':
            model = resnet34(num_classes=2)
        elif args.arch == 'resnet50':
            model = resnet50(num_classes=2)
        elif args.arch == 'resnet101':
            model = resnet101(num_classes=2)
        elif args.arch == 'resnet152':
            model = resnet152(num_classes=2)
        else:
            raise NotImplementedError(f"architecture {args.arch} is not implemented")
    elif args.arch.startswith('vgg'):
        if args.arch == 'vgg11_m4':
            model = vgg11_m4(num_classes=2)
        elif args.arch == 'vgg11_m4_bn':
            model = vgg11_m4_bn(num_classes=2)
        else:
            raise NotImplementedError(f"architecture {args.arch} is not implemented")
    else:
        raise NotImplementedError(f"architecture {args.arch} is not implemented")
    model = model.cuda()
    # 여러 GPU 병렬 처리
    model = torch.nn.parallel.DataParallel(model)
    wandb.watch(model)

    trans_train = transforms.Compose([transforms.Resize((96,96)), transforms.RandomHorizontalFlip(0.5), transforms.RandomChoice([transforms.RandomRotation(30), transforms.RandomPerspective()]), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.18, 0.18, 0.18))])
    #trans_train = transforms.Compose([transforms.Resize((96,96)), transforms.RandomHorizontalFlip(),transforms.RandomPerspective(), transforms.RandomRotation(30), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.18, 0.18, 0.18))])
    trans = transforms.Compose([transforms.Resize((96,96)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.18, 0.18, 0.18))])

    #trainset = torchvision.datasets.ImageFolder(root='./data/archive/Training', transform=trans_train)
    #validset = torchvision.datasets.ImageFolder(root='./data/archive/Validation', transform=trans)
    trainset = classification(0, trans_train)
    validset = classification(1, trans)
    testset = classification(2, trans)

    dataloader_train = DataLoader(trainset, shuffle=True, num_workers=10, batch_size=args.batch_size)
    dataloader_val = DataLoader(validset, shuffle=False, num_workers=10, batch_size=args.batch_size)
    dataloader_test = DataLoader(testset, shuffle=False, num_workers=10, batch_size=args.batch_size)

    #train_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in dataloader_train]
    #train_stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in dataloader_train]

    #train_meanR = np.mean([m[0] for m in train_meanRGB])
    #train_meanG = np.mean([m[1] for m in train_meanRGB])
    #train_meanB = np.mean([m[2] for m in train_meanRGB])
    #train_stdR = np.mean([s[0] for s in train_stdRGB])
    #train_stdG = np.mean([s[1] for s in train_stdRGB])
    #train_stdB = np.mean([s[2] for s in train_stdRGB])


    # LR schedule
    lr = args.lr_base
    lr_per_epoch = []
    for epoch in range(args.epochs):
        if epoch in args.lr_drop_epochs:
            lr *= args.lr_drop_rate
        lr_per_epoch.append(lr)

    # define loss and optimizer(손실 함수와 옵티마이저 SGD)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_base, momentum=0.9, weight_decay=5e-4)

    # save_path
    current_time = strftime('%Y-%m-%d_%H:%M', gmtime())
    save_dir = os.path.join(f'checkpoints/{current_time}')
    os.makedirs(save_dir,  exist_ok=True)

    # train and val
    best_perform, best_epoch = -100, -100
    for epoch in range(1, args.epochs+1):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_per_epoch[epoch-1]
        print(f"Training at epoch {epoch}. LR {lr_per_epoch[epoch-1]}")

        train.train(model, dataloader_train, criterion, optimizer, epoch=epoch)
        val_acc = val.val(model, dataloader_val, epoch=epoch)
        test_acc = test.test(model, dataloader_test, epoch=epoch)

        save_data = {'epoch': epoch,
                     'accuracy': val_acc,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}

        torch.save(save_data, os.path.join(save_dir, f'{epoch:03d}.pth.tar'))
        if epoch > 1:
            os.remove(os.path.join(save_dir, f'{epoch-1:03d}.pth.tar'))
        if val_acc >= best_perform:
            best_perform = val_acc
            best_epoch = epoch
            torch.save(save_data, os.path.join(save_dir, 'best.pth.tar'))

        print(f"best performance {best_perform} at epoch {best_epoch}")
        wandb.log({
            "val_acc": val_acc,
            "test_acc": test_acc
        })
