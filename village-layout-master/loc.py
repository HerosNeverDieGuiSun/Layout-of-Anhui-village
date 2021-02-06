import argparse
import torch
import torch.optim as optim
from loc_dataset import LocDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models import *
import os
from random import shuffle
import math

"""
Module that predicts the category of the next house
"""

# parser = argparse.ArgumentParser(description='category')
# parser.add_argument('--data-folder', type=str, default="txt_data_divide", metavar='S')
# parser.add_argument('--num-workers', type=int, default=6, metavar='N')
# parser.add_argument('--last-epoch', type=int, default=-1, metavar='N')
# parser.add_argument('--train-size', type=int, default=5000, metavar='N')
# parser.add_argument('--save-dir', type=str, default="cat_test", metavar='S')
# parser.add_argument('--save-every-n-epochs', type=int, default=5, metavar='N')
# parser.add_argument('--lr', type=float, default=0.0005, metavar='N')
# args = parser.parse_args()

args = {'data_folder': './txt_data_divide',
        'save_dir': './trainedModels',
        'epoch': 4,
        'save_every_n_epochs': 1,
        'block_category': 3,
        'counts_house_categories': 10,
        'lr': 0.0005,
        'batch_size': 3,
        'last-epoch':-1,
        'centroid_weight': 10,
        'train_percentage': 0.8,
        'validation_percentage': 0.2,
        'use_cuda': False
        }

if not os.path.exists(args['save_dir']):
    os.mkdir(args['save_dir'])

# ---------------------------------------------------------------------------------------
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2, num_input_channels=17, use_fc=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=4, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.use_fc = use_fc
        if use_fc:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        return x


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


class DownConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(DownConvBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, stride=2, kernel_size=4, padding=1)
        self.bn = nn.BatchNorm2d(outplanes)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class UpConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(UpConvBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, stride=1, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(outplanes)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = F.upsample(x, mode='nearest', scale_factor=2)
        return self.act(self.bn(self.conv(x)))


class Location(nn.Module):
    def __init__(self, num_classes, num_input_channels):
        super(Location, self).__init__()

        self.model = nn.Sequential(
            nn.Dropout(p=0.2),
            resnet34(num_input_channels=num_input_channels),
            nn.Dropout(p=0.1),
            UpConvBlock(512, 256),
            UpConvBlock(256, 128),
            UpConvBlock(128, 64),
            UpConvBlock(64, 32),
            UpConvBlock(32, 16),
            UpConvBlock(16, 8),
            nn.Dropout(p=0.1),
            nn.Conv2d(8, num_classes, 1, 1)
        )

    def forward(self, x):
        x = self.model(x)
        return x
# ---------------------------------------------------------------------------------------


def divide_dataset(source, train_percentage):
    cwd = os.getcwd()  # 返回当前工作目录
    dir_name = [source + '/loc_train', source + '/loc_validation']
    # 创建训练集loc_train和测试集loc_validation文件夹
    if not os.path.exists(dir_name[0]):
        os.mkdir(dir_name[0])
    if not os.path.exists(dir_name[1]):
        os.mkdir(dir_name[1])

    # 保存被分到train和validation的编号
    id_train = []
    id_validation = []

    # ---------- 为训练集和测试集创建cnts文件夹 ---------- #
    if not os.path.exists(dir_name[0] + '/cnts'):
        os.mkdir(dir_name[0] + '/cnts')
    if not os.path.exists(dir_name[1] + '/cnts'):
        os.mkdir(dir_name[1] + '/cnts')

    # 创建cnts下的各个block类别文件夹
    raw_cnts_path = source + '/cnts'
    for i in range(5):
        train_cnts_path = dir_name[0] + '/cnts/{}_blocks'.format(i)
        validation_cnts_path = dir_name[1] + '/cnts/{}_blocks'.format(i)
        if not os.path.exists(train_cnts_path):
            os.mkdir(train_cnts_path)
        if not os.path.exists(validation_cnts_path):
            os.mkdir(validation_cnts_path)

        temp_cnts_path = raw_cnts_path + '/{}_blocks'.format(i)
        all_cnts_files = os.listdir(temp_cnts_path)
        shuffle(all_cnts_files)  # 打乱
        seperate_num = int(len(all_cnts_files) * train_percentage)
        train_files = all_cnts_files[0:seperate_num]
        validation_files = all_cnts_files[seperate_num:]

        temp_id_train = []
        temp_id_validation = []

        for file in train_files:
            temp_id_train.append(file.split('_')[0])  # 记录划分后的train文件夹中cnts的文件编号
            src_path = temp_cnts_path + '/' + file
            file_path = train_cnts_path + '/' + file
            src = (cwd + src_path[1:]).replace('/', '\\')
            des = cwd + file_path[1:].replace('/', '\\')
            os.system('copy {} {}'.format(src, des))

        for file in validation_files:
            temp_id_validation.append(file.split('_')[0])  # 记录划分后的validation文件夹中cnts的文件编号
            src_path = temp_cnts_path + '/' + file
            file_path = validation_cnts_path + '/' + file
            print('cwd is :', cwd)
            src = cwd + src_path[1:].replace('/', '\\')
            des = cwd + file_path[1:].replace('/', '\\')
            os.system('copy {} {}'.format(src, des))

        id_train.append(temp_id_train)
        id_validation.append(temp_id_validation)

    # ---------- 为训练集和测试集创建info文件夹 ---------- #
    if not os.path.exists(dir_name[0] + '/info'):
        os.mkdir(dir_name[0] + '/info')
    if not os.path.exists(dir_name[1] + '/info'):
        os.mkdir(dir_name[1] + '/info')

    raw_info_path = source + '/info'
    for i in range(5):
        train_info_path = dir_name[0] + '/info/{}_blocks'.format(i)
        validation_info_path = dir_name[1] + '/info/{}_blocks'.format(i)
        if not os.path.exists(train_info_path):
            os.mkdir(train_info_path)
        if not os.path.exists(validation_info_path):
            os.mkdir(validation_info_path)

        temp_info_path = raw_info_path + '/{}_blocks'.format(i)
        # 按照cnts中的编号来取info中文件
        train_files = [id + '_info_{}.txt'.format(i) for id in id_train[i]]
        validation_files = [id + '_info_{}.txt'.format(i) for id in id_validation[i]]

        for file in train_files:
            src_path = temp_info_path + '/' + file
            file_path = train_info_path + '/' + file
            src = cwd + src_path[1:].replace('/', '\\')
            des = cwd + file_path[1:].replace('/', '\\')
            os.system('copy {} {}'.format(src, des))

        for file in validation_files:
            src_path = temp_info_path + '/' + file
            file_path = validation_info_path + '/' + file
            src = cwd + src_path[1:].replace('/', '\\')
            des = cwd + file_path[1:].replace('/', '\\')
            os.system('copy {} {}'.format(src, des))


def train(num_epoch, train_loader, loss_fun, validation_loader, model, optimizer, save_per_epoch, save_dir, use_cuda=False):
    model.train()
    for current_epoch in range(num_epoch):
        total_loss = 0

        for batch_idx, (input_img, target) in enumerate(train_loader):
            if use_cuda:
                input_img, target = input_img.cuda(), target.cuda()

            optimizer.zero_grad()
            # print(input_img.shape)
            # 计算网络输出
            output = model(input_img)
            # 计算损失、梯度和做反向传播
            loss = loss_fun(output, target)
            total_loss += loss
            loss.backward()
            optimizer.step()

        if current_epoch % save_per_epoch == 0:
            LOG('<=========================== Epoch {} ===========================>'.format(current_epoch))
            validate(validation_loader, model)
            torch.save(model.state_dict(), f"{save_dir}/nextloc_{current_epoch}.pt")
            torch.save(optimizer.state_dict(), f"{save_dir}/nextloc_optim_backup.pt")
            LOG('<=== Model and optimizer have been saved ====>')


def validate(validation_loader, model, use_cuda = False):
    model.eval()
    total_items = len(validation_loader)
    total_loss = 0

    for batch_idx, (input_img, target) in enumerate(validation_loader):
        if use_cuda:
            input_img, target = input_img.cuda(), target.cuda()

        with torch.no_grad():
            print(input_img.shape)
            output = model(input_img)
            loss = F.cross_entropy(output, target)
        total_loss += loss.cpu().data.numpy()

    LOG('validation items is: {}'.format(total_items))
    LOG('total_loss is: {}'.format(total_loss))
    model.train()

# ---------------------------------------------------------------------------------------


if __name__ == '__main__':

    logfile = open(f"{args['save_dir']}/log.txt", 'w')
    def LOG(msg):
        print(msg)
        logfile.write(msg + '\n')
        logfile.flush()


    LOG('# ----- Divide dataset to train_dataset and validation_dataset ----- #')
    divide_dataset(args['data_folder'], args['train_percentage'])
    LOG('done')

    LOG('# ----- Building train datasets ----- #')
    train_dataset = LocDataset(args['data_folder'] + '/loc_train', args['block_category'])
    train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], drop_last=True)
    LOG('done')

    LOG('# ----- Building validation datasets ----- #')
    validation_dataset = LocDataset(args['data_folder'] + '/loc_validation', args['block_category'])
    validation_dataloader = DataLoader(validation_dataset, batch_size=args['batch_size'], drop_last=True)
    LOG('done')

    LOG('# ----- Building model ----- #')
    input_channels = args['counts_house_categories'] + 4
    if args['use_cuda']:
        model = Location(args['counts_house_categories']+1, input_channels).cuda()
    else:
        model = Location(args['counts_house_categories']+1, input_channels)

    LOG('# ----- Building optimizer、CrossEntropyLoss----- #')
    optimizer = optim.Adam(list(model.parameters()), lr=args['lr'], weight_decay=2e-6)
    weight = [args['centroid_weight'] for i in range(args['counts_house_categories'] + 1)]
    weight[0] = 1
    print(weight)

    if args['use_cuda']:
        weight = torch.from_numpy(np.asarray(weight)).float().cuda()
    else:
        weight = torch.from_numpy(np.asarray(weight)).float()
    cross_entropy = nn.CrossEntropyLoss(weight=weight)
    softmax = nn.Softmax()
    LOG('done')

    LOG('# ----- Basic information ----- #')
    LOG('train epochs: {}'.format(args['epoch']))
    LOG('save_per_epoch: {}'.format(args['save_every_n_epochs']))
    LOG('batch_size: {}'.format(args['batch_size']))
    LOG('block_category: {}\n'.format(args['block_category']))

    LOG('# ----- train ----- #')
    train(args['epoch'], train_dataloader, cross_entropy, validation_dataloader, model, optimizer, args['save_every_n_epochs'], args['save_dir'], use_cuda=False)

    logfile.close()