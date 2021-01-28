import argparse
import torch
import torch.optim as optim
from category_dataset import CategoryDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from models import *
import os
from random import shuffle

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
        'epoch': 8,
        'save_every_n_epochs': 1,
        'block_category': 3,
        'counts_house_categories': 10,
        'lr': 0.0005,
        'batch_size': 3,
        'latent_dim': 200,
        'train_percentage': 0.8,
        'validation_percentage': 0.2,
        'use_cuda': False
        }

if not os.path.exists(args['save_dir']):
    os.mkdir(args['save_dir'])

# ---------------------------------------------------------------------------------------
class NextCategory(nn.Module):

    def __init__(self, n_input_channels, n_categories, bottleneck_size):
        super(NextCategory, self).__init__()

        activation = nn.LeakyReLU()

        self.cat_prior_img = nn.Sequential(
            resnet18(num_input_channels=n_input_channels, num_classes=bottleneck_size),
            nn.BatchNorm1d(bottleneck_size),
            activation
        )
        self.cat_prior_counts = nn.Sequential(
            nn.Linear(n_categories, bottleneck_size),
            nn.BatchNorm1d(bottleneck_size),
            activation,
            nn.Linear(bottleneck_size, bottleneck_size),
            nn.BatchNorm1d(bottleneck_size),
            activation
        )
        self.cat_prior_final = nn.Sequential(
            nn.Linear(2*bottleneck_size, bottleneck_size),
            nn.BatchNorm1d(bottleneck_size),
            activation,
            # +1 -> the 'stop' category
            nn.Linear(bottleneck_size, n_categories+1)
        )

    def forward(self, input_scene, catcount):
        cat_img = self.cat_prior_img(input_scene)
        cat_count = self.cat_prior_counts(catcount)
        catlogits = self.cat_prior_final(torch.cat([cat_img, cat_count], dim=-1))
        return catlogits
# ---------------------------------------------------------------------------------------


def divide_dataset(source, train_percentage):
    cwd = os.getcwd()  # 返回当前工作目录
    dir_name = [source + '/train', source + '/validation']
    # 创建训练集train和测试集validation文件夹
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


def train(num_epoch, train_loader, train_dataset, validation_loader, model, optimizer, save_per_epoch, save_dir, use_cuda=False):
    model.train()
    for current_epoch in range(num_epoch):
        # total_loss = 0

        for batch_idx, (input_img, predict_categories, existing_categories_counts) in enumerate(train_loader):

            predict_categories = torch.squeeze(predict_categories)

            if use_cuda:
                input_img, predict_categories, existing_categories_counts = input_img.cuda(), predict_categories.cuda(), existing_categories_counts.cuda()

            # 计算网络输出
            output = model(input_img, existing_categories_counts)

            # 计算损失、梯度和做反向传播
            loss = F.cross_entropy(output, predict_categories)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if current_epoch % save_per_epoch == 0:
            LOG('<=========================== Epoch {} ===========================>'.format(current_epoch))
            LOG('p_removing: {}'.format(train_dataset.p_removing))
            validate(validation_loader, model)
            torch.save(model.state_dict(), f"{save_dir}/nextcat_{current_epoch}.pt")
            torch.save(optimizer.state_dict(), f"{save_dir}/nextcat_optim_backup.pt")
            LOG('<=== Model and optimizer have been saved ====>')


        if current_epoch <= 1:
            train_dataset.p_removing = 0.0
        if current_epoch > 2:
            train_dataset.p_removing = 0.3
        if current_epoch > 3:
            train_dataset.p_removing = 0.5
        if current_epoch > 4:
            train_dataset.p_removing = 0.7


def validate(validation_loader, model, use_cuda = False):
    model.eval()
    total_items = len(validation_loader)
    total_correct = 0
    total_loss = 0
    total_accuracy = 0

    for batch_idx, (input_img, predict_categories, existing_categories_counts) in enumerate(validation_loader):
        predict_categories = torch.squeeze(predict_categories)

        if use_cuda:
            input_img, predict_categories, existing_categories_counts = input_img.cuda(), predict_categories.cuda(), existing_categories_counts.cuda()

        with torch.no_grad():
            print(input_img.shape)
            output = model(input_img, existing_categories_counts)
            loss = F.cross_entropy(output, predict_categories)

        total_loss += loss.cpu().data.numpy()
        predict_cat = output.max(-1)[1]
        total_correct += (predict_cat == predict_categories).sum()

    total_accuracy = total_correct / total_items

    LOG('validation items is: {}'.format(total_items))
    LOG('total_loss is: {}'.format(total_loss))
    LOG('total_accuracy is {}'.format(total_accuracy))
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
    train_dataset = CategoryDataset(args['data_folder'] + '/train', args['block_category'], args['counts_house_categories'], 0.0)
    train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], drop_last=True)
    LOG('done')

    LOG('# ----- Building validation datasets ----- #')
    validation_dataset = CategoryDataset(args['data_folder'] + '/validation', args['block_category'], args['counts_house_categories'], 0.4)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args['batch_size'], drop_last=True)
    LOG('done')

    LOG('# ----- Building model ----- #')
    input_channels = args['counts_house_categories'] + 4
    if args['use_cuda']:
        model = NextCategory(input_channels, args['counts_house_categories'], args['latent_dim']).cuda()
    else:
        model = NextCategory(input_channels, args['counts_house_categories'], args['latent_dim'])

    LOG('# ----- Building optimizer ----- #')
    optimizer = optim.Adam(list(model.parameters()), lr=args['lr'])
    LOG('done')

    LOG('# ----- Basic information ----- #')
    LOG('train epochs: {}'.format(args['epoch']))
    LOG('save_per_epoch: {}'.format(args['save_every_n_epochs']))
    LOG('batch_size: {}'.format(args['batch_size']))
    LOG('block_category: {}\n'.format(args['block_category']))

    LOG('# ----- train ----- #')
    train(args['epoch'], train_dataloader, train_dataset, validation_dataloader, model, optimizer, args['save_every_n_epochs'], args['save_dir'], use_cuda=False)

    logfile.close()