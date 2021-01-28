import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as F
import torchvision
import numpy as np
import random
import math
from random import shuffle
from models.utils import *
from orient_dataset import OrientDataset
from torch.utils.data import DataLoader
import utils
import os
import pdb

"""
Module that predicts the orientation of the next object
"""
# ---------------------------------------------------------------------------------------
args = {'data_folder': './txt_data_divide',
        'save_dir': './trainedModels',
        'epoch': 4,
        'save_every_n_epochs': 1,
        'block_category': 3,
        'counts_house_categories': 10,
        'lr': 0.0005,
        'batch_size': 3,
        'centroid_weight': 10,
        'train_percentage': 0.8,
        'validation_percentage': 0.2,
        'latent_size': 10,
        'hidden_size': 40,
        'use_cuda': True
        }

if not os.path.exists(args['save_dir']):
    os.mkdir(args['save_dir'])


# ---------------------------------------------------------------------------------------

class Orientation(nn.Module):

    def make_net_fn(self, netdict, makefn):
        def net_fn(cat):
            # We assume that the data loader has extracted the single category index
            #    that we'll be using for this batch
            cat = str(cat)
            if cat in netdict:
                return netdict[cat]
            else:
                net = makefn().cuda()
                # net = makefn()
                netdict[cat] = net
                return net

        return net_fn

    def __init__(self, latent_size, hidden_size, num_input_channels):
        super(Orientation, self).__init__()
        self.latent_size = latent_size
        self.testing = False

        def make_encoder():
            return nn.Sequential(
                nn.Linear(2, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, 2 * latent_size)
            )

        def make_cond_prior():
            return nn.Sequential(
                # 64 -> 32
                DownConvBlock(num_input_channels, 8),
                # 32 -> 16
                DownConvBlock(8, 16),
                # 16 -> 8
                DownConvBlock(16, 32),
                # 8 -> 4
                DownConvBlock(32, 64),
                # 4 -> 1
                nn.AdaptiveAvgPool2d(1),
                # Final linear layer
                Reshape(-1, 64),
                nn.Linear(64, latent_size)
            )

        def make_generator():
            return nn.Sequential(
                nn.Linear(2 * latent_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, 2)
            )

        def make_discriminator():
            return nn.Sequential(
                # 64 -> 32
                DownConvBlock(num_input_channels + 2, 8),
                # 32 -> 16
                DownConvBlock(8, 16),
                # 16 -> 8
                DownConvBlock(16, 32),
                # 8 -> 4
                DownConvBlock(32, 64),
                # 4 -> 1
                nn.AdaptiveAvgPool2d(1),
                # Final linear layer
                Reshape(-1, 64),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        self.encoders = nn.ModuleDict()
        self.cond_priors = nn.ModuleDict()
        self.generators = nn.ModuleDict()
        self.discriminators = nn.ModuleDict()

        self.encoder = self.make_net_fn(self.encoders, make_encoder)
        self.cond_prior = self.make_net_fn(self.cond_priors, make_cond_prior)
        self.generator = self.make_net_fn(self.generators, make_generator)
        self.discriminator = self.make_net_fn(self.discriminators, make_discriminator)

    def encode(self, t_orient, cat):
        mu_logvar = self.encoder(cat)(t_orient)
        return torch.split(mu_logvar, self.latent_size, dim=1)

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        gdis = torch.distributions.Normal(mu, std)
        return gdis.rsample()

    def generate(self, noise, walls, cat):

        enc_walls = self.cond_prior(cat)(walls)
        gen_out = self.generator(cat)(torch.cat([noise, enc_walls], dim=1))
        go1, go2 = torch.split(gen_out, 1, dim=1)
        orient_x = F.tanh(go1)
        orient_y = torch.sqrt(1.0 - orient_x * orient_x)
        y_sign_p = F.sigmoid(go2)
        if self.testing:
            y_sign = torch.where(y_sign_p > 0.5, torch.ones_like(orient_y), -torch.ones_like(orient_y))
            orient_y *= y_sign
            orient = torch.stack([orient_x, orient_y], dim=1).squeeze()
            if len(list(orient.size())) == 1:
                orient = orient.unsqueeze(0)
            return orient
        else:
            return orient_x, y_sign_p

    def discriminate(self, walls, loc, orient, dims, cat):
        sdf = render_oriented_sdf((256, 256), dims, loc, orient)
        return self.discriminator(cat)(torch.cat([sdf, walls], dim=1))

    def set_requires_grad(self, phase, cat):
        if phase == 'D':
            set_requires_grad(self.generator(cat), False)
            set_requires_grad(self.discriminator(cat), True)
            set_requires_grad(self.encoder(cat), False)
            set_requires_grad(self.cond_prior(cat), False)
        elif phase == 'G':
            set_requires_grad(self.generator(cat), True)
            set_requires_grad(self.discriminator(cat), False)
            set_requires_grad(self.encoder(cat), False)
            set_requires_grad(self.cond_prior(cat), True)
        elif phase == 'VAE':
            set_requires_grad(self.generator(cat), True)
            set_requires_grad(self.discriminator(cat), False)
            set_requires_grad(self.encoder(cat), True)
            set_requires_grad(self.cond_prior(cat), True)
        else:
            raise ValueError(f'Unrecognized phase {phase}')

    def save(self, filename):
        torch.save({
            'cats_seen': list(self.generators.keys()),
            'state': self.state_dict()
        }, filename)

    def load(self, filename):
        blob = torch.load(filename)
        for cat in blob['cats_seen']:
            _ = self.encoder(cat)
            _ = self.cond_prior(cat)
            _ = self.generator(cat)
            _ = self.discriminator(cat)
        self.load_state_dict(blob['state'])


class Optimizers:

    def make_optimizer_fn(self, optimizers, list_of_netfns):
        this = self

        def optimizer_fn(cat):
            cat = str(cat)
            if cat in optimizers:
                return optimizers[cat]
            else:
                params = []
                for netfn in list_of_netfns:
                    params.extend(list(netfn(cat).parameters()))
                optimizer = optim.Adam(params, lr=this.lr)
                optimizers[cat] = optimizer
                return optimizer

        return optimizer_fn

    def __init__(self, model, lr):
        self.lr = lr
        self.g_optimizers = {}
        self.d_optimizers = {}
        self.e_optimizers = {}
        self.g_optimizer = self.make_optimizer_fn(self.g_optimizers, [model.generator, model.cond_prior])
        self.d_optimizer = self.make_optimizer_fn(self.d_optimizers, [model.discriminator])
        self.e_optimizer = self.make_optimizer_fn(self.e_optimizers, [model.encoder])

    def save(self, filename):
        g_state = {cat: opt.state_dict() for cat, opt in self.g_optimizers.items()}
        d_state = {cat: opt.state_dict() for cat, opt in self.d_optimizers.items()}
        e_state = {cat: opt.state_dict() for cat, opt in self.e_optimizers.items()}
        torch.save([g_state, d_state, e_state], filename)

    def load(self, filename):
        states = torch.load(filename)
        g_state = states[0]
        d_state = states[1]
        e_state = states[2]
        for cat, state in g_state:
            self.g_optimizer(cat).load_state_dict(state)
        for cat, state in d_state:
            self.d_optimizer(cat).load_state_dict(state)
        for cat, state in e_state:
            self.e_optimizer(cat).load_state_dict(state)


# ---------------------------------------------------------------------------------------

def divide_dataset(source, train_percentage):
    cwd = os.getcwd()  # 返回当前工作目录
    dir_name = [source + '/orient_train', source + '/orient_validation']
    # 创建训练集orient_train和测试集orient_validation文件夹
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


# ---------------------------------------------------------------------------------------

def train(num_epoch, train_loader, validation_loader, model, optimizer, save_per_epoch, save_dir, use_cuda=False):
    # dataset.prepare_same_category_batches(batch_size)
    model.train()
    for current_epoch in range(num_epoch):
        for batch_idx, (input_img, t_counts, t_cat, t_loc, t_orient) in enumerate(train_loader):
            t_cat = torch.squeeze(t_cat)
            # # Verify that we've got only one category in this batch
            # t_cat_0 = t_cat[0]
            # assert ((t_cat == t_cat_0).all())
            # t_cat = t_cat_0.item()

            actual_batch_size = input_img.shape[0]

            # pdb.set_trace()
            if args['use_cuda']:
                input_img, t_counts, t_cat, t_loc, t_orient = input_img.cuda(), t_counts.cuda(), t_cat.cuda(), t_loc.cuda(), t_orient.cuda()

            d_loc, d_orient = default_loc_orient(actual_batch_size)
            input_img = inverse_xform_img(input_img, t_loc, d_orient, 256)

            model.set_requires_grad('VAE', t_cat)
            optimizers.g_optimizer(t_cat).zero_grad()
            optimizers.e_optimizer(t_cat).zero_grad()

            # pdb.set_trace()
            mu, logvar = model.encode(t_orient, t_cat)
            kld_loss = unitnormal_normal_kld(mu, logvar)
            z = model.sample(mu, logvar)

            # t_orient: cos sin
            fake_x, fake_ysign_p = model.generate(z, input_img, t_cat)
            real_x = t_orient[:, 0]  # cos
            real_ysign = (t_orient[:, 1] >= 0).float()  # sin
            x_recon_loss = F.l1_loss(fake_x.squeeze(), real_x)
            y_recon_loss = F.binary_cross_entropy(fake_ysign_p, real_ysign)
            recon_loss = x_recon_loss + y_recon_loss

            vae_loss = recon_loss + kld_loss
            vae_loss.backward()
            optimizers.g_optimizer(t_cat).step()
            optimizers.e_optimizer(t_cat).step()

        if current_epoch % save_per_epoch == 0:
            LOG('<=========================== Epoch {} ===========================>'.format(current_epoch))
            validate(validation_loader, model)
            model.save(f'{save_dir}/model_{current_epoch}.pt')
            optimizers.save(f'{save_dir}/opt_{current_epoch}.pt')
            LOG('<=== Model and optimizer have been saved ====>')


def validate(validation_loader, model, use_cuda=False):
    model.eval()
    total_items = len(validation_loader)
    total_loss = 0.0

    # valid_dataset.prepare_same_category_batches(batch_size)
    for batch_idx, (input_img, t_counts, t_cat, t_loc, t_orient) in enumerate(validation_loader):
        t_cat = torch.squeeze(t_cat)
        # Verify that we've got only one category in this batch
        # t_cat_0 = t_cat[0]
        # assert ((t_cat == t_cat_0).all())
        # t_cat = t_cat_0.item()

        actual_batch_size = input_img.shape[0]

        input_img, t_counts, t_cat, t_loc, t_orient = input_img.cuda(), t_counts.cuda(), t_cat.cuda(), t_loc.cuda(), t_orient.cuda()
        d_loc, d_orient = default_loc_orient(actual_batch_size)
        input_img = inverse_xform_img(input_img, t_loc, d_orient, 256)

        mu, logvar = model.encode(t_orient, t_cat)
        kld_loss = unitnormal_normal_kld(mu, logvar)
        z = model.sample(mu, logvar)

        fake_x, fake_ysign_p = model.generate(z, input_img, t_cat)
        real_x = t_orient[:, 0]  # cos
        real_ysign = (t_orient[:, 1] >= 0).float()  # sin
        x_recon_loss = F.l1_loss(fake_x.squeeze(), real_x)
        y_recon_loss = F.binary_cross_entropy(fake_ysign_p, real_ysign)
        recon_loss = x_recon_loss + y_recon_loss

        vae_loss = recon_loss + kld_loss

        total_loss += vae_loss

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
    train_dataset = OrientDataset(args['data_folder'] + '/orient_train', args['block_category'],
                                  args['counts_house_categories'])
    train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], drop_last=True)
    LOG('done')

    LOG('# ----- Building validation datasets ----- #')
    validation_dataset = OrientDataset(args['data_folder'] + '/orient_validation', args['block_category'],
                                       args['counts_house_categories'])
    validation_dataloader = DataLoader(validation_dataset, batch_size=args['batch_size'], drop_last=True)
    LOG('done')

    LOG('# ----- Building model ----- #')
    input_channels = args['counts_house_categories'] + 4
    if args['use_cuda']:
        model = Orientation(args['latent_size'], args['hidden_size'], input_channels).cuda()
    else:
        model = Orientation(args['latent_size'], args['hidden_size'], input_channels)

    LOG('# ----- Building optimizer、CrossEntropyLoss----- #')
    optimizers = Optimizers(model=model, lr=0.0005)
    LOG('done')

    LOG('# ----- Basic information ----- #')
    LOG('train epochs: {}'.format(args['epoch']))
    LOG('save_per_epoch: {}'.format(args['save_every_n_epochs']))
    LOG('batch_size: {}'.format(args['batch_size']))
    LOG('block_category: {}\n'.format(args['block_category']))

    LOG('# ----- train ----- #')
    train(args['epoch'], train_dataloader, validation_dataloader, model, optimizers, args['save_every_n_epochs'],
          args['save_dir'], use_cuda=False)

    logfile.close()
