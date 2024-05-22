from fnmatch import translate
from re import I
import torch
import torch.nn.functional as F
import random
import numpy as np
from torch.autograd.variable import Variable
from tools.utilize import *
from model.unit.unit import *
from metrics.metrics import mae, psnr, ssim 
from tools.visualize import plot_sample


###ADDED LIBRARY###
import kornia as kt

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import kornia.geometry.transform as kt

__all__ = ['Base']

class Base():
    def __init__(self, config, train_loader, valid_loader, assigned_loader,
                 device, file_path, batch_limit_weight=1.0):

        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.assigned_loader = assigned_loader
        self.device = device
        self.file_path = file_path
        self.batch_limit_weight = batch_limit_weight
        self.valid = 1
        self.fake = 0
        self.batch_size = config['batch_size']


        # # fid stats
        # self.fid_stats_from_a_to_b = '{}/{}/{}_{}_fid_stats.npz'.format(
        #     self.config['fid_dir'], self.config['dataset'], self.config['source_domain'], self.config['target_domain'])
        # self.fid_stats_from_b_to_a = '{}/{}/{}_{}_fid_stats.npz'.format(
        #     self.config['fid_dir'], self.config['dataset'], self.config['target_domain'], self.config['source_domain'])

        # model, two modality, 1 and 2, the aim is to generate 2 from 1
        self.generator_from_a_to_b_enc = None
        self.generator_from_b_to_a_enc = None
        self.generator_from_a_to_b_dec = None
        self.generator_from_b_to_a_dec = None
        self.discriminator_from_a_to_b = None
        self.discriminator_from_b_to_a = None

        # loss
        self.criterion_recon = torch.nn.L1Loss().to(self.device)
        self.criterion_pixel = torch.nn.L1Loss().to(device)
        self.criterion_gan_from_a_to_b = torch.nn.MSELoss().to(device)
        self.criterion_gan_from_b_to_a = torch.nn.MSELoss().to(device)
        self.criterion_pixelwise_from_a_to_b = torch.nn.L1Loss().to(device)
        self.criterion_pixelwise_from_b_to_a = torch.nn.L1Loss().to(device)
        self.criterion_identity = torch.nn.L1Loss().to(device)
        self.criterion_sr = torch.nn.L1Loss().to(device)

        # optimizer
        self.optimizer_generator = None
        self.optimizer_discriminator_from_a_to_b = None
        self.optimizer_discriminator_from_b_to_a = None

        self.lr_scheduler_generator = None
        self.lr_scheduler_discriminator_from_a_to_b = None
        self.lr_scheduler_discriminator_from_b_to_a = None

        # if self.config['reg_gan']:
        #     self.optimizer_reg = torch.optim.Adam(self.reg.parameters(),
        #                         lr=self.config['lr'], betas=[self.config['beta1'], self.config['beta2']])

        self.batch_limit = int(self.config['data_num'] * self.batch_limit_weight / self.config['batch_size'])
        if self.config['debug']:
            self.batch_limit = 2

    def calculate_basic_gan_loss(self, images):
        pass

    def train_epoch(self, inf=''):
        for i, batch in enumerate(self.train_loader):
            if i > self.batch_limit:
                break

            """
            Train Generators
            """

            self.optimizer_generator.zero_grad()

            imgs, tmps = self.collect_generated_images(batch=batch)
            real_a, real_b, fake_a, fake_b, fake_fake_a, fake_fake_b = imgs

            # gan loss
            loss_gan_basic = self.calculate_basic_gan_loss([imgs, tmps])

            # total loss
            loss_generator_total = loss_gan_basic

            # reg loss
            if self.config['reg_gan']:
                self.optimizer_reg.zero_grad()
                reg_trans = self.reg(fake_b, real_b)
                sysregist_from_a_to_b = self.spatial_transformer(fake_b, reg_trans, self.device)
                loss_sr = self.criterion_sr(sysregist_from_a_to_b, real_b)
                # loss_sm = smooothing_loss(reg_trans)

            # idn loss
            if self.config['identity']:
                loss_identity_fake_b = self.criterion_identity(fake_b, fake_fake_a)
                loss_identity_fake_a = self.criterion_identity(fake_a, fake_fake_b)
                loss_identity = self.config['lambda_identity'] * (loss_identity_fake_a + loss_identity_fake_b)
                loss_generator_total += loss_identity

            # reg loss
            # if self.config['reg_gan']:
            #     loss_reg = self.config['lambda_corr'] * loss_sr + self.config['lambda_smooth'] * loss_sm
            #     loss_generator_total += loss_reg

            loss_generator_total.backward()

            # torch.nn.utils.clip_grad_norm_(self.generator_from_a_to_b_enc.parameters(), max_norm=5, norm_type=2)
            # torch.nn.utils.clip_grad_norm_(self.generator_from_a_to_b_dec.parameters(), max_norm=5, norm_type=2)
            # torch.nn.utils.clip_grad_norm_(self.generator_from_b_to_a_enc.parameters(), max_norm=5, norm_type=2)
            # torch.nn.utils.clip_grad_norm_(self.generator_from_b_to_a_dec.parameters(), max_norm=5, norm_type=2)

            self.optimizer_generator.step()

            if self.config['reg_gan']:
                self.optimizer_reg.step()

            """
            Train Discriminator
            """
            self.optimizer_discriminator_from_a_to_b.zero_grad()
            self.optimizer_discriminator_from_b_to_a.zero_grad()

            loss_discriminator_from_a_to_b = self.discriminator_from_a_to_b.compute_loss(
                real_a, self.valid) + self.discriminator_from_a_to_b.compute_loss(fake_a.detach(), self.fake)
            loss_discriminator_from_b_to_a = self.discriminator_from_b_to_a.compute_loss(
                real_b, self.valid) + self.discriminator_from_b_to_a.compute_loss(fake_b.detach(), self.fake)

            loss_discriminator_from_a_to_b_total = loss_discriminator_from_a_to_b
            loss_discriminator_from_b_to_a_total = loss_discriminator_from_b_to_a

            loss_discriminator_from_a_to_b_total.backward(retain_graph=True)
            loss_discriminator_from_b_to_a_total.backward(retain_graph=True)

            # torch.nn.utils.clip_grad_norm_(self.discriminator_from_a_to_b.parameters(), max_norm=5, norm_type=2)
            # torch.nn.utils.clip_grad_norm_(self.discriminator_from_b_to_a.parameters(), max_norm=5, norm_type=2)

            self.optimizer_discriminator_from_b_to_a.step()
            self.optimizer_discriminator_from_a_to_b.step()

            # print log
            infor = '\r{}[Batch {}/{}] [Gen loss: {:.4f}] [Dis loss: {:.4f}, {:.4f}]'.format(inf, i, self.batch_limit,
                        loss_generator_total.item(), loss_discriminator_from_a_to_b.item(), loss_discriminator_from_b_to_a.item())

            if self.config['identity']:
                infor = '{} [Idn Loss: {:.4f}]'.format(infor, loss_identity.item())

            # if self.config['reg_gan']:
            #     infor = '{} [Reg Loss: {:.4f}]'.format(infor, loss_reg.item())
                
            print(infor, flush=True, end=' ')

        # update learning rates
        self.lr_scheduler_generator.step()
        self.lr_scheduler_discriminator_from_a_to_b.step()
        self.lr_scheduler_discriminator_from_b_to_a.step()


    def collect_compute_result_for_evaluation(self):
        # initialize fake_b_list
        fake_b_list = torch.randn(self.config['batch_size'], 1, self.config['size'], self.config['size'])
        fake_a_list = torch.randn(self.config['batch_size'], 1, self.config['size'], self.config['size'])
        # to reduce gpu memory for evaluation
        mae_list, psnr_list, ssim_list = [], [], []
        for i, batch in enumerate(self.valid_loader):
            imgs, tmps = self.collect_generated_images(batch=batch)
            real_a, real_b, fake_a, fake_b, fake_fake_a, fake_fake_b = imgs

            mae_value = mae(real_b, fake_b) 
            psnr_value = psnr(real_b, fake_b)
            ssim_value = ssim(real_b, fake_b)
            mae_list.append(mae_value)
            psnr_list.append(psnr_value)
            ssim_list.append(ssim_value)

        
        return fake_b_list, mae_list, psnr_list,  ssim_list

    @torch.no_grad()
    def evaluation(self, direction='from_a_to_b'):
    
        fake_list = torch.randn(self.config['batch_size'], 1, self.config['size'], self.config['size'])

        if direction == 'both':
            a_mae_list, a_psnr_list, a_ssim_list = [], [], []
            b_mae_list, b_psnr_list, b_ssim_list = [], [], []
            a_fid = 0
            b_fid = 0
        else:
            mae_list, psnr_list, ssim_list = [], [], []
            fid_value = 0

        if direction == 'from_b_to_a':
            for i, batch in enumerate(self.valid_loader):
                imgs, tmps = self.collect_generated_images(batch=batch)
                real_a, real_b, fake_a, fake_b, fake_fake_a, fake_fake_b = imgs



                mae_value = mae(real_a, fake_a) 
                psnr_value = psnr(real_a, fake_a)
                ssim_value = ssim(real_a, fake_a)
                mae_list.append(mae_value)
                psnr_list.append(psnr_value)
                ssim_list.append(ssim_value)

            return average(mae_list), average(psnr_list), average(ssim_list), fid_value     

        elif direction == 'from_a_to_b':
            for i, batch in enumerate(self.valid_loader):
                imgs, tmps = self.collect_generated_images(batch=batch)
                real_a, real_b, fake_a, fake_b, fake_fake_a, fake_fake_b = imgs


                mae_value = mae(real_b, fake_b) 
                psnr_value = psnr(real_b, fake_b)
                ssim_value = ssim(real_b, fake_b)
                mae_list.append(mae_value)
                psnr_list.append(psnr_value)
                ssim_list.append(ssim_value)


            return average(mae_list), average(psnr_list), average(ssim_list) #, fid_value     
        
        elif direction == 'both':
            for i, batch in enumerate(self.valid_loader):
                imgs, _ = self.collect_generated_images(batch=batch)
                real_a, real_b, fake_a, fake_b, _, _ = imgs

                # mae                
                a_mae = mae(real_a, fake_a)
                b_mae = mae(real_b, fake_b)

                # psnr
                a_psnr = psnr(real_a, fake_a)
                b_psnr = psnr(real_b, fake_b)

                # ssim
                a_ssim = ssim(real_a, fake_a)
                b_ssim = ssim(real_b, fake_b)

                a_mae_list.append(a_mae)
                a_psnr_list.append(a_psnr)
                a_ssim_list.append(a_ssim)

                b_mae_list.append(b_mae)
                b_psnr_list.append(b_psnr)
                b_ssim_list.append(b_ssim)

            
            return (average(a_mae_list), average(a_psnr_list), average(a_ssim_list), a_fid,
                    average(b_mae_list), average(b_psnr_list), average(b_ssim_list), b_fid) 
        else:
            raise NotImplementedError('Direction Has Not Been Implemented Yet')

    def get_model(self, description='centralized'):
        return self.generator_from_a_to_b_enc, self.generator_from_b_to_a_enc, self.generator_from_a_to_b_dec,\
             self.generator_from_b_to_a_dec, self.discriminator_from_a_to_b, self.discriminator_from_b_to_a

    def set_model(self, gener_from_a_to_b_enc, gener_from_a_to_b_dec, gener_from_b_to_a_enc,\
                gener_from_b_to_a_dec, discr_from_a_to_b, discr_from_b_to_a):
        self.generator_from_a_to_b_enc = gener_from_a_to_b_enc
        self.generator_from_a_to_b_dec = gener_from_a_to_b_dec
        self.generator_from_b_to_a_enc = gener_from_b_to_a_enc
        self.generator_from_b_to_a_dec = gener_from_b_to_a_dec
        self.discriminator_from_a_to_b = discr_from_a_to_b
        self.discriminator_from_b_to_a = discr_from_b_to_a

    def collect_generated_images(self, batch):
        pass


    def collect_feature(self, batch):
        pass

    @torch.no_grad()
    def infer_images(self, save_img_path, data_loader):
        for i, batch in enumerate(data_loader):
            imgs, tmps = self.collect_generated_images(batch=batch)
            real_a, real_b, fake_a, fake_b, fake_fake_a, fake_fake_b = imgs
            if i <= self.config['num_img_save']:
                img_path = '{}/{}-slice-{}'.format(
                    save_img_path, batch['name_a'][0], batch['slice_num'].numpy()[0])

                mae_value = mae(real_b, fake_b).item() 
                psnr_value = psnr(real_b, fake_b).item()
                ssim_value = ssim(real_b, fake_b).item()
                    
                img_all = torch.cat((real_a, real_b, fake_a, fake_b, fake_fake_a, fake_fake_b), 0)
                save_image(img_all, 'all_m_{:.4f}_p_{:.4f}_s_{:.4f}.png'.format(mae_value, psnr_value, ssim_value), img_path)

                save_image(real_a, 'real_a.png', img_path)
                save_image(real_b, 'real_b.png', img_path)
                save_image(fake_a, 'fake_a.png', img_path)
                save_image(fake_b, 'fake_b.png', img_path)
                save_image(fake_fake_a, 'fake_fake_a.png', img_path)
                save_image(fake_fake_b, 'fake_fake_b.png', img_path)

            
    @torch.no_grad()
    def visualize_feature(self, epoch, save_img_path, data_loader):
        real_a, fake_a, real_b, fake_b = [], [], [], []
        for i, batch in enumerate(data_loader):
            if i == int(self.config['plot_num_sample']):
                break
            real_a_feature, fake_a_feature, real_b_feature, fake_b_feature = self.collect_feature(batch=batch)

            real_a_feature = np.mean(real_a_feature.cpu().detach().numpy().reshape(
                len(batch[self.config['source_domain']]), 512, 8, 4, 2), axis=(1, 2, 3))
            fake_a_feature = np.mean(fake_a_feature.cpu().detach().numpy().reshape(
                len(batch[self.config['source_domain']]), 512, 8, 4, 2), axis=(1, 2, 3))
            real_b_feature = np.mean(real_b_feature.cpu().detach().numpy().reshape(
                len(batch[self.config['source_domain']]), 512, 8, 4, 2), axis=(1, 2, 3))
            fake_b_feature = np.mean(fake_b_feature.cpu().detach().numpy().reshape(
                len(batch[self.config['source_domain']]), 512, 8, 4, 2), axis=(1, 2, 3))

            if i == 0:
                real_a = real_a_feature
                fake_a = fake_a_feature
                real_b = real_b_feature
                fake_b = fake_b_feature
            else:
                real_a = np.concatenate([real_a, real_a_feature], axis=0)
                fake_a = np.concatenate([fake_a, fake_a_feature], axis=0)
                real_b = np.concatenate([real_b, real_b_feature], axis=0)
                fake_b = np.concatenate([fake_b, fake_b_feature], axis=0)

        plot_sample(real_a, fake_a, real_b, fake_b, step=epoch, img_path=save_img_path, descript='Epoch')

        with open(save_img_path.replace('.png', '.npy'), 'wb') as f:
            np.save(f, np.array([real_a, fake_a, real_b, fake_b]))