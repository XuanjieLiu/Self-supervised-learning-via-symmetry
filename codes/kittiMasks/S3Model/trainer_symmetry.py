import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from codes.kittiMasks.S3Model.eval_linear_proj import calc_group_mse
from codes.S3Ball.shared import DEVICE
from codes.kittiMasks.S3Model.normal_rnn import Conv2dGruConv2d
# temp_dir = path.abspath(path.join(
#     path.dirname(__file__), '../..',
# ))
# sys.path.append(temp_dir)
from codes.kittiMasks.kitti_mask_dataloader_seqCut import KittiMasks
from codes.S3Ball.symmetry import make_translation_batch, make_rotation_batch, do_seq_symmetry, symm_trans, symm_rota
from codes.loss_counter import LossCounter
from codes.common_utils import create_results_path_if_not_exist
# assert sys.path.pop(-1) == temp_dir


def is_need_train(train_config):
    loss_counter = LossCounter([])
    iter_num = loss_counter.load_iter_num(train_config['train_record_path'])
    if train_config['max_iter_num'] > iter_num:
        print("Continue training", flush=True)
        return True
    else:
        print("No more training is needed", flush=True)
        return False


def vector_z_score_norm(vector, mean=None, std=None):
    if mean is None:
        mean = torch.mean(vector, [k for k in range(vector.ndim - 1)])
    if std is None:
        std = torch.std(vector, [j for j in range(vector.ndim - 1)])
    return (vector - mean) / std, mean, std


class Trainer:
    def __init__(self, config, is_train=True):
        train_dataset = KittiMasks(path=config['train_data_path'])
        eval_dataset = KittiMasks(path=config['eval_data_path'])
        self.train_data_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        self.model = Conv2dGruConv2d(config).to(DEVICE)
        self.eval_data_loader = DataLoader(eval_dataset, batch_size=config['batch_size'], shuffle=False)
        self.mse_loss = nn.MSELoss(reduction='sum').to(DEVICE)
        self.model.to(DEVICE)
        self.model_path = config['model_path']
        self.checkpoint_path = config['checkpoint_path']
        self.kld_loss_scalar = config['kld_loss_scalar']
        self.z_rnn_loss_scalar = config['z_rnn_loss_scalar']
        self.z_symm_loss_scalar = config['z_symm_loss_scalar']
        self.enable_sample = config['enable_sample']
        self.checkpoint_interval = config['checkpoint_interval']
        self.latent_code_num = config['latent_code_num']
        self.t_batch_multiple = config['t_batch_multiple']
        self.r_batch_multiple = config['r_batch_multiple']
        self.t_recon_batch_multiple = config['t_recon_batch_multiple']
        self.r_recon_batch_multiple = config['r_recon_batch_multiple']
        self.t_range = config['t_range']
        self.r_range = config['r_range']
        self.t_n_dims = config['t_n_dims']
        self.r_n_dims = config['r_n_dims']
        self.learning_rate = config['learning_rate']
        self.scheduler_base_num = config['scheduler_base_num']
        self.max_iter_num = config['max_iter_num']
        self.base_len = config['base_len']
        self.train_result_path = config['train_result_path']
        self.train_record_path = config['train_record_path']
        self.eval_record_path = config['eval_record_path']
        self.log_interval = config['log_interval']
        self.eval_interval = config['eval_interval']
        self.sample_prob_param_alpha = config['sample_prob_param_alpha']
        self.sample_prob_param_beta = config['sample_prob_param_beta']
        self.enable_SRS = config['enable_SRS']
        self.enable_SRSD = config['enable_SRSD']
        self.is_save_img = config['is_save_img']
        self.is_prior_model = config['is_prior_model']

    def save_result_imgs(self, img_list, name, seq_len):
        os.makedirs(self.train_result_path, exist_ok=True)
        result = torch.cat([img[0] for img in img_list], dim=0)
        save_image(result, self.train_result_path + str(name) + '.png', seq_len)

    def get_sample_prob(self, step):
        alpha = self.sample_prob_param_alpha
        beta = self.sample_prob_param_beta
        return alpha / (alpha + np.exp((step + beta) / alpha))

    def gen_sample_points(self, base_len, total_len, step, enable_sample):
        if not enable_sample:
            return []
        sample_rate = self.get_sample_prob(step)
        sample_list = []
        for i in range(base_len, total_len):
            r = np.random.rand()
            if r > sample_rate:
                sample_list.append(i)
        return sample_list

    def resume(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(self.model.load_tensor(self.model_path))
            print(f"Model is loaded", flush=True)
        else:
            print("New model is initialized", flush=True)

    def eval(self, iter_num, eval_loss_counter):
        print("=====================start eval=======================", flush=True)
        self.model.eval()
        z_gt_list = []
        label_list = []
        z0_rnn_list = []
        vae_loss_list = []
        rnn_recon_loss_list = []
        data_shape = None
        for batch_ndx, sample in enumerate(self.eval_data_loader):
            data = sample[0].to(DEVICE)
            data = data.to(DEVICE)
            data_shape = data.size()
            z_rp, mu, logvar = self.model.batch_seq_encode_to_z(data)
            sample_points = list(range(z_rp.size(1)))[self.base_len:]
            z0_rnn = self.model.predict_with_symmetry(mu, sample_points, lambda z: z)
            rnn_recon_loss = self.calc_rnn_loss(data[:, 1:, :, :, :], mu, z0_rnn)[0].item()
            vae_loss = self.calc_vae_loss(data, z_rp, mu, logvar)[0].item()
            z_gt_list.append(mu.detach())
            label_list.append(sample[1])
            z0_rnn_list.append(z0_rnn.detach())
            vae_loss_list.append(vae_loss)
            rnn_recon_loss_list.append(rnn_recon_loss)
        self.model.train()
        tensor_z_gt = torch.stack(z_gt_list)
        tensor_Y = torch.stack(label_list)
        tensor_Y = tensor_Y.reshape(tensor_Y.size(0)*tensor_Y.size(1)*tensor_Y.size(2), tensor_Y.size(3))
        tensor_X = tensor_z_gt.reshape(tensor_z_gt.size(0)*tensor_z_gt.size(1)*tensor_z_gt.size(2), tensor_z_gt.size(3))
        xyz_mse = calc_group_mse(tensor_X, tensor_Y)[3]
        tensor_z0_Rnn = torch.stack(z0_rnn_list)
        norm_z_gt, mean_z_gt, std_z_gt = vector_z_score_norm(tensor_z_gt)
        norm_z0_Rnn, mean_z0_Rnn, std_z0_Rnn = vector_z_score_norm(tensor_z0_Rnn, mean_z_gt, std_z_gt)
        rnn_z_loss = nn.MSELoss()(norm_z0_Rnn, norm_z_gt[:, :, 1:, :]).item()
        vae_recon_loss_iter_mean = np.mean(vae_loss_list) / data_shape[1] * (data_shape[1] - 1)
        rnn_recon_loss_iter_mean = np.mean(rnn_recon_loss_list)
        vae_recon_loss_pixel_mean = vae_recon_loss_iter_mean / data_shape[0] / (data_shape[1] - 1) / data_shape[2] / \
                                    data_shape[3] / data_shape[4]
        rnn_recon_loss_pixel_mean = rnn_recon_loss_iter_mean / data_shape[0] / (data_shape[1] - 1) / data_shape[2] / \
                                    data_shape[3] / data_shape[4]
        eval_loss_counter.add_values([vae_recon_loss_iter_mean, rnn_recon_loss_iter_mean,
                                      vae_recon_loss_pixel_mean, rnn_recon_loss_pixel_mean, rnn_z_loss, xyz_mse])
        eval_loss_counter.record_and_clear(self.eval_record_path, iter_num, round_idx=4)
        print("=====================end eval=======================", flush=True)

    def scheduler_func(self, curr_iter):
        return self.scheduler_base_num ** curr_iter

    def train(self):
        self.model.train()
        self.resume()
        train_loss_counter = LossCounter(['loss_ED', 'loss_ERnnD', 'loss_Rnn',
                                          'loss_TRnnTr_rnn', 'loss_RRnnRr_rnn',
                                          'loss_TRnnTr_z1', 'loss_RRnnRr_z1',
                                          'loss_TRnnTrD_x1', 'loss_RRnnRrD_x1',
                                          'KLD'])
        eval_loss_counter = LossCounter(['loss_ED', 'loss_ERnnD', 'loss_ED_mean', 'loss_ERnnD_mean', 'loss_Rnn_norm', 'xyz_mse'])
        iter_num = train_loss_counter.load_iter_num(self.train_record_path)
        curr_iter = iter_num
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: self.scheduler_func(curr_iter))
        is_continue = True
        while is_continue:
            for batch_ndx, sample in enumerate(self.train_data_loader):
                print(curr_iter)
                data = sample[0].to(DEVICE)
                data = data.to(DEVICE)
                is_log = (curr_iter % self.log_interval == 0 and curr_iter != 0)
                recon_list = [data[:, 1:, ...]] if is_log and self.is_save_img else None
                is_eval = curr_iter % self.eval_interval == 0
                optimizer.zero_grad()
                I_sample_points = self.gen_sample_points(self.base_len, data.size(1), curr_iter, self.enable_sample)
                T_sample_points = self.gen_sample_points(self.base_len, data.size(1), curr_iter, self.enable_sample)
                R_sample_points = self.gen_sample_points(self.base_len, data.size(1), curr_iter, self.enable_sample)
                z_rpm, mu, logvar = self.model.batch_seq_encode_to_z(data)
                if self.t_batch_multiple:
                    T, Tr = make_translation_batch(
                        data.size(0) * self.t_batch_multiple,
                        self.t_n_dims, self.latent_code_num,
                        t_range=self.t_range,
                    )
                if self.r_batch_multiple:
                    R, Rr = make_rotation_batch(
                        data.size(0), self.r_batch_multiple,
                        n_dims=self.r_n_dims,
                        angel_range=self.r_range,
                    )
                if self.is_prior_model:
                    z0_rnn = self.model.predict_with_symmetry(z_rpm, I_sample_points, lambda z: z)
                    rnn_loss = self.calc_rnn_loss(data[:, 1:, :, :, :], z_rpm, z0_rnn, recon_list)
                else:
                    z0_rnn = torch.zeros([z_rpm.size(0), z_rpm.size(1)-1, z_rpm.size(2)])
                    rnn_loss = torch.zeros(2)
                vae_loss = self.calc_vae_loss(data, z_rpm, mu, logvar, recon_list)
                if self.enable_SRSD and self.t_batch_multiple:
                    T_z_loss = self.batch_symm_z_loss(
                        z_rpm, z0_rnn, T_sample_points, self.t_batch_multiple,
                        lambda z: symm_trans(z, T), lambda z: symm_trans(z, Tr)
                    )
                    T_recon_loss = self.batch_symm_recon_loss(
                        data[:, 1:, :, :, :], z_rpm,
                        T_sample_points, self.t_recon_batch_multiple,
                        lambda z: symm_trans(z, T), lambda z: symm_trans(z, Tr),
                    )
                else:
                    T_z_loss = torch.zeros(2, device=DEVICE)
                    T_recon_loss = torch.tensor(0, device=DEVICE)
                if self.enable_SRSD and self.r_batch_multiple:
                    R_z_loss = self.batch_symm_z_loss(
                        z_rpm, z0_rnn, R_sample_points, self.r_batch_multiple,
                        lambda z: symm_rota(z, R), lambda z: symm_rota(z, Rr)
                    )
                    R_recon_loss = self.batch_symm_recon_loss(
                        data[:, 1:, :, :, :], z_rpm,
                        R_sample_points, self.r_recon_batch_multiple,
                        lambda z: symm_rota(z, R), lambda z: symm_rota(z, Rr),
                    )
                else:
                    R_z_loss = torch.zeros(2, device=DEVICE)
                    R_recon_loss = torch.tensor(0, device=DEVICE)
                loss = self.loss_func(vae_loss, rnn_loss, T_z_loss, R_z_loss, T_recon_loss, R_recon_loss, train_loss_counter)
                loss.backward()
                optimizer.step()
                scheduler.step()

                if is_log:
                    self.model.save_tensor(
                        self.model.state_dict(),
                        self.model_path,
                    )
                    print(train_loss_counter.make_record(curr_iter), flush=True)
                    train_loss_counter.record_and_clear(self.train_record_path, curr_iter)
                    # self.save_result_imgs(recon_list, f'{i}_{str(I_sample_points)}', z_rpm.size(1) - 1)
                if is_eval:
                    self.eval(curr_iter, eval_loss_counter)
                if curr_iter % self.checkpoint_interval == 0 and curr_iter != 0:
                    self.model.save_tensor(
                        self.model.state_dict(),
                        self.checkpoint_path % curr_iter,
                    )
                curr_iter+=1
                if curr_iter == self.max_iter_num:
                    is_continue = False

    def batch_symm_z_loss(self, z_gt, z0_rnn, sample_points, symm_batch_multiple, symm_func, symm_reverse_func):
        z_gt_repeat = z_gt.repeat(symm_batch_multiple, 1, 1)
        z0_S_rnn = self.model.predict_with_symmetry(z_gt_repeat, sample_points, symm_func)
        z0_rnn_repeat = z0_rnn.repeat(symm_batch_multiple, 1, 1)
        zloss_S_rnn_Sr__rnn, zloss_S_rnn_Sr__z1 = \
            self.calc_symm_loss(z_gt_repeat, z0_rnn_repeat, z0_S_rnn, symm_reverse_func)
        return zloss_S_rnn_Sr__rnn / symm_batch_multiple, zloss_S_rnn_Sr__z1 / symm_batch_multiple

    def calc_rnn_loss(self, x1, z_gt, z0_rnn, recon_list=None):
        recon_next = self.model.batch_seq_decode_from_z(z0_rnn)
        xloss_ERnnD = nn.BCELoss(reduction='sum')(recon_next, x1)
        zloss_Rnn = self.z_rnn_loss_scalar * self.mse_loss(z0_rnn, z_gt[:, 1:, :])
        if recon_list is not None:
            recon_list.append(recon_next.detach())
        return xloss_ERnnD, zloss_Rnn

    def calc_vae_loss(self, data, z_gt, mu, logvar, recon_list=None):
        recon = self.model.batch_seq_decode_from_z(z_gt)
        recon_loss = nn.BCELoss(reduction='sum')(recon, data)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar)) * self.kld_loss_scalar
        if recon_list is not None:
            recon_list.append(recon[:, 1:].detach())
        return recon_loss, KLD

    def calc_symm_loss(self, z_gt, z0_rnn, z0_S_rnn, symm_reverse_func):
        z0_S_rnn_Sr = do_seq_symmetry(z0_S_rnn, symm_reverse_func)
        z1 = z_gt[:, 1:, :]
        zloss_S_rnn_Sr__rnn = self.z_symm_loss_scalar * self.mse_loss(z0_S_rnn_Sr, z0_rnn)
        zloss_S_rnn_Sr__z1 = self.z_symm_loss_scalar * self.mse_loss(z0_S_rnn_Sr, z1)
        return zloss_S_rnn_Sr__rnn, zloss_S_rnn_Sr__z1

    def batch_symm_recon_loss(self, x1, z_gt, sample_points, symm_recon_batch_multiple, symm_func, symm_reverse_func):
        z_gt_repeat = z_gt.repeat(symm_recon_batch_multiple, 1, 1)
        z0_S_rnn = self.model.predict_with_symmetry(z_gt_repeat, sample_points, symm_func)
        z0_S_rnn_Sr = do_seq_symmetry(z0_S_rnn, symm_reverse_func)
        z0_S_rnn_Sr_D = self.model.batch_seq_decode_from_z(z0_S_rnn_Sr)
        x1_rp = x1.repeat(symm_recon_batch_multiple, 1, 1, 1, 1)
        symm_recon_loss = nn.BCELoss(reduction='sum')(z0_S_rnn_Sr_D, x1_rp) / symm_recon_batch_multiple
        return symm_recon_loss

    def loss_func(self, vae_loss, rnn_loss, T_loss, R_loss, T_recon_loss, R_recon_loss, loss_counter):
        xloss_ED, KLD = vae_loss
        xloss_ERnnD, zloss_Rnn = rnn_loss
        zloss_T_rnn_Tr__rnn, zloss_T_rnn_Tr__z1 = T_loss
        zloss_R_rnn_Rr__rnn, zloss_R_rnn_Rr__z1 = R_loss

        loss = torch.zeros(1)[0].to(DEVICE)
        loss += xloss_ED + KLD + xloss_ERnnD
        loss += zloss_Rnn
        loss += zloss_T_rnn_Tr__z1 + zloss_R_rnn_Rr__z1
        loss += T_recon_loss + R_recon_loss
        if self.enable_SRS:
            loss += zloss_T_rnn_Tr__rnn + zloss_R_rnn_Rr__rnn

        loss_counter.add_values([xloss_ED.item(), xloss_ERnnD.item(), zloss_Rnn.item(),
                                 zloss_T_rnn_Tr__rnn.item(), zloss_R_rnn_Rr__rnn.item(),
                                 zloss_T_rnn_Tr__z1.item(), zloss_R_rnn_Rr__z1.item(),
                                 T_recon_loss.item(), R_recon_loss.item(),
                                 KLD.item()
                                 ])
        return loss

    def eval_a_checkpoint(self, checkpoint_num, checkpoint_path):
        eval_loss_counter = LossCounter(['loss_ED', 'loss_ERnnD', 'loss_ED_mean', 'loss_ERnnD_mean', 'loss_Rnn_norm', 'xyz_mse'])
        self.model.load_state_dict(self.model.load_tensor(checkpoint_path))
        self.eval(checkpoint_num, eval_loss_counter)
