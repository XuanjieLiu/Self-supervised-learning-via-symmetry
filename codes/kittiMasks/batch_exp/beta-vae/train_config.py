import math
import os

data_root = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../../data/kitti')
CONFIG = {
    'train_data_path': f'{data_root}/kitti_train.pickle',
    'eval_data_path': f'{data_root}/kitti_eval.pickle',
    'rnn_latent_code_num': 3,
    'latent_code_num': 3,
    't_batch_multiple': 0,
    'r_batch_multiple': 0,
    't_recon_batch_multiple': 0,
    'r_recon_batch_multiple': 0,
    't_range': (-1, 1),
    'r_range': (-math.pi, math.pi),
    't_n_dims': 3,
    'r_n_dims': 2, 
    'rnn_num_layers': 1,
    'rnn_hidden_size': 256,
    'model_path': 'Conv2dGruConv2d_symmetry.pt',
    'checkpoint_path': 'checkpoint_%d.pt',
    'kld_loss_scalar': 0.01,
    'enable_SRS': True,
    'enable_SRSD': True,
    'z_rnn_loss_scalar': 2,
    'z_symm_loss_scalar': 2,
    'enable_sample': True,
    'checkpoint_interval': 2000,
    'learning_rate': 1e-3,
    'scheduler_base_num': 0.99999,
    'max_iter_num': 50001,
    'base_len': 4,
    'train_result_path': 'TrainingResults/',
    'train_record_path': "Train_record.txt",
    'eval_record_path': "Eval_record.txt",
    'log_interval': 250,
    'eval_interval': 500,
    'sample_prob_param_alpha': 1200,
    'sample_prob_param_beta': 6000,
    'rnn_type': 'RNN',
    'is_save_img': True,
    'batch_size': 32,
    'is_prior_model': False,
}
