import math
import os
data_root = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../../Ball3DImg')

CONFIG = {
    'train_data_path': f'{data_root}/32_32_0.2_20_3_init_points_subset_512/',
    'latent_code_num': 3,
    'batch_size': 32,
    't_batch_multiple': 0,
    'r_batch_multiple': 0,
    't_range': (-1, 1),
    'r_range': (-math.pi, math.pi),
    'rnn_num_layers': 1,
    'rnn_hidden_size': 256,
    'eval_data_path': f'{data_root}/32_32_0.2_20_3_init_points_EvalSet/',
    'model_path': 'current_model.pt',
    'kld_loss_scalar': 0,
    'enable_SRS': True,
    'enable_SRSD': True,
    'z_rnn_loss_scalar': 2,
    'z_symm_loss_scalar': 2,
    'enable_sample': True,
    'checkpoint_interval': 5000,
    'learning_rate': 1e-3,
    'scheduler_base_num': 0.99999,
    'max_iter_num': 80001,
    'base_len': 5,
    'train_result_path': 'TrainingResults/',
    'train_record_path': "Train_record.txt",
    'eval_record_path': "Eval_record.txt",
    'log_interval': 100,
    'eval_interval': 1000,
    'sample_prob_param_alpha': 2200,
    'sample_prob_param_beta': 6000,
    'rnn_type': 'RNN',
    'is_save_img': False,
}
