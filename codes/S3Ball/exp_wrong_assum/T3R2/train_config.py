import math

CONFIG = {
    'train_data_path': '../../Ball3DImg/32_32_0.2_20_3_init_points_subset_2048/',
    'latent_code_num': 3,
    't_batch_multiple': 4,
    'r_batch_multiple': 4,
    't_recon_batch_multiple': 4,
    'r_recon_batch_multiple': 4,
    't_range': (-1, 1),
    'r_range': (-math.pi, math.pi),
    't_n_dims': 3, 
    'r_n_dims': 2, 
    'rnn_num_layers': 1,
    'rnn_hidden_size': 256,
    'eval_data_path': '../../Ball3DImg/32_32_0.2_20_3_init_points_EvalSet/',
    'model_path': 'Conv2dGruConv2d_symmetry.pt',
    'kld_loss_scalar': 0.01,
    'enable_SRS': True,
    'enable_SRSD': True,
    'z_rnn_loss_scalar': 2,
    'z_symm_loss_scalar': 2,
    'enable_sample': True,
    'checkpoint_interval': 10000,
    'learning_rate': 1e-3,
    'scheduler_base_num': 0.99999,
    'max_iter_num': 150001,
    'base_len': 5,
    'train_result_path': 'TrainingResults/',
    'train_record_path': "Train_record.txt",
    'eval_record_path': "Eval_record.txt",
    'log_interval': 200,
    'eval_interval': 1000,
    'sample_prob_param_alpha': 2200,
    'sample_prob_param_beta': 8000,
    'rnn_type': 'RNN',
    'is_save_img': True,
}
