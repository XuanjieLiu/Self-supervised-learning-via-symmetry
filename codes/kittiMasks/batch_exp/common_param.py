
BATCH_EXP = [
    {
        'dir': 'beta_vae',
        'name': 'beta-VAE'
    }, {
        'dir': 'noSymm_vae',
        'name': 'Ours w/o symmetry'
    }, {
        'dir': 't3_k1_ae',
        'name': 'T3 AE',
    }, {
        'dir': 't3_k1_vae',
        'name': 'T3 VAE',
    }, {
        'dir': 't3_r2_k1_ae',
        'name': 'T3 R2 AE',
    }, {
        'dir': 't3_r2_k1_vae',
        'name': 'T3 R2 VAE',
    },
]

SUB_EXP_LIST = [str(i) for i in range(1, 31)]

CHECK_POINT_NUM_LIST = [i*2000 for i in range(1, 11)]

EVAL_RECORD_NAME = 'Eval_record.txt'

AMENDED_EVAL_NAME = 'amended_eval.txt'
