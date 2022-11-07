import os
from os import path
import sys
from importlib import reload

temp_dir = path.abspath(path.join(
    path.dirname(__file__), '../..', 
))
sys.path.append(temp_dir)
from S3Ball.model.evalEncoder import Group, evalEncoder
assert sys.path.pop(-1) == temp_dir

group_names = ['T2', 'R2', 'T1R2', 'T3R2', 'T2R3']

groups = []

for group_name in group_names:
    temp_dir = path.abspath(group_name)
    sys.path.append(temp_dir)
    import train_config
    train_config = reload(train_config)
    assert sys.path.pop(-1) == temp_dir

    config = train_config.CONFIG
    print('T R =', config['t_batch_multiple'], config['r_batch_multiple'])
    input('Please verify. Enter...')
    g = Group(group_name, group_name, config)
    groups.append(g)

evalEncoder(
    groups, 6, 150000, 
    '../Ball3DImg/32_32_0.2_20_3_init_points_EvalSet/', 
    '.', 
)
