import os
from os import path
import sys
from importlib import reload

temp_dir = path.abspath(path.join(
    path.dirname(__file__), '../model', 
))
sys.path.append(temp_dir)
from tableEvalDecoder import Group, UI
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
    for a in 'tr':
        print(a, ':', config[a + '_batch_multiple'], ' times with n_dims =', config[a + '_n_dims'])
    # input('Please verify. Enter...')
    print()
    g = Group(group_name, group_name, config)
    groups.append(g)


ui = UI(groups, 6, '.', 'latest.pt', 'Wrong symm assumptions')
ui.win.mainloop()
