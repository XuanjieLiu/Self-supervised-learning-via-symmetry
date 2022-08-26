import sys
import os
# sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../../../'))
from codes.common_utils import create_results_path_if_not_exist

print(os.path.abspath(__file__))
print('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../../Ball3DImg'))

file_dir = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/')

data_root = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../../Ball3DImg')

print(file_dir)
os.chdir(file_dir)
create_results_path_if_not_exist('2')