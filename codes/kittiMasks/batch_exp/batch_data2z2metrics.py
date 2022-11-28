import sys
import os
from codes.kittiMasks.batch_exp.common_param import BATCH_EXP, SUB_EXP_LIST, CHECK_POINT_NUM_LIST, EVAL_RECORD_NAME, AMENDED_EVAL_NAME
import math
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../../../'))
from codes.S3Ball.evalLinearity_A_without_four_zeros_regress import Dataset
import torch
from sklearn.linear_model import LinearRegression
from codes.metrics.correlation import compute_mcc


def getErr(X, Y) -> torch.Tensor:
    regression = LinearRegression().fit(X, Y)
    return Y - regression.predict(X)


def log_Y(Y):
    Y1 = Y[..., 0:2]
    Y2 = Y[..., 2:]
    logy2 = torch.log(Y2)
    return torch.cat((Y1, logy2), dim=-1)


def tan_sqrt_Y(Y):
    Y1 = Y[..., 0:2]
    Y2 = Y[..., 2:]
    tan_sqrt_Y2 = 1 / torch.tan(torch.sqrt(Y2)/(64*6/math.pi))
    return torch.cat((Y1, tan_sqrt_Y2), dim=-1)


def normal_projection(X, Y):
    std = Y.std(dim=0)
    Y_norm = Y / std
    err = getErr(X, Y_norm)
    x_mse = err[:, 0].square().mean().item()
    y_mse = err[:, 1].square().mean().item()
    z_mse = err[:, 2].square().mean().item()

    xyz_mse = (x_mse + z_mse + y_mse) / 3
    # return x_mse, y_mse, z_mse, xyz_mse
    return round(x_mse, 4), round(y_mse, 4), round(z_mse, 4), round(xyz_mse, 4)


BATCH_ROOT = os.path.dirname(os.path.abspath(__file__))

def append_record(cp_num_list, amended_list, origin_file, output_file):
    with open(origin_file, 'r') as f:
        records = f.readlines()
    records = [r.rstrip() for r in records]
    filtered_record = list(filter(lambda l: int(l.split('-')[0]) in cp_num_list, records))
    assert (len(amended_list) == len(filtered_record)), "len(amended_list) is not equal to len(filtered_record)"
    new_record = []
    for i in range(len(filtered_record)):
        new_record.append(f'{filtered_record[i]},{amended_list[i]}\n')
    with open(output_file, 'w') as f2:
        f2.writelines(new_record)
    print(filtered_record)

if __name__ == '__main__':
    for exp in BATCH_EXP:
        exp_path = os.path.join(BATCH_ROOT, exp['dir'])
        os.chdir(exp_path)
        for sub_exp in SUB_EXP_LIST:
            sub_exp_path = os.path.join(exp_path, sub_exp)
            os.chdir(sub_exp_path)
            amended_list = []
            for num in CHECK_POINT_NUM_LIST:
                output_name = f'data2z_{num}.txt'
                dataset = Dataset(output_name)
                X = dataset.X
                Y = dataset.Y
                logArea_Y = log_Y(dataset.Y)
                tanSqrt_Y = tan_sqrt_Y(dataset.Y)
                # normal_proj = normal_projection(X, Y)[3]
                logArea_proj = normal_projection(X, logArea_Y)[3]
                tanSqrt_proj = normal_projection(X, tanSqrt_Y)[3]
                amended_list.append(f'logArea_mse:{logArea_proj},tanSqrt_mse:{tanSqrt_proj}')
            append_record(CHECK_POINT_NUM_LIST, amended_list, EVAL_RECORD_NAME, AMENDED_EVAL_NAME)
