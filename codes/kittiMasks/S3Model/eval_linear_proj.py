import torch
from sklearn.linear_model import LinearRegression


def getErr(X, Y) -> torch.Tensor:
    regression = LinearRegression().fit(X, Y)
    return Y - regression.predict(X)


def calc_group_mse(X, Y):
    std = Y.std(dim=0)
    Y_norm = Y / std
    err = getErr(X.cpu().detach(), Y_norm.cpu().detach())
    x_mse = err[:, 0].square().mean().item()
    y_mse = err[:, 1].square().mean().item()
    z_mse = err[:, 2].square().mean().item()
    xyz_mse = (x_mse + z_mse + y_mse) / 3
    return x_mse, y_mse, z_mse, xyz_mse

# def save_z_and_gt(z, gt, save_path):
#     z_expend = z.reshape(z.size(0)*z.size(1), z.size(2)).detach()
#     gt_expend = gt.reshape(gt.size(0)*gt.size(1), gt.size(2))
#     assert (len(z_expend)==len(gt_expend)), f"the length of z ({len(z_expend)}) is not equal to gt {len(gt_expend)}"
#     split_str = ','
#     with open(save_path, 'w') as f:
