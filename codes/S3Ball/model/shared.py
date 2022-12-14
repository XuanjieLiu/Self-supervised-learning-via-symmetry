import torch

HAS_CUDA = torch.cuda.is_available()
CUDA = torch.device("cuda:0")
CPU  = torch.device("cpu")
if HAS_CUDA:
    DEVICE = CUDA
    print('We have CUDA.')
    gpu_name = torch.cuda.get_device_name(DEVICE)
    print(f'{gpu_name = }', flush=True)
else:
    DEVICE = CPU
    print("We DON'T have CUDA.")

def loadModel(ModelClass, filename: str, config):
    model = ModelClass(config).to(DEVICE)
    model.load_state_dict(torch.load(
        filename, map_location=DEVICE,
    ))
    model.eval()
    return model
