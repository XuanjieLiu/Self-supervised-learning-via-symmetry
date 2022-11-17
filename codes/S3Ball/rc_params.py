from matplotlib import pyplot as plt

def init(font_size=14):
    plt.rcParams.update({
        'text.usetex': True, 
        'font.family': 'serif', 
        'font.serif': ['Computer Modern'], 
        'font.size': font_size, 
    })
