import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

reg_loss_ignore = ['module.mlp_o.conv0.conv.weight', 'module.mlp_o.conv1.conv.weight', 'module.mlp_o.conv2.conv.weight']

def plot_grad_flow(named_parameters, save_path, verbose=True):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())

    if verbose:
        for agi, ag in enumerate(ave_grads):
            if ag == 0. and layers[agi] not in reg_loss_ignore:
                print(f'Warning: layer {layers[agi]} has 0 gradient')

    # axarr = f.add_subplot(1,1,1) # here is where you add the subplot to f
    plt.figure(figsize=(10,10))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    plt.savefig(save_path)

def convert_official_weights(model, odict):
    """ Loads the published weights (net_latest) into our model"""

    cnt = 0
    okeys = list(odict.keys())
    nkeys = list(model.state_dict().keys())
    ndict = model.state_dict()
    max_disp = model.max_disp

    # load disp backbone weights
    for i in range(20):
        oname = okeys[cnt]
        nname = nkeys[20+i]
        ndict[nname] = odict[oname]
        cnt += 1

    # load rgb backbone weights
    for i in range(20):
        oname = okeys[cnt]
        nname = nkeys[i]
        ndict[nname] = odict[oname]
        cnt += 1

    # load decoder weights
    for i in range(28):
        oname = okeys[cnt]
        nname = nkeys[54+i]
        ndict[nname] = odict[oname]
        cnt += 1

    # load MLP_c and MLP_o weights
    for i in range(14):
        oname = okeys[cnt]
        nname = nkeys[40+i]
        if i == 6 or i == 7: # conv3 layers
            ndict[nname] = odict[oname][:max_disp,...]
        else:
            ndict[nname] = odict[oname]
        cnt += 1

    return ndict

def convert_official_weights_teacher(model, odict):
    """ Loads the published weights (net_latest) into our model"""

    cnt = 0
    okeys = list(odict.keys())
    nkeys = list(model.state_dict().keys())
    ndict = model.state_dict()
    max_disp = model.max_disp

    # load disp backbone weights
    for i in range(20):
        oname = okeys[cnt]
        nname = nkeys[20+i]
        ndict[nname] = odict[oname]
        cnt += 1

    # load rgb backbone weights
    for i in range(20):
        oname = okeys[cnt]
        nname = nkeys[i]
        ndict[nname] = odict[oname]
        cnt += 1

    # for i in range(20,len(nkeys)):
    #     print(i, nkeys[i])

    # load decoder weights
    for i in range(28):
        oname = okeys[cnt]
        nname = nkeys[48+i]
        ndict[nname] = odict[oname]
        cnt += 1

    return ndict
