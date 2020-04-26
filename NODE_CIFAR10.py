# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 03:06:54 2020

@author: jv204
"""
#%%
from __future__ import print_function
import argparse
import os
import random
import math
from IPython.display import clear_output

from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable

import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import matplotlib.cm
from IPython.display import HTML
#print(torch.get_num_threads())
#torch.set_num_threads(8)
#print(torch.get_num_threads())
#%%
#  Set random seed for reproducibility
manualSeed = 375
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
#%%
# @dingdonged
use_cuda = torch.cuda.is_available()
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
# @dingdonged
def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() \
                   for i, el in zip(preds, output)]

#  @dingdonged
def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() \
                           else "red"))
    return fig
#  @dingdonged
def ode_solve(z0, t0, t1, f):
    """
    Simplest Euler ODE initial value solver
    """
    h_max = 0.05
    n_steps = math.ceil((abs(t1 - t0)/h_max).max().item())

    h = (t1 - t0)/n_steps
    t = t0
    z = z0

    for i_step in range(n_steps):
        z = z + h * f(z, t)
        t = t + h
    return z
#  @dingdonged
class ODEF(nn.Module):
    def forward_with_grad(self, z, t, grad_outputs):
        """Compute f and a df/dz, a df/dp, a df/dt"""
        batch_size = z.shape[0]

        out = self.forward(z, t)

        a = grad_outputs
        adfdz, adfdt, *adfdp = torch.autograd.grad(
            (out,), (z, t) + tuple(self.parameters()), grad_outputs=(a),
            allow_unused=True, retain_graph=True
        )
        # grad method automatically sums gradients for batch items, 
        # we have to expand them back 
        if adfdp is not None:
            adfdp = torch.cat([p_grad.flatten() \
                               for p_grad in adfdp]).unsqueeze(0)
            adfdp = adfdp.expand(batch_size, -1) / batch_size
        if adfdt is not None:
            adfdt = adfdt.expand(batch_size, 1) / batch_size
        return out, adfdz, adfdt, adfdp

    def flatten_parameters(self):
        p_shapes = []
        flat_parameters = []
        for p in self.parameters():
            p_shapes.append(p.size())
            flat_parameters.append(p.flatten())
        return torch.cat(flat_parameters)
#  @dingdonged
class ODEAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z0, t, flat_parameters, func):
        assert isinstance(func, ODEF)
        bs, *z_shape = z0.size()
        time_len = t.size(0)

        with torch.no_grad():
            z = torch.zeros(time_len, bs, *z_shape).to(z0)
            z[0] = z0
            for i_t in range(time_len - 1):
                z0 = ode_solve(z0, t[i_t], t[i_t+1], func)
                z[i_t+1] = z0

        ctx.func = func
        ctx.save_for_backward(t, z.clone(), flat_parameters)
        return z

    @staticmethod
    def backward(ctx, dLdz):
        """
        dLdz shape: time_len, batch_size, *z_shape
        """
        func = ctx.func
        t, z, flat_parameters = ctx.saved_tensors
        time_len, bs, *z_shape = z.size()
        n_dim = np.prod(z_shape)
        n_params = flat_parameters.size(0)

        # Dynamics of augmented system to be calculated backwards in time
        def augmented_dynamics(aug_z_i, t_i):
            """
            tensors here are temporal slices
            t_i - is tensor with size: bs, 1
            aug_z_i - is tensor with size: bs, n_dim*2 + n_params + 1
            """
            z_i, a = aug_z_i[:, :n_dim], aug_z_i[:, n_dim:2*n_dim]  
            # ignore parameters and time

            # Unflatten z and a
            z_i = z_i.view(bs, *z_shape)
            a = a.view(bs, *z_shape)
            with torch.set_grad_enabled(True):
                t_i = t_i.detach().requires_grad_(True)
                z_i = z_i.detach().requires_grad_(True)
                func_eval, adfdz, adfdt, adfdp = \
                    func.forward_with_grad(z_i, t_i, grad_outputs=a)  
                    # bs, *z_shape
                adfdz = adfdz.to(z_i) if adfdz is not None \
                    else torch.zeros(bs, *z_shape).to(z_i)
                adfdp = adfdp.to(z_i) if adfdp is not None \
                    else torch.zeros(bs, n_params).to(z_i)
                adfdt = adfdt.to(z_i) if adfdt is not None \
                    else torch.zeros(bs, 1).to(z_i)

            # Flatten f and adfdz
            func_eval = func_eval.view(bs, n_dim)
            adfdz = adfdz.view(bs, n_dim) 
            return torch.cat((func_eval, -adfdz, -adfdp, -adfdt), dim=1)

        dLdz = dLdz.view(time_len, bs, n_dim)  # flatten dLdz for convenience
        with torch.no_grad():
            ## Create placeholders for output gradients
            # Prev computed backwards adjoints
            # to be adjusted by direct gradients
            adj_z = torch.zeros(bs, n_dim).to(dLdz)
            adj_p = torch.zeros(bs, n_params).to(dLdz)
            # In contrast to z and p we need to return gradients for all times
            adj_t = torch.zeros(time_len, bs, 1).to(dLdz)

            for i_t in range(time_len-1, 0, -1):
                z_i = z[i_t]
                t_i = t[i_t]
                f_i = func(z_i, t_i).view(bs, n_dim)

                # Compute direct gradients
                dLdz_i = dLdz[i_t]
                dLdt_i = torch.bmm(torch.transpose(dLdz_i.unsqueeze(-1), 1, 2),
                                   f_i.unsqueeze(-1))[:, 0]

                # Adjusting adjoints with direct gradients
                adj_z += dLdz_i
                adj_t[i_t] = adj_t[i_t] - dLdt_i

                # Pack augmented variable
                aug_z = torch.cat((z_i.view(bs, n_dim), adj_z, 
                                   torch.zeros(bs, n_params).to(z),
                                   adj_t[i_t]), dim=-1)

                # Solve augmented system backwards
                aug_ans = ode_solve(aug_z, t_i, t[i_t-1], augmented_dynamics)

                # Unpack solved backwards augmented system
                adj_z[:] = aug_ans[:, n_dim:2*n_dim]
                adj_p[:] += aug_ans[:, 2*n_dim:2*n_dim + n_params]
                adj_t[i_t-1] = aug_ans[:, 2*n_dim + n_params:]

                del aug_z, aug_ans

            ## Adjust 0 time adjoint with direct gradients
            # Compute direct gradients 
            dLdz_0 = dLdz[0]
            dLdt_0 = torch.bmm(torch.transpose(dLdz_0.unsqueeze(-1), 1, 2), 
                               f_i.unsqueeze(-1))[:, 0]

            # Adjust adjoints
            adj_z += dLdz_0
            adj_t[0] = adj_t[0] - dLdt_0
        return adj_z.view(bs, *z_shape), adj_t, adj_p, None
#  @dingdonged
class NeuralODE(nn.Module):
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        assert isinstance(func, ODEF)
        self.func = func

    def forward(self, z0, t=torch.Tensor([0., 1.]), 
                return_whole_sequence=False):
        t = t.to(z0)
        z = ODEAdjoint.apply(z0, t, self.func.flatten_parameters(), self.func)
        if return_whole_sequence:
            return z
        else:
            return z[-1]
#  @dingdonged
def norm(dim):
    return nn.BatchNorm2d(dim)
#  @dingdonged
def conv3x3(in_feats, out_feats, stride=1):
    return nn.Conv2d(in_feats, out_feats, kernel_size=3, stride=stride, 
                     padding=1, bias=False)
#  @dingdonged
def add_time(in_tensor, t):
    bs, c, w, h = in_tensor.shape
    return torch.cat((in_tensor, t.expand(bs, 1, w, h)), dim=1)
#  @dingdonged
class ConvODEF(ODEF):
    def __init__(self, dim):
        super(ConvODEF, self).__init__()
        self.conv1 = conv3x3(dim + 1, dim)
        self.norm1 = norm(dim)
        self.conv2 = conv3x3(dim + 1, dim)
        self.norm2 = norm(dim)

    def forward(self, x, t):
        xt = add_time(x, t)
        h = self.norm1(torch.relu(self.conv1(xt)))
        ht = add_time(h, t)
        dxdt = self.norm2(torch.relu(self.conv2(ht)))
        return dxdt
#  @dingdonged
class ContinuousNeuralCIFAR10Classifier(nn.Module):
    def __init__(self, ode):
        super(ContinuousNeuralCIFAR10Classifier, self).__init__()
        self.downsampling = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
        )
        self.feature = ode
        self.norm = norm(64)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.downsampling(x)
        x = self.feature(x)
        x = self.norm(x)
        x = self.avg_pool(x)
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        x = x.view(-1, shape)
        out = self.fc(x)
        return out

#%%
func = ConvODEF(64)
ode = NeuralODE(func)
CNNODE = ContinuousNeuralCIFAR10Classifier(ode)
if use_cuda:
    CNNODE = CNNODE.cuda()
#%%
batch_size = 96
train_loader_real = torch.utils.data.DataLoader(
    dset.CIFAR10("data/cifar10", train=True, download=True,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                          (0.2023, 0.1994, 0.2010))])), 
    batch_size=batch_size, shuffle=True)
test_loader_real = torch.utils.data.DataLoader(
    dset.CIFAR10("data/cifar10", train=False, download=True,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465),
                                          (0.2023, 0.1994, 0.2010))])),
    batch_size=128, shuffle=True)
#%%
optimizer = torch.optim.Adam(CNNODE.parameters())
#%%
n_epochs = 20
#  @jv204 or @JWolff98
retrain_model = input("Do you want to train a new model [True/False]: ")
if retrain_model:
    train_losses = []
    train_iters = 0
    train_accuracy = []
    validation_losses = []
    validation_iters = 0
    validation_accuracy = []
    for epoch in range(1, n_epochs + 1):
        #  Training
        num_items_training = 0
        train_epoch_losses = []
        train_epoch_accuracy = 0.0
        CNNODE.train()
        criterion = nn.CrossEntropyLoss()
        print(f"Training Epoch {epoch}...")
        for batch_idx, (data, target) in \
            tqdm(enumerate(train_loader_real),total=len(train_loader_real)):
            if use_cuda:
                data = data.cuda()
                target = target.cuda()
            
            optimizer.zero_grad()
            CNNODE.zero_grad()
            output = CNNODE(data)
            err = criterion(output, target)
            err.backward()
            optimizer.step()
            num_items_training += data.shape[0]
            train_epoch_accuracy += \
                torch.sum(torch.argmax(output, dim=1) == target).item()
            train_accuracy.append(train_epoch_accuracy / num_items_training)
            if batch_idx % 20 == 0:
                 print('[%d/%d][%d/%d]\tLoss: %.4f'
                  % (epoch, n_epochs, batch_idx, len(train_loader_real),
                     err.item()))
            train_losses.append(err.item())
            train_epoch_losses.append(err.item())
            train_iters += 1
        print("Train loss: {:.5f}%".format(np.mean(train_epoch_losses)))
        print()
        
        # Validation
        validation_epoch_accuracy = 0.0
        num_items_validation = 0
        validation_epoch_losses = []
        CNNODE.eval()
        criterion = nn.CrossEntropyLoss()
        print(f"Testing...")
        with torch.no_grad():
            for batch_idx, (data, target) in tqdm(enumerate(test_loader_real),
                                                  total=len(test_loader_real)):
                if use_cuda:
                    data = data.cuda()
                    target = target.cuda()
                output = CNNODE(data)
                loss = criterion(output, target)
                validation_losses.append(loss.item())
                validation_epoch_losses.append(loss.item())
                validation_epoch_accuracy += \
                    torch.sum(torch.argmax(output,dim=1) == target).item()
                num_items_validation += data.shape[0]
                validation_accuracy.append((validation_epoch_accuracy)/\
                                           num_items_validation)
                validation_iters += 1
            print(
            "Validation loss: {:.5f}%".format(np.mean(validation_epoch_losses))
                  )
            print(
            "Validation Accuracy: {:.3f}%".format((validation_epoch_accuracy\
                                                   * 100)/num_items_validation)
                  )
    retrain_model = False

#%%
torch.save(CNNODE.state_dict(), "CNNODE%.pth" %(n_epochs))

#%%                
plt.figure(figsize=(10,5))
plt.title("CNNODE Loss During Training")
plt.plot(train_losses,label="Training Loss", color ='green')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

#%%
plt.figure(figsize=(10,5))
plt.title("CNNODE Accuracy During Training")
plt.plot(train_accuracy,label="Training Accuraccy")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

#%%
plt.figure(figsize=(10,5))
plt.title("CNNODE Loss During Validation")
plt.plot(validation_losses,label="Validation Loss", color='yellow')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

#%%
plt.figure(figsize=(10,5))
plt.title("CNNODE Accuracy During Validation")
plt.plot(validation_accuracy,label="Validation Accuracy", color='red')
plt.xlabel("iterations")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



            
        
                     
        
                
    
