import os

import torch
from device import device
from torch import nn


def D_train(x: torch.Tensor, G: nn.Module, D: nn.Module, D_optimizer: torch.optim.Optimizer, criterion: nn.Module):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real = x.to(device)
    y_real = torch.ones(x.shape[0], 1, device=device)

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on fake
    z = torch.randn(x.shape[0], 100, device=device)
    x_fake = G(z)
    y_fake = torch.zeros(x.shape[0], 1, device=device)

    D_output =  D(x_fake)

    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return  D_loss.data.item()


def G_train(x: torch.Tensor, G: nn.Module, D: nn.Module, G_optimizer: torch.optim.Optimizer, criterion: nn.Module):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100, device=device)
    y = torch.ones(x.shape[0], 1, device=device)

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()



def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def load_model(G, folder):
    ckpt = torch.load(os.path.join(folder,'G.pth'), map_location=device, weights_only=True)
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G


def WD_train(x: torch.Tensor, G: nn.Module, D: nn.Module, D_optimizer: torch.optim.Optimizer, weight_clip: float):
    #=======================Train the discriminator=======================#

    # train discriminator on real
    x_real = x.to(device)
    D_output_real = D(x_real).reshape(-1)

    # train discriminator on fake
    z = torch.randn(x.shape[0], 100, device=device)
    x_fake = G(z)
    D_output_fake =  D(x_fake).reshape(-1)

    # gradient backprop & optimize ONLY D's parameters
    D_loss = -(torch.mean(D_output_real) - torch.mean(D_output_fake))
    D.zero_grad()
    D_loss.backward()
    D_optimizer.step()

    for p in D.parameters():
        p.data.clamp_(-weight_clip, weight_clip)

    return  D_loss.data.item()

def WG_train(x: torch.Tensor, G: nn.Module, D: nn.Module, G_optimizer: torch.optim.Optimizer):
    #=======================Train the generator=======================#

    z = torch.randn(x.shape[0], 100, device=device)

    G_output = G(z)
    D_output = D(G_output).reshape(-1)
    G_loss = - torch.mean(D_output)
    G.zero_grad()

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()