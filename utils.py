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

def save_modelsv2(G, D, G_path, D_path):
    torch.save(G.state_dict(), G_path)
    torch.save(D.state_dict(), D_path)


def load_model(G, folder):
    ckpt = torch.load(os.path.join(folder,'G.pth'), map_location=device, weights_only=True)
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G

def load_modelv2(G, G_path):
    ckpt = torch.load(G_path, map_location=device, weights_only=True)
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G


def WD_train(x: torch.Tensor, G: nn.Module, D: nn.Module, D_optimizer: torch.optim.Optimizer, weight_clip: float):
    #=======================Train the discriminator=======================#

    # train discriminator on real
    x_real = x.to(device)
    D_output_real = D(x_real).reshape(-1)

    # train discriminator on fake
    z = torch.randn(x.shape[0], 100, device=device)
    x_fake = G(z).detach()
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

def gradient_penalty(D: nn.Module, real, fake):
    real = real.detach()
    fake = fake.detach()
    Batch_size, h_w = real.shape
    epsilon = torch.rand((Batch_size,1)).repeat(1,h_w).to(device)
    interpolated_images = (real * epsilon + fake * (1 - epsilon)).requires_grad_(True)

    #Calculate discriminator scores
    mixed_scores = D(interpolated_images)

    # Calculate gradients of mixed_scores with respect to interpolated_images
    gradient = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated_images,
        grad_outputs=torch.ones_like(mixed_scores),  # Backpropagate a vector of ones
        create_graph=True,  # Retain computation graph for higher-order derivatives
        retain_graph=True,
        only_inputs=True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gd_penalty = torch.mean((gradient_norm -1)**2)
    return gd_penalty


def WGPD_train(x: torch.Tensor, G: nn.Module, D: nn.Module, D_optimizer: torch.optim.Optimizer, lambda_gp: float):
    #=======================Train the discriminator=======================#

    # train discriminator on real
    x_real = x.to(device)
    D_output_real = D(x_real).reshape(-1)

    # train discriminator on fake
    z = torch.randn(x.shape[0], 100, device=device)
    x_fake = G(z).detach()
    D_output_fake =  D(x_fake).reshape(-1)

    # gradient backprop & optimize ONLY D's parameters
    gp = gradient_penalty(D, x_real, x_fake)
    D_loss = -(torch.mean(D_output_real) - torch.mean(D_output_fake)) + lambda_gp * gp
    D.zero_grad()
    D_loss.backward()
    D_optimizer.step()

    return  D_loss.data.item()


#First implementation, not sure that it works
def WGPD_latent_reweighting_train(x: torch.Tensor, G: nn.Module, D: nn.Module, D_optimizer: torch.optim.Optimizer, W: nn.Module, lambda_gp: float, lambda1=0.1, lambda2=0.1):
    #======================= Train the Discriminator =======================#

    # Train on real data
    x_real = x.to(device)
    D_output_real = D(x_real).reshape(-1)

    # Generate fake samples with weighted importance
    z = torch.randn(x.shape[0], 100, device=device)
    x_fake = G(z).detach()
    w_z = W(z).reshape(-1)  # Importance weights for each latent vector
    D_output_fake = D(x_fake).reshape(-1)
    
    # Calculate EMD with latent reweighting
    Delta = torch.min(D_output_fake).detach()
    EMD = torch.mean(w_z * (D_output_fake - Delta)) - torch.mean(D_output_real)

    # Apply gradient penalty (GP)
    gp = gradient_penalty(D, x_real, x_fake)
    
    # Discriminator loss with gradient penalty
    D_loss = -EMD + lambda_gp * gp

    D.zero_grad()
    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()