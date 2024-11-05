import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from device import device
from model import Discriminator, Generator, Critic
from torchvision import datasets, transforms
from tqdm import trange
from utils import WD_train, WG_train, save_models, save_modelsv2
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=5e-5,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Size of mini-batches for SGD")
    parser.add_argument("--critic_iterations", type=int, default=5,
                        help="Parameter n_critic in the paper.") #Number of time we train the discriminator
    #more than the generator (if = 5, 5 updates of gradient for discriminator and 1 for generator)
    parser.add_argument("--weight_clip", type=float, default=0.01,
                        help="Clip the value of the weights of the discrminator between -c and c.")


    args = parser.parse_args()


    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Data Pipeline
    print('Dataset loading...')
    # MNIST Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
        # transforms.Lambda(lambda x: x.to(device)),
    ])

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size, shuffle=False)
    print('Dataset Loaded.')


    print('Model Loading...')
    mnist_dim = 784

    # Slower when there is zero or only one GPU
    # G = torch.nn.DataParallel(Generator(g_output_dim=mnist_dim)).to(device)
    # D = torch.nn.DataParallel(Discriminator(mnist_dim)).to(device)

    G = Generator(g_output_dim=mnist_dim).to(device)
    D = Critic(mnist_dim).to(device)


    # model = DataParallel(model).to(device)
    print('Model loaded.')
    # Optimizer


    # define optimizers
    G_optimizer = optim.RMSprop(G.parameters(), lr = args.lr)
    D_optimizer = optim.RMSprop(D.parameters(), lr = args.lr)

    print('Start Training :')
    D_losses = []

    n_epoch = args.epochs
    for epoch in trange(1, n_epoch + 1, leave=True):
        epoch_D_loss = 0
        num_batches = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim)
            
            # Create a new iterator for train_loader to fetch fresh batches
            data_iter = iter(train_loader)
            
            # Update Discriminator using different batches in each iteration
            for _ in range(args.critic_iterations):
                try:
                    x, _ = next(data_iter)  # Get the next batch
                except StopIteration:
                    # Restart the iterator if it runs out of data
                    data_iter = iter(train_loader)
                    x, _ = next(data_iter)
                
                x = x.view(-1, mnist_dim)
                D_loss = WD_train(x, G, D, D_optimizer, weight_clip=args.weight_clip)
                epoch_D_loss += -D_loss
                num_batches += 1
            
            # Generator training after critic iterations
            G_loss = WG_train(x, G, D, G_optimizer)
        
        average_D_loss = epoch_D_loss / num_batches
        D_losses.append(average_D_loss)
                
        if epoch % 5 == 0:
            save_modelsv2(G, D, f'checkpoints_train/wG2_{epoch}.pth', f'checkpoints_train/wD2_{epoch}.pth')

    print('Training done')
    plt.plot(range(1, n_epoch + 1), D_losses)
    plt.xlabel('Epoch')
    plt.ylabel('D_loss')
    plt.title('Discriminator Loss over Epochs')
    
    # Save the plot with n_epoch in the filename
    plot_filename = f'project/plots/D_loss_diff_batch{n_epoch}.png'
    plt.savefig(plot_filename)
    plt.show()
