from matplotlib import pyplot as plt
import torch
import torchvision
from tqdm import tqdm
from model import Critic, Generator
from torch import nn


# w^phi in paper
class Reweighter(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(100, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


# Discriminator: d_alpha
def train_reweighting(dataset, discriminator: nn.Module, generator: nn.Module, *, device):
    reweighter = Reweighter()

    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=4e-4, betas=(0.5, 0.5))
    optimizer_reweighter = torch.optim.Adam(reweighter.parameters(), lr=1e-4, betas=(0.5, 0.5))

    for parameter in generator.parameters():
        parameter.requires_grad = False

    batch_size = 100
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    lambda1 = 10.0
    lambda2 = 3.0
    m = 3.0

    # discriminator.load_state_dict(torch.load('checkpoints/D2.pth', weights_only=True))
    # reweighter.load_state_dict(torch.load('checkpoints/W.pth', weights_only=True))

    g = torch.Generator(device=device).manual_seed(0)
    zs = torch.randn(16, 100, device=device, generator=g)

    for _ in range(10):
        optimizer_discriminator.zero_grad()

        for _ in range(3):
            for x, _ in tqdm(loader):
                z = torch.randn(batch_size, 100, device=device)
                emd = (discriminator(x.view(-1, 784)) - reweighter(z) * torch.sigmoid(discriminator(generator(z)))).mean()
                gp = 0.0 # 1.0 * (torch.autograd.grad(emd, discriminator.parameters(), create_graph=True) - 1) ** 2
                loss = -emd + gp
                loss.backward()

                optimizer_discriminator.step()

        optimizer_reweighter.zero_grad()

        z = torch.randn(batch_size, 100, device=device)
        discriminated = torch.sigmoid(discriminator(generator(z)))
        weights = reweighter(z)

        delta = discriminated.min()
        emd = (weights * (discriminated - delta)).mean()
        r_norm = (weights.mean() - 1) ** 2
        r_clip = (torch.relu(weights - m) ** 2).mean()
        loss = emd + lambda1 * r_norm + lambda2 * r_clip

        loss.backward()
        optimizer_reweighter.step()


        torch.save(discriminator.state_dict(), 'checkpoints/D2.pth')
        torch.save(reweighter.state_dict(), 'checkpoints/W.pth')

        fig, axs = plt.subplots(ncols=4, nrows=4, figsize=(12, 12))

        with torch.no_grad():
            images = generator(zs).view(axs.size, 28, 28).cpu().numpy()
            weights = reweighter(zs)[:, 0].cpu().numpy()

        for index, ax in enumerate(axs.flat):
            ax.imshow(images[index, :, :])
            ax.set_title(f'{weights[index]:.3f}')

        fig.savefig('weights.png')



# train_reweighting()

if __name__ == '__main__':
    discriminator = Critic(784)
    discriminator.load_state_dict(torch.load('checkpoints/D.pth', weights_only=True))

    generator = Generator(784)
    generator.load_state_dict(torch.load('checkpoints/G.pth', weights_only=True))

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=0.5, std=0.5),
        # transforms.Lambda(lambda x: x.to(device)),
    ])

    dataset = torchvision.datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)

    train_reweighting(dataset, discriminator, generator, device='cpu')
