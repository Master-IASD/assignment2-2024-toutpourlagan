
import sys
import os

# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

import argparse
import os

import torch
import torchvision
from device import device
from model import Generator
from utils import load_model, load_modelv2

#Number of samples we want to generate
n = 1000

# Set the path to checkpoints in the parent directory
checkpoints_path = os.path.join(parent_dir, 'checkpoints_train/wgpG_100.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for training.")
    args = parser.parse_args()




    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784

    model = Generator(g_output_dim=mnist_dim).to(device)
    model = load_modelv2(model, checkpoints_path)
    model = torch.nn.DataParallel(model).to(device)
    model.eval()

    print('Model loaded.')



    print('Start Generating')
    #os.makedirs('samples', exist_ok=True)

    n_samples = 0
    with torch.no_grad():
        while n_samples<n:
            z = torch.randn(args.batch_size, 100).to(device)
            x = model(z)
            x = x.reshape(args.batch_size, 28, 28)
            for k in range(x.shape[0]):
                if n_samples<n:
                    torchvision.utils.save_image(x[k:k+1], os.path.join('generated_samples/WganGP', f'{n_samples}.png'))
                    n_samples += 1
