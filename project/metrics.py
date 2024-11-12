import torch
import torchvision.datasets
from torcheval.metrics import FrechetInceptionDistance
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def compute_metrics(real: torch.Tensor, generated: torch.Tensor, *, k: int):
  def f(test: torch.Tensor, ref: torch.Tensor):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(ref.numpy())

    distances, _ = neigh.kneighbors()
    thresholds = torch.as_tensor(distances[:, -1]) ** 2

    distances_test = ((test[:, None, :] - ref[None, :, :]) ** 2).sum(dim=2)
    return (distances_test <= thresholds).any(dim=1).to(torch.float32).mean().item()

  return (
    f(generated, real),
    f(real, generated),
  )


# def compute_fid(real: torch.Tensor, generated: torch.Tensor):
#   fid = FrechetInceptionDistance()
#   fid.update(real, True)
#   fid.update(generated, False)

#   return fid.compute()


def compute_fid(generated_data: torch.Tensor, batch_size: int):
  transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(mean=0.5, std=0.5),
  ])

  def transform_image(x: torch.Tensor):
    return (x.view(-1, 1, 28, 28).repeat((1, 3, 1, 1)) * 2.0 - 1.0).clamp(0.0, 1.0)

  dataset = torchvision.datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)

  indices = torch.randperm(len(dataset))
  dataset = torch.utils.data.Subset(dataset, indices[:generated_data.size(0)])

  real_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
  generated_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(generated_data), batch_size=batch_size, shuffle=False)

  fid = FrechetInceptionDistance()
  
  print("Calcul du FID pour les images réelles...")

  for real, _ in tqdm(real_loader, desc="Processing real images"):
    fid.update(transform_image(real), True)

  print("Calcul du FID pour les images générées...")
  for generated, in tqdm(generated_loader, desc="Processing generated images"):
    fid.update(transform_image(generated), False)

  return fid.compute()