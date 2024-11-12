import torch
import torchvision.datasets
from torcheval.metrics import FrechetInceptionDistance
from sklearn.neighbors import NearestNeighbors


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


def compute_fid(real_data: torch.Tensor):
  transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(mean=0.5, std=0.5),
  ])

  dataset = torchvision.datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)
  batch_size = 100

  real_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
  generated_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(real_data), batch_size=batch_size, shuffle=False)

  fid = FrechetInceptionDistance()

  for real in real_loader:
    fid.update(real, True)

  for generated in generated_loader:
    fid.update(generated, False)

  return fid.compute()
