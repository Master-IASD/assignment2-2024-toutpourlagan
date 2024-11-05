import torch
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
