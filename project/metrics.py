import torch


def compute_metrics(real: torch.Tensor, generated: torch.Tensor, *, k: int):
  def f(test: torch.Tensor, ref: torch.Tensor):
    distances = ((ref[:, None, :] - ref[None, :, :]) ** 2).sum(dim=2)
    distances_sorted, _ = torch.sort(distances, dim=1)
    thresholds = distances_sorted[:, k]

    distances_test = ((test[:, None, :] - ref[None, :, :]) ** 2).sum(dim=2)

    return (distances_test <= thresholds).any(dim=1)

  return (
    f(generated, real).to(torch.float32).mean().item(), # Precision
    f(real, generated).to(torch.float32).mean().item(), # Recall
  )
