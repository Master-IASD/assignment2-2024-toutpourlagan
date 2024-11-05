import torch


def compute_metrics(real: torch.Tensor, generated: torch.Tensor, *, k: int):
  def f(test: torch.Tensor, ref: torch.Tensor):
    distances = ((ref[:, None, :] - ref[None, :, :]) ** 2).sum(dim=2)

    # Using k + 1 because the first element is the distance to itself and it thus zero
    thresholds = torch.topk(distances, k + 1, dim=1, largest=False).values.max(dim=1).values

    distances_test = ((test[:, None, :] - ref[None, :, :]) ** 2).sum(dim=2)
    return (distances_test <= thresholds).any(dim=1)

  return (
    f(generated, real).to(torch.float32).mean().item(),
    f(real, generated).to(torch.float32).mean().item(),
  )
