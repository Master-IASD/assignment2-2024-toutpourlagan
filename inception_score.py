import torch
import torch.nn.functional as F
import numpy as np
from cnn_classification import CNNModel
from device import device
from model import Generator
from utils import load_model

def inception_score(images, model, splits=10):
    """
    Calcul de l'Inception Score pour les images générées.
    
    Args:
        images (torch.Tensor): Tensor contenant les images générées (taille N x C x H x W).
        model (torch.nn.Module): Modèle CNN classifiant les images.
        splits (int): Nombre de splits pour réduire la variance du score.

    Returns:
        float: Inception Score moyen
    """
    N = len(images)
    preds = []

    # Passer chaque image générée par le modèle pour obtenir p(y|x)
    with torch.no_grad():
        for img in images:
            pred = F.softmax(model(img.unsqueeze(0)), dim=1)  # Ajouter une dimension batch
            preds.append(pred.cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    
    # Calculer l'Inception Score par groupes de `splits`
    scores = []
    for i in range(splits):
        part = preds[i * (N // splits): (i + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores.append(np.exp(np.mean(np.sum(part * (np.log(part) - np.log(py)), axis=1))))
    
    return np.mean(scores), np.std(scores)

# Exemple d'utilisation
# images = torch.randn(1000, 1, 28, 28)  # Exemples d'images générées
# model = CNN_pretrained_MNIST()  # Charger ton modèle classifieur
# mean_score, std_score = inception_score(images, model)
# print(f"Inception Score: {mean_score} ± {std_score}")


def IS(path_model_cnn = 'models/model_cnn_classifier.ckpt', path_model_gan = 'checkpoints'):
    
    model_cnn = CNNModel()
    model_cnn.load_state_dict(torch.load(path_model_cnn, map_location=device))
    model_cnn.to(device)
    model_cnn.eval()

    mnist_dim = 784
    model = Generator(g_output_dim=mnist_dim).to(device)
    model = load_model(model, 'checkpoints')
    model = torch.nn.DataParallel(model).to(device)
    model.eval()

    z = torch.randn(2048, 100).to(device)
    images = model(z)
    images = images.view(images.size(0), 1, 28, 28)

    mean_score, std_score = inception_score(images, model_cnn)
    return mean_score, std_score

