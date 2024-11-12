import argparse
import os
import torch
import torchvision
from device import device
from model import Generator
from utils import load_model
from cnn_classification import CNNModel  # Assure-toi d'importer le modèle classificateur
from inception_score import inception_score  # Assure-toi d'importer la fonction IS

# Paramètres pour le calcul de l'Inception Score
SCORE_INTERVAL = 500  # Intervalle pour calculer le score
BATCH_SIZE = 2048  # Taille du lot pour générer des images
TOTAL_IMAGES = 10000  # Nombre total d'images à générer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate and evaluate GAN images.')
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for generation.")
    args = parser.parse_args()

    # Charger le générateur
    print('Loading Generator model...')
    mnist_dim = 784
    model = Generator(g_output_dim=mnist_dim).to(device)
    model = load_model(model, 'checkpoints')
    model = torch.nn.DataParallel(model).to(device)
    model.eval()
    print('Generator model loaded.')

    # Charger le classificateur CNN pour l'Inception Score
    print('Loading CNN classifier for Inception Score...')
    model_cnn = CNNModel()
    model_cnn.load_state_dict(torch.load('models/model_cnn_classifier.ckpt', map_location=device))
    model_cnn.to(device)
    model_cnn.eval()
    print('CNN classifier loaded.')

    # Génération et évaluation
    print('Starting generation...')
    os.makedirs('samples', exist_ok=True)
    n_samples = 0
    generated_images = []

    with torch.no_grad():
        while n_samples < TOTAL_IMAGES:
            z = torch.randn(args.batch_size, 100).to(device)
            x = model(z)
            x = x.reshape(args.batch_size, 1, 28, 28)
            
            for img in x:
                if n_samples < TOTAL_IMAGES:
                    generated_images.append(img)
                    torchvision.utils.save_image(img, os.path.join('samples', f'{n_samples}.png'))
                    n_samples += 1
            
            # Calculer l'Inception Score toutes les SCORE_INTERVAL images générées
            if n_samples % SCORE_INTERVAL == 0:
                print(f'\nCalculating Inception Score at {n_samples} samples...')
                images_tensor = torch.stack(generated_images).to(device)
                mean_score, std_score = inception_score(images_tensor, model_cnn)
                print(f"Inception Score at {n_samples} samples: {mean_score:.4f} ± {std_score:.4f}")
                
                # Réinitialiser la liste pour éviter de dépasser la mémoire
                generated_images = []

    print("Generation and evaluation completed.")