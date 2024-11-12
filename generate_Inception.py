import argparse
import os
import torch
import torchvision
import numpy as np
from device import device
from model import Generator
from cnn_classification import CNNModel
from utils import load_model
from inception_score import inception_score
from project.metrics import compute_fid

def generate_with_rejection_sampling(generator, classifier, batch_size, max_images, score_threshold):
    os.makedirs('samples_filtered', exist_ok=True)
    n_samples = 0
    total_inception_scores = []
    accepted_images = []
    
    with torch.no_grad():
        while n_samples < max_images:
            # Générer un lot d'images
            z = torch.randn(batch_size, 100).to(device)
            generated_images = generator(z)
            generated_images = generated_images.view(batch_size, 1, 28, 28)
            
            # Calculer l'Inception Score pour le lot d'images
            mean_score, _ = inception_score(generated_images, classifier)
            
            # Vérifier si le score est au-dessus du seuil
            if mean_score >= score_threshold:
                for i in range(batch_size):
                    # Sauvegarder chaque image acceptée
                    torchvision.utils.save_image(generated_images[i], os.path.join('samples_filtered', f'{n_samples}.png'))
                    accepted_images.append(generated_images[i])  # Ajouter l'image à la liste des images acceptées
                    n_samples += 1
                    total_inception_scores.append(mean_score)
                    
                    if n_samples >= max_images:
                        break
            else:
                print(f"Lot rejeté : score Inception = {mean_score} < seuil = {score_threshold}")

    # Calculer et afficher le score moyen des images acceptées
    final_score = np.mean(total_inception_scores)
    print(f"Score Inception final (images retenues) : {final_score}")
    
    # Convertir la liste des images acceptées en un tenseur unique
    accepted_images_tensor = torch.stack(accepted_images)
    
    return accepted_images_tensor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Images with Rejection Sampling based on Inception Score.')
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for generation.")
    parser.add_argument("--max_images", type=int, default=100, help="Number of images to save.")
    parser.add_argument("--score_threshold", type=float, default=5.4, help="Minimum Inception Score threshold.")
    args = parser.parse_args()

    # Chargement du modèle de générateur
    mnist_dim = 784
    generator = Generator(g_output_dim=mnist_dim).to(device)
    generator = load_model(generator, 'checkpoints')
    generator = torch.nn.DataParallel(generator).to(device)
    generator.eval()

    # Chargement du modèle de classification
    classifier = CNNModel().to(device)
    classifier.load_state_dict(torch.load('models/model_cnn_classifier.ckpt', map_location=device))
    classifier.eval()

    # Générer les images en appliquant le rejection sampling
    generated_images_tensor = generate_with_rejection_sampling(generator, classifier, args.batch_size, args.max_images, args.score_threshold)
    print(f"score fid : {compute_fid(generated_images_tensor, args.batch_size)}")