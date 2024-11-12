import argparse
import os
import torch
import torchvision
from device import device
from model import Generator
from utils import load_model
from cnn_classification import CNNModel  # Importer ton modèle CNN

def generate_images_with_rejection_sampling(generator, classifier, batch_size, threshold, max_images=10000):
    os.makedirs('samples_filtered', exist_ok=True)
    n_samples = 0
    
    with torch.no_grad():
        while n_samples < max_images:
            # Génération des images
            z = torch.randn(batch_size, 100).to(device)
            generated_images = generator(z)
            generated_images = generated_images.view(batch_size, 1, 28, 28)
            
            # Calcul des probabilités de classes avec le classifieur
            probabilities = torch.softmax(classifier(generated_images), dim=1)
            max_probs, _ = probabilities.max(dim=1)  # Probabilité maximale pour chaque image

            # Filtrage des images avec une probabilité maximale inférieure au seuil
            for i in range(batch_size):
                if max_probs[i] >= threshold:
                    # Sauvegarder l'image si elle passe le filtre
                    torchvision.utils.save_image(generated_images[i], os.path.join('samples_filtered', f'{n_samples}.png'))
                    n_samples += 1
                    if n_samples >= max_images:
                        break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Images with Rejection Sampling.')
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for generation.")
    parser.add_argument("--threshold", type=float, default=0.8, help="Confidence threshold for rejection sampling.")
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
    generate_images_with_rejection_sampling(generator, classifier, args.batch_size, args.threshold)