import torch
import os
import json
import glob
from PIL import Image
import matplotlib.pyplot as plt

def main():
    save_dir = 'cruft/ViT-45_000_000-relu-l1-3e-05_e5542e/latents-23757696/' 

    output_dir = 'cruft/650_latents'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path_files = sorted(glob.glob(os.path.join(save_dir, 'file_paths_*.json')))
    latent_files = sorted(glob.glob(os.path.join(save_dir, 'latents_*.pt')))

    all_file_paths = []
    all_latents = []

    for fp_file, latent_file in zip(file_path_files, latent_files):
        with open(fp_file, 'r') as f:
            file_paths = json.load(f)
        latents = torch.load(latent_file)

        all_file_paths.extend(file_paths)
        all_latents.append(latents)

    all_latents = torch.cat(all_latents, dim=0)

    num_features = 650
    num_top = 9  # We need top 9 images per feature

    for feature_idx in range(num_features):
        print(feature_idx)
        feature_values = all_latents[:, feature_idx]

        topk_values, topk_indices = torch.topk(feature_values, num_top + 1) # we want at least one extra

        if topk_values.min() <= 0:
            print(f'Feature {feature_idx} has less than 9 associated images')
            continue

        # Get the corresponding file paths
        topk_file_paths = [all_file_paths[i] for i in topk_indices]

        # Save the values and indices
        result = {
            'indices': topk_indices.tolist(),
            'values': topk_values.tolist(),
            'file_paths': topk_file_paths
        }
        with open(os.path.join(output_dir, f'feature_{feature_idx}_top9.json'), 'w') as f:
            json.dump(result, f)

        # Load the images
        images = []
        for path in topk_file_paths:
            img = Image.open(path).convert('RGB')
            images.append(img)

        # Plot the images in a 3x3 grid
        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        for ax, img in zip(axes.flatten(), images):
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'feature_{feature_idx}_top9.png'))
        plt.close(fig)

        print(f'Processed feature {feature_idx}')

if __name__ == '__main__':
    main()
