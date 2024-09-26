import torch
import os
import json
import glob
from PIL import Image
import matplotlib.pyplot as plt

def get_top9(latent_dir):
    file_path_files = sorted(glob.glob(os.path.join(latent_dir, 'file_paths_*.json')))
    latent_files = sorted(glob.glob(os.path.join(latent_dir, 'latents_*.pt')))

    image_dir = os.path.join(latent_dir, 'images')

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    print('saving images to:', image_dir)


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

        feature_dir = os.path.join(image_dir, f'feature_{feature_idx}')

        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)

        with open(os.path.join(feature_dir, f'{feature_idx}_top9.json'), 'w') as f:
            json.dump(result, f)

        # Load the images
        images = []
        for i, path in enumerate(topk_file_paths):

            if i >= num_top:
                break    
            img = Image.open(path).convert('RGB')
            images.append(img)

            # save image
            img.save(os.path.join(feature_dir, f'{feature_idx}_top9_{i}.png'))


        # Plot the images in a 3x3 grid
        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        for ax, img in zip(axes.flatten(), images):
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(feature_dir, f'{feature_idx}_grid.png'))
        plt.close(fig)

        print(f'saved images for feature {feature_idx}')

if __name__ == '__main__':
    get_top9('cruft/ViT-3mil-topkk-32-experts-None_1aaa89/latents-2969600')