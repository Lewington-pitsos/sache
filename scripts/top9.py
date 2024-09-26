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

    num_features = 650
    num_top = 9  # We need top 9 images per feature

    # Initialize topk values and indices for all features
    topk_values = torch.full((num_features, num_top), float('-inf'))
    topk_indices = torch.full((num_features, num_top), -1, dtype=torch.long)
    topk_file_paths = [[''] * num_top for _ in range(num_features)]
    indices_gt_zero = [[] for _ in range(num_features)]

    cumulative_index = 0  # Keeps track of the global index across batches
    cumulative_file_paths = []

    for batch_num, (fp_file, latent_file) in enumerate(zip(file_path_files, latent_files)):
        print(f'Processing batch {batch_num+1}/{len(latent_files)}')
        with open(fp_file, 'r') as f:
            file_paths = json.load(f)
        latents = torch.load(latent_file)[:, :650]  # shape: [batch_size, num_features]

        batch_size = latents.size(0)
        current_indices = torch.arange(cumulative_index, cumulative_index + batch_size, dtype=torch.long)
        cumulative_index += batch_size

        cumulative_file_paths.extend(file_paths)

        # Transpose latents to shape [num_features, batch_size]
        current_values = latents.t()  # shape: [num_features, batch_size]

        # Expand current_indices to shape [num_features, batch_size]
        current_indices_expanded = current_indices.unsqueeze(0).expand(num_features, -1)  # shape: [num_features, batch_size]

        # Concatenate current batch with previous topk
        total_values = torch.cat((topk_values, current_values), dim=1)  # shape: [num_features, num_top + batch_size]
        total_indices = torch.cat((topk_indices, current_indices_expanded), dim=1)  # shape: [num_features, num_top + batch_size]

        # Perform topk across the concatenated values
        topk = torch.topk(total_values, k=num_top, dim=1)
        topk_values = topk.values  # shape: [num_features, num_top]
        indices_in_total = topk.indices  # shape: [num_features, num_top]

        # Gather the corresponding global indices
        topk_global_indices = torch.gather(total_indices, 1, indices_in_total)  # shape: [num_features, num_top]

        # Update topk_indices
        topk_indices = topk_global_indices

        # Update topk_file_paths
        for feature_idx in range(num_features):
            indices_in_total_feature = indices_in_total[feature_idx]
            total_file_paths = topk_file_paths[feature_idx] + file_paths
            topk_file_paths[feature_idx] = [total_file_paths[i] for i in indices_in_total_feature.tolist()]

        # Update indices_gt_zero
        mask = current_values > 0  # shape: [num_features, batch_size]
        for feature_idx in range(num_features):
            indices = current_indices[mask[feature_idx]]
            indices_gt_zero[feature_idx].extend(indices.tolist())

    # After processing all batches, save the results
    for feature_idx in range(num_features):
        print(f'Processing feature {feature_idx}')

        if topk_values[feature_idx].min() <= 0:
            print(f'Feature {feature_idx} has less than {num_top} associated images')
            continue

        result = {
            'indices': topk_indices[feature_idx].tolist(),
            'values': topk_values[feature_idx].tolist(),
            'indices_gt_zero': indices_gt_zero[feature_idx],
            'file_paths': topk_file_paths[feature_idx]
        }

        feature_dir = os.path.join(image_dir, f'feature_{feature_idx}')
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)

        with open(os.path.join(feature_dir, f'{feature_idx}_top9.json'), 'w') as f:
            json.dump(result, f)

        # Load the images
        images = []
        for i, path in enumerate(topk_file_paths[feature_idx]):
            if i >= num_top:
                break
            img = Image.open(path).convert('RGB')
            images.append(img)

            # Save image
            img.save(os.path.join(feature_dir, f'{feature_idx}_top9_{i}.png'))

        # Plot the images in a 3x3 grid
        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        for ax, img in zip(axes.flatten(), images):
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(feature_dir, f'{feature_idx}_grid.png'))
        plt.close(fig)

        print(f'Saved images for feature {feature_idx}')

if __name__ == '__main__':
    get_top9('cruft/ViT-3mil-topkk-32-experts-None_1aaa89/latents-2969600')
