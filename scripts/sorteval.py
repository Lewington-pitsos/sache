import torch
import os
import json
import glob
import random
import itertools

def main(n_evals=5):
    save_dir = 'cruft/ViT-45_000_000-relu-l1-3e-05_e5542e/latents-23757696/'
    output_dir = 'cruft/650_latents'

    # Load all the latents and file paths
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

    all_latents = torch.cat(all_latents, dim=0)  # Shape: (num_images, num_features)

    num_features = 650

    # Identify features with activations (those with top9 files)
    features_with_activation = []
    topk_indices_dict = {}  # Store topk indices for each feature

    for feature_idx in range(num_features):
        json_file = os.path.join(output_dir, f'feature_{feature_idx}_top9.json')
        png_file = os.path.join(output_dir, f'feature_{feature_idx}_top9.png')
        if os.path.exists(json_file) and os.path.exists(png_file):
            features_with_activation.append(feature_idx)
            with open(json_file, 'r') as f:
                data = json.load(f)
                topk_indices = data['indices']
                topk_indices_dict[feature_idx] = topk_indices

    total_features = len(features_with_activation)


    n_combinations = total_features * (total_features - 1) // 2

    if n_combinations < n_evals:
        print(f"Not enough feature pairs to make {n_evals} pairs without replacement.")
        print(f"Total feature pairs: {n_combinations}")
        return

    feature_pairs = set()

    while len(feature_pairs) < n_evals:
        a = random.choice(features_with_activation)
        b = random.choice(features_with_activation)

        if a == b:
            continue

        if (a, b) in feature_pairs or (b, a) in feature_pairs:
            continue

        feature_pairs.add((a, b))

    # Process each pair
    for idx, (feature1_idx, feature2_idx) in enumerate(feature_pairs, 1):
        print(f"\nProcessing pair {idx}: Features {feature1_idx} and {feature2_idx}")

        # Print the filename of the 3x3 top9 file for each feature
        png_file1 = os.path.join(output_dir, f'feature_{feature1_idx}_top9.png')
        png_file2 = os.path.join(output_dir, f'feature_{feature2_idx}_top9.png')

        print(f"Top9 image file for feature {feature1_idx}: {png_file1}")
        print(f"Top9 image file for feature {feature2_idx}: {png_file2}")

        # Get the topk indices for each feature
        topk1_indices = torch.tensor(topk_indices_dict[feature1_idx], dtype=torch.long)
        topk2_indices = torch.tensor(topk_indices_dict[feature2_idx], dtype=torch.long)

        # Get images that activate for feature1 (positive activations)
        feature1_values = all_latents[:, feature1_idx]
        activated_indices_feature1 = (feature1_values > 0).nonzero(as_tuple=True)[0]

        # Exclude top 9 images from both features
        excluded_indices = torch.cat([topk1_indices, topk2_indices]).unique()

        # Remaining images after exclusion
        remaining_indices = torch.tensor(
            [idx.item() for idx in activated_indices_feature1 if idx not in excluded_indices]
        )

        if remaining_indices.numel() == 0:
            print("No remaining images after excluding top images.")
            continue

        # Select one query image at random
        query_index = random.choice(remaining_indices.tolist())
        query_path = all_file_paths[query_index]

        print(f"Query image path: {query_path}")

        if idx >= n_evals:
            break

if __name__ == '__main__':
    main(n_evals=4)
