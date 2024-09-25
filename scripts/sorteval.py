import time
from numpy import imag
import torch
import os
import json
import glob
import random
import openai
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import concurrent
import base64

def load_topk_indices(output_dir: str, feature_idx: int) -> List[int]:
    json_file = os.path.join(output_dir, f'feature_{feature_idx}_top9.json')
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON file for feature {feature_idx} not found.")
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data.get('indices', [])

def load_image_paths(output_dir: str, feature_idx: int) -> List[str]:
    """
    Loads the top 9 image paths for a given feature index.
    Assumes images are named as 'feature_{feature_idx}_top9_{i}.png' where i is 1-9.
    """
    image_files = sorted(glob.glob(os.path.join(output_dir, f'feature_{feature_idx}_top9_*.png')))
    if len(image_files) != 9:
        print(image_files)
        raise ValueError(f"Expected 9 images for feature {feature_idx}, but found {len(image_files)}.")
    return image_files

def encode_image_to_base64(image_path: str) -> str:
    """
    Encodes an image file to a base64 string.

    Args:
        image_path (str): The file path to the image.

    Returns:
        str: The base64-encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def construct_prompt(
    feature1_idx: int,
    feature2_idx: int,
    query_example: Dict,
    output_dir: str
) -> List[Dict]:
    """
    Constructs the prompt with text and individual images for GPT-4 evaluation.

    Args:
        feature1_idx (int): Index of the first feature.
        feature2_idx (int): Index of the second feature.
        query_example (Dict): The query example to evaluate.
        output_dir (str): Directory where images are stored.

    Returns:
        List[Dict]: A list of content blocks containing text and images.
    """
    content = []

    # Introduction Text
    intro_text = """
    In this task, we will provide you with information about two neurons and a single example. Your job is to accurately predict which of the two neurons is more likely to be activated by the given example.

    Each neuron activates for a specific concept. To help you understand the concepts represented by the neurons, we are providing you with a set of examples that caused each neuron to activate. Each example shows a string fragment. Note that a neuron's activity may be influenced just as much by the surrounding context as the activating token, so make sure to pay attention to the full string fragments.
    """
    content.append({"type": "text", "text": intro_text})

    # Neuron 1 Images
    neuron1_image_paths = load_image_paths(output_dir, feature1_idx)
    for img_path in neuron1_image_paths:
        print(f"Neuron 1 image path: {img_path}")
        encoded_image = encode_image_to_base64(img_path)
        image_block = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}",
                "detail": "low"
            }
        }
        content.append(image_block)

    # Neuron 2 Images
    neuron2_image_paths = load_image_paths(output_dir, feature2_idx)
    for img_path in neuron2_image_paths:
        print(f"Neuron 2 image path: {img_path}")
        encoded_image = encode_image_to_base64(img_path)
        image_block = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}",
                "detail": "low"
            }
        }
        content.append(image_block)

    # Specific Example Info
    specific_example_text = "\nSpecific Example:\n"
    specific_example_text += json.dumps(query_example, indent=2)
    content.append({"type": "text", "text": specific_example_text})

    # Specific Example Image
    query_image_path = query_example.get("file_path")
    if query_image_path and os.path.exists(query_image_path):
        encoded_query_image = encode_image_to_base64(query_image_path)
        query_image_block = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_query_image}",
                "detail": "low"
            }
        }
        content.append(query_image_block)

    # Final Instruction
    final_instruction = """
    Now we will present you with an example. Based on your understanding of the neurons' functions, please predict which neuron is more likely to be activated by this example.

    Which neuron is more likely to be activated by this example? Think step-by-step, but end your answer with "ANSWER: 1" or "ANSWER: 2." Please adhere to this format and do not write anything after your answer. If you're not confident, please still provide your best guess.
    """
    content.append({"type": "text", "text": final_instruction})

    return content

def send_to_gpt4(content_blocks: List[Dict]) -> str:
    """
    Sends the prompt with images to GPT-4 and returns the response.

    Args:
        content_blocks (List[Dict]): A list of content blocks containing text and images.

    Returns:
        str: The response from GPT-4.
    """
    # Convert content_blocks to a single string with appropriate formatting
    # Assuming GPT-4 can parse the structured content; otherwise, adjust accordingly
    # Here, we'll concatenate text and image URLs in order

    prompt = ""
    for block in content_blocks:
        if block["type"] == "text":
            prompt += block["text"] + "\n\n"
        elif block["type"] == "image_url":
            prompt += f"![Image]({block['image_url']['url']})\n\n"

    messages = [
        {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Correct model name
            messages=messages,
            temperature=0,      # Set to 0 for deterministic responses
            max_tokens=750      # Adjust based on your needs
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"Error communicating with OpenAI API: {e}")
        return "Error"

def evaluate_pair(idx, feature1_idx, feature2_idx, all_latents, all_file_paths, topk_indices_dict, output_dir, evaluations):
    eval_start = time.time()
    print(f"\nProcessing pair {idx}: Features {feature1_idx} and {feature2_idx}")

    # Load top9 indices for both features
    topk1_indices = topk_indices_dict[feature1_idx]
    topk2_indices = topk_indices_dict[feature2_idx]

    # Get images that activate for feature1 (positive activations)
    feature1_values = all_latents[:, feature1_idx]
    activated_indices_feature1 = (feature1_values > 0).nonzero(as_tuple=True)[0]

    # Exclude top 9 images from both features
    excluded_indices = torch.tensor(
        list(set(topk1_indices + topk2_indices))
    )

    # Exclude all indices which activate for feature 2
    activated_indices_feature2 = (all_latents[:, feature2_idx] > 0).nonzero(as_tuple=True)[0]
    excluded_indices = torch.cat([excluded_indices, activated_indices_feature2])

    # Remaining images after exclusion
    remaining_indices = torch.tensor(
        [i.item() for i in activated_indices_feature1 if i.item() not in excluded_indices.tolist()]
    )

    print('Remaining indices:', remaining_indices.shape)

    if remaining_indices.numel() == 0:
        print("No remaining images after excluding top images.")
        return None

    # Select one query image at random
    query_index = random.choice(remaining_indices.tolist())
    query_path = all_file_paths[query_index]

    print(f"Query image path: {query_path}")

    # Prepare the query example
    query_example = {
        "file_path": query_path
        # Activation scores removed as per request
        # Add more details if available
    }

    # Construct the prompt
    prompt = construct_prompt(
        feature1_idx=feature1_idx,
        feature2_idx=feature2_idx,
        query_example=query_example,
        output_dir=output_dir
    )

    # Send the prompt to GPT-4 and get the response
    answer = send_to_gpt4(prompt)

    print(f"GPT-4 Response: {answer}")

    return {
        "feature1_idx": feature1_idx,
        "feature2_idx": feature2_idx,
        "query_example": query_example,
        "gpt4_response": answer,
        'correct_index': query_index,
        'correct': True if 'ANSWER: 1' in answer else False,
        'time_taken': time.time() - eval_start
    }

def main(n_evals=5, n_workers=1):
    with open('.credentials.json', 'r') as f:
        credentials = json.load(f)
    
    openai.api_key = credentials['OPENAI_API_KEY']

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
        # Remove the check for grid image
        # png_file = os.path.join(output_dir, f'feature_{feature_idx}_top9.png')
        # if os.path.exists(json_file) and os.path.exists(png_file):
        if os.path.exists(json_file):
            features_with_activation.append(feature_idx)
            with open(json_file, 'r') as f:
                data = json.load(f)
                topk_indices = data.get('indices', [])
                topk_indices_dict[feature_idx] = topk_indices

    total_features = len(features_with_activation)

    n_combinations = total_features * (total_features - 1) // 2

    if n_combinations < n_evals:
        print(f"Not enough feature pairs to make {n_evals} pairs without replacement.")
        print(f"Total feature pairs: {n_combinations}")
        return

    feature_pairs = set()

    while len(feature_pairs) < n_evals:
        a, b = random.sample(features_with_activation, 2)
        if (a, b) in feature_pairs or (b, a) in feature_pairs:
            continue
        feature_pairs.add((a, b))

    feature_pairs_list = list(feature_pairs)
    evaluations = {}  # Dictionary to store evaluations

    if n_workers == 1:
        for idx, (feature1_idx, feature2_idx) in enumerate(feature_pairs, 1):
            evaluation = evaluate_pair(
                idx, feature1_idx, feature2_idx, all_latents, all_file_paths, topk_indices_dict, output_dir, evaluations
            )

            if evaluation is not None:
                evaluations[f'pair_{idx}'] = evaluation
                print(f"Correct: {evaluation['correct']}")
                print('time_taken', evaluations[f'pair_{idx}']['time_taken'])
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(
                    evaluate_pair,
                    idx=i+1,
                    feature1_idx=pair[0],
                    feature2_idx=pair[1],
                    all_latents=all_latents,
                    all_file_paths=all_file_paths,
                    topk_indices_dict=topk_indices_dict,
                    output_dir=output_dir
                ): i+1 for i, pair in enumerate(feature_pairs_list)
            }

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    evaluation = future.result()
                    if evaluation is not None:
                        evaluations[f'pair_{idx}'] = evaluation
                        print(f"Pair {idx} Correct: {evaluation['correct']}")
                        print(f"Pair {idx} Time Taken: {evaluation['time_taken']:.2f} seconds")
                except Exception as exc:
                    print(f"Pair {idx} generated an exception: {exc}")

    # Calculate and print the average accuracy
    correct = sum(1 for v in evaluations.values() if v['correct'])
    accuracy = correct / len(evaluations) if evaluations else 0
    print(f"\nAverage accuracy: {accuracy:.2f}")

    # Save all evaluations to a JSON file
    with open('gpt4_evaluations.json', 'w') as f:
        json.dump(evaluations, f, indent=2)

    print("\nAll evaluations have been saved to 'gpt4_evaluations.json'.")

if __name__ == '__main__':
    main(n_evals=10)
