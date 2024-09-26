import time
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
    image_files = sorted(glob.glob(os.path.join(output_dir, f'feature_{feature_idx}_top9_*.png')))
    if len(image_files) != 9:
        print(image_files)
        raise ValueError(f"Expected 9 images for feature {feature_idx}, but found {len(image_files)}.")
    return image_files

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def construct_prompt(
    f1grid_file: str,
    f2grid_file: str,
    query_example_path: str,
) -> List[Dict]:
    content = []

    # Introduction Text
    intro_text = """
    In this task, we will provide you with information about two neurons and a single example. Your job is to accurately predict which of the two neurons is more likely to be activated by the given example.

    Each neuron activates for a specific concept. To help you understand the concepts represented by the neurons, we are providing you with a set of example images that caused each neuron to activate.
    """
    content.append({"type": "text", "text": intro_text})


    neuron2_text = f"\nNeuron 1 Examples:\n"
    content.append({"type": "text", "text": neuron2_text})

    # Neuron 1 Images
    print(f"Neuron 1 image path: {f1grid_file}")
    encoded_image = encode_image_to_base64(f1grid_file)
    image_block = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{encoded_image}",
            "detail": "high"
        }
    }
    content.append(image_block)

    neuron2_text = f"\nNeuron 2 Examples:\n"
    content.append({"type": "text", "text": neuron2_text})
    
    print(f"Neuron 2 image path: {f2grid_file}")
    encoded_image = encode_image_to_base64(f2grid_file)
    image_block = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{encoded_image}",
            "detail": "high"
        }
    }
    content.append(image_block)

    # Specific Example Info
    specific_example_text = "\nSpecific Example:\n"
    content.append({"type": "text", "text": specific_example_text})

    # Specific Example Image
    query_image_path = query_example_path
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
    Now we will present you with another example image. Based on your understanding of the neurons' functions, please predict which neuron is more likely to be activated by this example.

    Which neuron is more likely to be activated by this example? Think step-by-step, but end your answer with "ANSWER: 1" or "ANSWER: 2." Please adhere to this format and do not write anything after your answer. If you're not confident, please still provide your best guess.
    """
    content.append({"type": "text", "text": final_instruction})

    return content

def send_to_gpt4(content_blocks: List[Dict]) -> str:
    messages = [
        {"role": "user", "content": content_blocks}
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,      # Set to 0 for deterministic responses
            max_tokens=750      # Adjust based on your needs
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"Error communicating with OpenAI API: {e}")
        return "Error"

def save_to_test_dir(test_dir, f1grid_file, f2grid_file, feature1_idx, feature2_idx, query_path):
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)


    this_test_dir = os.path.join(test_dir, f'test_{feature1_idx}-{feature2_idx}')
    if not os.path.exists(os.path.join(test_dir, this_test_dir)):
        os.makedirs(os.path.join(test_dir, this_test_dir))

    os.symlink(f1grid_file, os.path.join(test_dir, this_test_dir, f'{feature1_idx}_grid.png'))
    os.symlink(f2grid_file, os.path.join(test_dir, this_test_dir, f'{feature2_idx}_grid.png'))
    os.symlink(query_path, os.path.join(test_dir, this_test_dir, 'query.png'))

def evaluate_pair(idx, feature1_idx, feature2_idx, all_latents, all_file_paths, topk_indices_dict, top9dir, test_dir=None):
    eval_start = time.time()
    print(f"\nProcessing pair {idx}: Features {feature1_idx} and {feature2_idx}")

    # Randomly select which feature to use for querying: 1 or 2
    selected_feature_number = random.choice([1, 2])
    if selected_feature_number == 1:
        selected_feature_idx = feature1_idx
        other_feature_idx = feature2_idx
        print(f"Selected Feature 1 (Index: {selected_feature_idx}) for querying.")
    else:
        selected_feature_idx = feature2_idx
        other_feature_idx = feature1_idx
        print(f"Selected Feature 2 (Index: {selected_feature_idx}) for querying.")

    # Load topk indices for both features
    topk_selected = topk_indices_dict[selected_feature_idx]
    topk_other = topk_indices_dict[other_feature_idx]

    # Get images that activate for the selected feature (positive activations)
    selected_feature_values = all_latents[:, selected_feature_idx]
    activated_indices_selected = (selected_feature_values > 0).nonzero(as_tuple=True)[0]

    # Exclude topk images from both features
    excluded_indices = torch.tensor(
        list(set(topk_selected + topk_other))
    )

    # Exclude all indices which activate for the other feature
    other_feature_values = all_latents[:, other_feature_idx]
    activated_indices_other = (other_feature_values > 0).nonzero(as_tuple=True)[0]
    excluded_indices = torch.cat([excluded_indices, activated_indices_other])

    # Remaining images after exclusion
    remaining_indices = torch.tensor(
        [i.item() for i in activated_indices_selected if i.item() not in excluded_indices.tolist()]
    )

    print('Remaining indices:', remaining_indices.shape)

    if remaining_indices.numel() == 0:
        print("No remaining images after excluding top images.")
        return None

    # Select one query image at random from the selected feature
    query_index = random.choice(remaining_indices.tolist())
    query_path = all_file_paths[query_index]

    print(f"Query image path: {query_path}")

    # Prepare the query example
    f1grid_file = os.path.join(top9dir, f'feature_{feature1_idx}', f'{feature1_idx}_grid.png')
    f2grid_file = os.path.join(top9dir, f'feature_{feature2_idx}', f'{feature2_idx}_grid.png')
    
    # Construct the prompt
    prompt = construct_prompt(
        f1grid_file=f1grid_file,
        f2grid_file=f2grid_file,
        query_example_path=query_path,
    )

    if test_dir is not None:
        save_to_test_dir(test_dir, f1grid_file, f2grid_file, feature1_idx, feature2_idx, query_path)

    # Send the prompt to GPT-4 and get the response
    answer = send_to_gpt4(prompt)

    print(f"GPT-4 Response: {answer}")

    # Determine if the GPT-4 answer is correct
    if selected_feature_number == 1:
        correct_answer = 'ANSWER: 1'
    else:
        correct_answer = 'ANSWER: 2'

    is_correct = correct_answer in answer.upper()

    return {
        "feature1_idx": feature1_idx,
        "feature2_idx": feature2_idx,
        "selected_feature_number": selected_feature_number,  # Indicates which feature was selected
        "query_example": query_path,
        "gpt4_response": answer,
        'correct_index': query_index,
        'correct': is_correct,
        'time_taken': time.time() - eval_start
    }

def run_sort_eval(
        latent_dir,
        n_evals=5, 
        n_workers=1,
        num_features=650,
    ):
    with open('.credentials.json', 'r') as f:
        credentials = json.load(f)
    openai.api_key = credentials['OPENAI_API_KEY']
    image_dir = os.path.join(latent_dir, 'images')

    file_path_files = sorted(glob.glob(os.path.join(latent_dir, 'file_paths_*.json')))
    latent_files = sorted(glob.glob(os.path.join(latent_dir, 'latents_*.pt')))

    all_file_paths = []
    all_latents = []

    for fp_file, latent_file in zip(file_path_files, latent_files):
        with open(fp_file, 'r') as f:
            file_paths = json.load(f)
        latents = torch.load(latent_file)

        all_file_paths.extend(file_paths)
        all_latents.append(latents)

    all_latents = torch.cat(all_latents, dim=0)  # Shape: (num_images, num_features)

    features_with_activation = []
    topk_indices_dict = {}  # Store topk indices for each feature

    for feature_idx in range(num_features):
        json_file = os.path.join(image_dir, f'feature_{feature_idx}/{feature_idx}_top9.json')
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

    if n_workers <= 1:
        for idx, (feature1_idx, feature2_idx) in enumerate(feature_pairs, 1):
            evaluation = evaluate_pair(
                idx, feature1_idx, feature2_idx, all_latents, all_file_paths, topk_indices_dict, image_dir
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
                    image_dir=image_dir
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

    correct = sum(1 for v in evaluations.values() if v['correct'])
    accuracy = correct / len(evaluations) if evaluations else 0
    print(f"\nAverage accuracy: {accuracy:.2f}")

    with open(os.path.join(image_dir, 'gpt4_evaluations.json'), 'w') as f:
        json.dump(evaluations, f, indent=2)

    print("\nAll evaluations have been saved to 'gpt4_evaluations.json'.")

if __name__ == '__main__':
    run_sort_eval(
        latent_dir = 'cruft/ViT-3mil-topkk-32-experts-None_1aaa89/latents-2969600',
        n_evals=2,
    )
