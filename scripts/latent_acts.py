import sys
import json
import os
import torch
from torch.utils.data import DataLoader 
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sache.hookedvit import SpecifiedHookedViT
from sache.imgloader import FilePathDataset
from top9 import get_top9

def main(
        sae_path,
        n_activations=250_000,
        save_every=20 * 16,
        batch_size=256,
        transformer_name='laion/CLIP-ViT-L-14-laion2B-s32B-b82K',
        hook_name="resid",
        layer=-2,
        device='cuda'
    ):

    data_directory = 'images'
    dataset = FilePathDataset(root_dir=data_directory)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    transformer = SpecifiedHookedViT(layer, hook_name, transformer_name, device=device)
    sae = torch.load(sae_path)
    
    n_steps = sae_path.split('/')[-1].split('.')[0]
    save_dir = "/".join(sae_path.split('/')[:-1]) + f'/latents-{n_steps}'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    latents = None
    file_paths = []
    with torch.no_grad():
        for i, (paths, batch) in tqdm(enumerate(dataloader), total=n_activations // batch_size):
            activations = transformer.cls_activations(batch)

            latent = sae.forward_descriptive(activations)['latent'].detach().cpu()
            if latents is None:
                latents = latent
            else:
                latents = torch.cat((latents, latent), dim=0)

            file_paths.extend(paths)

            if i > 0 and i % save_every == 0:
                with open(f'{save_dir}/file_paths_{i}.json', 'w') as f:
                    json.dump(file_paths, f)
                torch.save(latents, f'{save_dir}/latents_{i}.pt')
                file_paths = []
                latents = None

            if i * batch_size > n_activations:
                if len(file_paths) > 0:
                    with open(f'{save_dir}/file_paths_{i}.json', 'w') as f:
                        json.dump(file_paths, f)
                    torch.save(latents, f'{save_dir}/latents_{i}.pt')
                break

    print('finished generating latents')
    latent_dir = 'cruft/650_latents'
    get_top9(latent_dir, save_dir)


if __name__ == '__main__':
    main(
        sae_path='cruft/ViT-45_000_000-relu-l1-3e-05_e5542e/23757696.pt',
        n_activations=10_000,
    )