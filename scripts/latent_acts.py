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
        save_every=10,
        batch_size=2048,
        save_up_to=650,
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
    latent_dir = "/".join(sae_path.split('/')[:-1]) + f'/latents-{n_steps}'

    if not os.path.exists(latent_dir):
        os.makedirs(latent_dir)

    latents = None
    file_paths = []
    with torch.no_grad():
        for i, (paths, batch) in tqdm(enumerate(dataloader), total=n_activations // batch_size):
            activations = transformer.cls_activations(batch)

            latent = sae.forward_descriptive(activations)['latent'].detach().cpu()
            latent = latent[:, :save_up_to]

            if latents is None:
                latents = latent
            else:
                latents = torch.cat((latents, latent), dim=0)

            file_paths.extend(paths)


            if i > 0 and i % save_every == 0:
                with open(f'{latent_dir}/file_paths_{i}.json', 'w') as f:
                    json.dump(file_paths, f)
                torch.save(latents, f'{latent_dir}/latents_{i}.pt')
                file_paths = []
                latents = None

            if i * batch_size > n_activations:
                if len(file_paths) > 0:
                    with open(f'{latent_dir}/file_paths_{i}.json', 'w') as f:
                        json.dump(file_paths, f)
                    torch.save(latents, f'{latent_dir}/latents_{i}.pt')
                break

    get_top9(latent_dir)


if __name__ == '__main__':
    # main(
    #     sae_path='cruft/ViT-3mil-topkk-32-experts-None_1aaa89/2969600.pt',
    #     n_activations=250_000,
    # )

    # main( 
    #     sae_path='cruft/ViT-3mil-topkk-8-experts-32_703f58/2969600.pt',
    #     n_activations=250_000,
    # )

    # main(
    #     sae_path='cruft/ViT-3mil-relu-l1-9e-05_f0477c/2969600.pt',
    #     n_activations=250_000,
    # )

    main(
        sae_path='cruft/ViT-3mil-topkk-32-experts-8_5d073c/2969600.pt',
        n_activations=250_000,
    )

    # main(
    #     sae_path='cruft/ViT-3mil-topkk-16-experts-8_62b60e/2969600.pt',
    #     n_activations=250_000,
    # )


    # main(
    #     sae_path='cruft/ViT-3mil-topkk-8-experts-8_fa5f99/2969600.pt',
    #     n_activations=250_000,
    # )


    # main(
    #     sae_path='cruft/ViT-3mil-topkk-64-experts-8_3e556f/2969600.pt',
    #     n_activations=250_000,
    # )



    

    
