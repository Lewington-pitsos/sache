import torch
from sache.model import TopKSwitchSAE

def test_eagre_and_triton_decode_get_same_results():
    n_experts = 4
    d_in = 32
    k = 4
    batch_size = 8
    n_features = 64
    device='cuda'

    triton_sae = TopKSwitchSAE(k=k, n_features=n_features, n_experts=n_experts, d_in=d_in, device=device, efficient=True)
    triton_sae.eval()
    
    with torch.no_grad():
        pre_act = torch.randn(batch_size, n_features // n_experts, device=device)
        topk = torch.topk(pre_act, k=k, dim=-1)
        input = (topk, pre_act)

        dec = triton_sae.dec[2]
        eagre_latent, eagre_reconstruction = triton_sae._eagre_decode(input, dec)
        triton_latent, triton_reconstruction = triton_sae._triton_decode(input, dec.T)

    assert torch.allclose(eagre_reconstruction, triton_reconstruction)