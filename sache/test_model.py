import torch
from sache.model import TopKSwitchSAE

def test_eagre_and_triton_decode_get_same_results():
    n_experts = 4
    d_in = 32
    k = 4
    batch_size = 16
    n_features = 64


    triton_sae = TopKSwitchSAE(k=k, n_features=n_features, n_experts=n_experts, d_in=d_in, device='cpu', efficient=True)
    triton_sae.eval()
    
    with torch.no_grad():
        pre_act = torch.randn(batch_size, n_features // n_experts)
        topk = torch.topk(pre_act, k=k, dim=-1)

        dec = triton_sae.dec[0]
        eagre_latent, eagre_reconstruction = triton_sae._eagre_decode(topk, dec)
        triton_latent, triton_reconstruction = triton_sae._triton_decode(topk, dec)

    assert torch.allclose(eagre_reconstruction, triton_reconstruction)
    assert torch.allclose(eagre_latent, triton_latent)