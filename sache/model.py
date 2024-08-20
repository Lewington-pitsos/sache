import numpy as np
import torch

from sache.kernel import TritonDecoderAutograd

class SwitchSAE(torch.nn.Module):
    def __init__(self, n_features, n_experts, d_in, device):
        super(SwitchSAE, self).__init__()

        if n_features % n_experts != 0:
            raise ValueError(f'N features {n_features} must be divisible by number of experts {n_experts}')

        self.n_experts = n_experts
        self.device=device
        self.expert_dim = n_features // n_experts
        self.pre_b = torch.nn.Parameter(torch.randn(d_in, device=device) * 0.01)

        self.enc = torch.nn.Parameter(
            torch.randn(n_experts, d_in, self.expert_dim, device=device) / (2**0.5) / (d_in ** 0.5)
        )
        self.activation = torch.nn.ReLU()
        self.dec = torch.nn.Parameter(self.enc.mT.clone())

        self.router_b = torch.nn.Parameter(torch.randn(d_in, device=device) * 0.01)
        self.router = torch.nn.Parameter(torch.randn(d_in, n_experts, device=device) / (d_in ** 0.5))
        self.softmax = torch.nn.Softmax(dim=-1)

    def _encode(self, pre_activation):
        return self.activation(pre_activation)

    def _decode(self, latent, dec):
        return latent, latent @ dec # (n_to_expert, expert_dim), (n_to_expert, d_in)

    def forward_descriptive(self, activations): # activations: (batch_size, d_in)
        batch_size = activations.shape[0]
        # accumulators
        _full_recons = torch.zeros_like(activations) # (batch_size, d_in)
        _full_latent = torch.zeros((activations.size(0), self.expert_dim), device=self.device, requires_grad=False) # (batch_size, expert_dim)
        _was_active = torch.zeros(self.n_experts, self.expert_dim, device=self.device, requires_grad=False, dtype=bool) # (n_experts, expert_dim)
        
        expert_probabilities = self.softmax((activations - self.router_b) @ self.router) #  (batch_size, n_experts)
        expert_max_prob, expert_idx = torch.max(expert_probabilities, dim=-1) # (batch_size,), (batch_size,)

        expert_prop = torch.bincount(expert_idx, minlength=self.n_experts) / batch_size # (n_experts,)
        expert_weighting = torch.mean(expert_probabilities, dim=0) # (n_experts,)

        for expert_id in range(self.n_experts):
            if expert_id in expert_idx:
                expert_mask = expert_idx == expert_id # (n_to_expert,)
                expert_input = activations[expert_mask] - self.pre_b # (n_to_expert, d_in)

                routed_enc = self.enc[expert_id] # (d_in, expert_dim)
                routed_dec = self.dec[expert_id] # (expert_dim, d_in)
                latent = self._encode(expert_input @ routed_enc) # (n_to_expert, expert_dim)
                latent, reconstruction = self._decode(latent, routed_dec) # (n_to_expert, expert_dim), (n_to_expert, d_in)

                _full_latent[expert_mask] = latent
                _full_recons[expert_mask] = reconstruction 
                with torch.no_grad():
                    _was_active[expert_id] = torch.max(latent, dim=0).values > 1e-3

        _full_recons = expert_max_prob.unsqueeze(-1) * _full_recons + self.pre_b # (batch_size, d_in)

        return {
            'reconstruction': _full_recons, 
            'latent': _full_latent, 
            'active_latents': _was_active,
            'experts_chosen': expert_idx,
            'expert_prop': expert_prop,
            'expert_weighting': expert_weighting,
        }
              # (batch_size, d_in), (batch_size, expert_dim), (n_experts, expert_dim)

class TopKSwitchSAE(SwitchSAE):
    def __init__(self, k, *args, efficient=False, **kwargs):
        super(TopKSwitchSAE, self).__init__(*args, **kwargs)
        self.k = k
        self.efficient = efficient

        if self.efficient:
            self._decode = self._triton_decode
            self.dec = torch.nn.Parameter(self.dec.contiguous().mT) # requried for triton kernels
        else:
            self._decode = self._eagre_decode

    def _encode(self, pre_activation):
        return (torch.topk(pre_activation, k=self.k, dim=-1), pre_activation)

    def _triton_decode(self, latent_info, dec):
        topk, pre_activation = latent_info
        
        recons = TritonDecoderAutograd.apply(topk.indices, topk.values, dec)

        return pre_activation, recons

    def _eagre_decode(self, latent_info, dec):
        topk, pre_activation = latent_info

        latent = torch.zeros((topk.values.shape[0], dec.shape[0]), dtype=dec.dtype, device=dec.device) # (n_to_expert, expert_dim)
        latent.scatter_(dim=-1, index=topk.indices, src=topk.values)

        return latent, latent @ dec

class SAE(torch.nn.Module):
    def __init__(self, n_features, d_in, device):
        super(SAE, self).__init__()
        self.enc = torch.nn.Parameter(torch.randn(d_in, n_features, device=device) / np.sqrt(n_features))
        self.enc_b = torch.nn.Parameter(torch.randn(n_features, device=device) * 0.01)
        self.dec = torch.nn.Parameter(torch.randn(n_features, d_in, device=device) / np.sqrt(d_in))
        self.dec_b = torch.nn.Parameter(torch.zeros(d_in, dtype=torch.float32, device=device))
        self.activation = torch.nn.ReLU()

    def forward_descriptive(self, x):
        features = self.activation(x @ self.enc + self.enc_b) 
        return features @ self.dec + self.dec_b, features

    def forward(self, x):
        recon, _ = self.forward_descriptive(x)
        return recon
