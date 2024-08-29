import numpy as np
import torch

class SwitchSAE(torch.nn.Module):
    def __init__(self, n_features, n_experts, d_in, device, base_expert=False, pos_mask=False):
        super(SwitchSAE, self).__init__()

        if n_features % n_experts != 0:
            raise ValueError(f'N features {n_features} must be divisible by number of experts {n_experts}')

        self.n_experts = n_experts
        self.device=device
        self.base_expert = base_expert
        self.expert_dim = n_features // n_experts
        self.pre_b = torch.nn.Parameter(torch.randn(d_in, device=device) * 0.01)
        self.pos_mask = pos_mask

        self.enc = torch.nn.Parameter(torch.randn(self.n_experts, self.expert_dim, self.expert_dim, device=device) / (2**0.5) / (d_in ** 0.5))
        self.activation = torch.nn.ReLU()
        self.dec = torch.nn.Parameter(self.enc.mT.clone())

        self.router_b = torch.nn.Parameter(torch.randn(d_in, device=device) * 0.01)
        self.router = torch.nn.Parameter(torch.randn(d_in, n_experts, device=device) / (d_in ** 0.5))
        self.softmax = torch.nn.Softmax(dim=-1)

        if self.base_expert:
            self.base_enc = torch.nn.Parameter(torch.randn(d_in, self.expert_dim, device=device) / (2**0.5) / (d_in ** 0.5))
            self.base_dec = torch.nn.Parameter(self.base_enc.mT.clone())
            self.latent_dim = self.expert_dim * 2
        else:
            self.latent_dim = self.expert_dim


    def _encode(self, pre_activation):

        return self.activation(pre_activation)

    def _decode(self, latent, dec):
        return latent, latent @ dec # (n_to_expert, expert_dim), (n_to_expert, d_in)

    def forward_descriptive(self, activations, token_act): # activations: (batch_size, d_in)
        batch_size = activations.shape[0]
        # accumulators
        _full_recons = torch.zeros_like(activations) # (batch_size, d_in)
        _full_latent = torch.zeros((batch_size, self.latent_dim), device=self.device, requires_grad=False) # (batch_size, expert_dim)
        _was_active = torch.zeros(self.n_experts, self.latent_dim, device=self.device, requires_grad=False, dtype=bool) # (n_experts, expert_dim)
        
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

                if self.base_expert:
                    routed_enc = torch.concat([routed_enc, self.base_enc], dim=1)
                    routed_dec = torch.concat([self.base_dec, routed_dec], dim=0)
                
                latent = self._encode(expert_input @ routed_enc) # (n_to_expert, expert_dim)
                latent, reconstruction = self._decode(latent, routed_dec) # (n_to_expert, expert_dim), (n_to_expert, expert_dim)

                _full_latent[expert_mask] = latent
                _full_recons[expert_mask] = reconstruction
                with torch.no_grad():
                    _was_active[expert_id] = torch.max(latent, dim=0).values > 1e-3

        _full_recons = expert_max_prob.unsqueeze(-1) * _full_recons + self.pre_b # (batch_size, d_in)

        if token_act is not None:
            _full_recons = _full_recons + token_act

        return {
            'reconstruction': _full_recons, 
            'latent': _full_latent, 
            'active_latents': _was_active,
            'experts_chosen': expert_idx,
            'expert_prop': expert_prop,
            'expert_weighting': expert_weighting,
        }

def encode_topk(pre_activation, k):
    return torch.topk(pre_activation, k=k, dim=-1)

def eagre_decode(topk, dec):
    latent = torch.zeros((topk.values.shape[0], dec.shape[0]), dtype=dec.dtype, device=dec.device) # (n_to_expert, expert_dim)
    latent.scatter_(dim=-1, index=topk.indices, src=topk.values)

    return latent, latent @ dec

class TopKSwitchSAE(SwitchSAE):
    def __init__(self, k, *args, efficient=False, **kwargs):
        super(TopKSwitchSAE, self).__init__(*args, **kwargs)
        self.k = k
        self.efficient = efficient

        if self.efficient:
            from sache.kernel import triton_decode
            self._decode = triton_decode
            self.dec = torch.nn.Parameter(self.dec.mT) # requried for triton kernels
        else:
            self._decode = self._eagre_decode

    def _encode(self, pre_activation):
        return (encode_topk(pre_activation, self.k), pre_activation)

    def _eagre_decode(self, latent_info, dec):
        topk, pre_activation = latent_info
        return eagre_decode(topk, dec)

class SAE(torch.nn.Module):
    def __init__(self, n_features, d_in, device):
        super(SAE, self).__init__()

        self.pre_b = torch.nn.Parameter(torch.randn(d_in, device=device) * 0.01)
        self.enc = torch.nn.Parameter(torch.randn(d_in, n_features, device=device) / (2**0.5) / (d_in ** 0.5))
        self.dec = torch.nn.Parameter(self.enc.mT.clone())

        self.activation = torch.nn.ReLU()

    def _encode(self, x):
        return self.activation(x)
    
    def _decode(self, latent, dec):
        return latent, latent @ dec

    def forward_descriptive(self, x):

        latent = self._encode((x - self.pre_b) @ self.enc) # (n_to_expert, expert_dim)
        latent, reconstruction = self._decode(latent, self.dec)
        
        reconstruction = reconstruction + self.pre_b 

        return {
            'reconstruction': reconstruction,
            'latent': latent,
            'experts_chosen': None,
            'expert_prop': None,
            'expert_weighting': None,
            'active_latents': None,
        }

    def forward(self, x):
        recon = self.forward_descriptive(x)
        return recon

class TopKSAE(SAE):
    def __init__(self, k, *args, **kwargs):
        super(TopKSAE, self).__init__(*args, **kwargs)
        self.k = k
        self.activation = None

    def _encode(self, x):
        return encode_topk(x, self.k)

    def _decode(self, topk, dec):
        return eagre_decode(topk, dec)
        