import torch
import numpy as np

from sache.log import ProcessLogger
from sache.model import SAE, SwitchSAE, TopKSwitchSAE, TopKSAE

def get_histogram(tensor, bins=50):
    tensor = tensor.detach()
    hist = torch.histc(tensor, bins=bins, min=float(tensor.min()), max=float(tensor.max()))

    bin_edges = np.linspace(float(tensor.min()), float(tensor.max()), bins+1)

    hist_list = hist.tolist()
    bin_edges_list = bin_edges.tolist()

    return hist_list, bin_edges_list

class TrainLogger(ProcessLogger):
    def __init__(self, run_name, log_mean_std=False, max_sample=1024, *args, **kwargs):
        super(TrainLogger, self).__init__(run_name, *args, **kwargs)
        self.log_mean_std = log_mean_std
        self.max_sample = max_sample

    def log_sae(self, sae, info=None):
        if isinstance(sae, SAE):
            message = self._log_sae(sae)
        elif isinstance(sae, TopKSAE):
            message = self._log_sae(sae)
        elif isinstance(sae, TopKSwitchSAE):
            message = self._log_switch_sae(sae)
        elif isinstance(sae, SwitchSAE):
            message = self._log_switch_sae(sae)
        else:
            raise ValueError(f'Unknown SAE type {type(sae)}')

        if info is not None:
            for k in info.keys():
                if k in message:
                    raise ValueError(f'Key {k} already exists in message', message, info)
            message.update(info)
        
        self.log(message)

    def _log_switch_sae(self, sae, info=None):
        with torch.no_grad():
            ecounts, eedges = get_histogram(sae.enc)
            dcounts, dedges = get_histogram(sae.dec)
            routercounts, routeredges = get_histogram(sae.router)
            broutercounts, brouteredges = get_histogram(sae.router_b, bins=25)
            bprecounts, bpreedges = get_histogram(sae.pre_b, bins=25)
            
        return {
            'event': 'sae',
            'enc_experts': { 
                'counts': ecounts,
                'edges': eedges
            },
            'pre_b': {
                'counts': bprecounts,
                'edges': bpreedges
            },
            'dec_experts': {
                'counts': dcounts,
                'edges': dedges
            },
            'router_b': {
                'counts': broutercounts,
                'edges': brouteredges
            },
            'router': {
                'counts': routercounts,
                'edges': routeredges
            }
        }

    def _log_sae(self, sae, info=None):
        with torch.no_grad():
            ecounts, eedges = get_histogram(sae.enc)
            ebcounts, ebedges = get_histogram(sae.pre_b, bins=25)
            dcounts, dedges = get_histogram(sae.dec)
            
        return {
            'event': 'sae',
            'enc': { 
                'counts': ecounts,
                'edges': eedges
            },
            'enc_b': {
                'counts': ebcounts,
                'edges': ebedges
            },
            'dec': {
                'counts': dcounts,
                'edges': dedges
            },
        }

    def log_loss(self, mse, sum_mse, l1, loss, batch, latent, dead_pct, 
                expert_privilege, lr, position_mse, explained_variance, 
                variance_prop_mse, massive_activations):
        with torch.no_grad():
            message = {
                'event': 'training_batch', 
                'mse': mse.item(),
                'sum_mse': sum_mse.item(),
                'loss': loss.item(),
                'batch_learning_rate': lr
            }

            if variance_prop_mse is not None:
                message['variance_proportional_mse'] = variance_prop_mse.item()

            if explained_variance is not None:
                message['explained_variance'] = explained_variance.item()

            if massive_activations is not None:
                message['massive_activations'] = massive_activations.cpu().numpy().tolist()

            if position_mse is not None:
                message['position_mse'] = position_mse.cpu().numpy().tolist()

            if latent is not None:
                message['L0'] = (latent > 0).float().sum(-1).mean().item()

            if dead_pct is not None:
                message['dead_feature_prop'] = dead_pct.item()

            if expert_privilege is not None:
                message['expert_privilege'] = expert_privilege.item()

            if l1 is not None:
                message['l1'] = l1.item()

            if self.log_mean_std:
                message.update({
                    'input_mean': batch.mean(dim=(0, 1)).cpu().numpy().tolist(), 
                    'input_std': batch.std(dim=(0, 1)).cpu().numpy().tolist()
                })

            self.log(message)

    def log_batch(self, sae, batch, reconstruction, latent, experts_chosen):
        batch = batch[:self.max_sample]
        reconstruction = reconstruction[:self.max_sample]
        latent = latent[:self.max_sample]

        with torch.no_grad():
            binput, einput = get_histogram(batch)
            brecon, erecon = get_histogram(reconstruction)
            bdelta, edelta = get_histogram(batch - reconstruction)
            blatent, elatent = get_histogram(latent)

            bencgrad, eencgrad = get_histogram(sae.enc.grad)
            bdecgrad, edecgrad = get_histogram(sae.dec.grad)
            bpregrad, epregrad = get_histogram(sae.pre_b.grad)
            bdec, edec = get_histogram(sae.dec)


            info = {
                'input_hist': { 'counts': binput, 'edges': einput},
                'reconstruction_hist': { 'counts': brecon, 'edges': erecon},
                'delta_hist': { 'counts': bdelta, 'edges': edelta},
                'latent_hist': { 'counts': blatent, 'edges': elatent},
                
                'dec_hist': { 'counts': bdec, 'edges': edec},

                'enc_grad_hist': { 'counts': bencgrad, 'edges': eencgrad},
                'dec_grad_hist': { 'counts': bdecgrad, 'edges': edecgrad},
                'pre_grad_hist': { 'counts': bpregrad, 'edges': epregrad}
            }

            if experts_chosen is not None:
                experts_chosen = experts_chosen[:self.max_sample]
                bexperts, eexperts = get_histogram(experts_chosen, bins=sae.n_experts)

                info['experts_chosen_hist'] = { 'counts': bexperts, 'edges': eexperts}

            if hasattr(sae, 'k'):
                info['k'] = sae.k

            if hasattr(sae, 'router'):
                broutergrad, eroutergrad = get_histogram(sae.router.grad)
                info['router_grad_hist'] = { 'counts': broutergrad, 'edges': eroutergrad}

        self.log_sae(sae, info=info)