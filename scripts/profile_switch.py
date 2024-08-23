import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sache.train import TopKSwitchSAE
device='cuda'
n_trials = 10
es = [16, 32, 64, 128]
n_feats=24576
d_in = 768

bss = [8192*4, 8192*16, 8192 * 32]
largest_bs = max(bss)
results = defaultdict(list)
for bs in bss:
    for n_experts in es:
        print(bs, n_experts)
        sae = TopKSwitchSAE(k=32, n_features=n_feats, n_experts=n_experts, d_in=d_in, device=device, efficient=False)

        with torch.no_grad():
            batch = torch.empty(bs, d_in, device=device)
            sae.forward_descriptive(batch)

        config_results = []

        optimizer = torch.optim.Adam(sae.parameters(), lr=1e-4)

        for i in range(n_trials):
            outer_batch = torch.empty(largest_bs, d_in, device='cpu')
            torch.cuda.synchronize()
            start = time.time()
            
            for j in range(0, largest_bs, bs):
                batch = outer_batch[j:j+bs].detach().to(device)
                optimizer.zero_grad()

                output = sae.forward_descriptive(batch)

                mse = ((batch - output['reconstruction']) ** 2).sum(-1).mean()
                mean_pred_mse = ((batch - batch.mean(0)) ** 2).sum(-1).mean()


                loss = mse / mean_pred_mse

                loss.backward()
                optimizer.step()

                torch.cuda.synchronize()
            
            end = time.time()
            config_results.append((largest_bs / 350) / (end-start))

        results[str(n_experts)].append(config_results)

for k in results.keys():
    rk = results[k]
    means = [round(np.mean(trials).item(), 7) for trials in rk]
    stds = [round(np.std(trials).item(), 7) for trials in rk]

    plt.errorbar(bss, means, yerr=stds, fmt='-o', label=f'{k} Experts')
    plt.legend()
    plt.xlabel('Batch size')
    plt.ylabel('MB/s')
    plt.title('MB/s vs Batch Size on g4dn.xlarge')

    if not os.path.exists('cruft'):
        os.makedirs('cruft')

    plt.savefig('cruft/batch_size_performance.png')

    print('n_experts', k)

    for i in range(len(means)):
        print(means[i], stds[i])