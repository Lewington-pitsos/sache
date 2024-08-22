import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from collections import defaultdict

from sache.train import SwitchSAE
device='cuda'
n_trials = 10
es = [16, 32, 64, 128]
n_feats=24576
d_in = 768

bss = [8192, 8192*4, 8192*16, 8192 * 32, 8192*64, 8192*128]
results = defaultdict(list)
for bs in bss:
    for n_experts in es:
        switch_sae = SwitchSAE(n_features=n_feats, n_experts=n_experts, d_in=d_in, device=device)

        # warm up
        input = torch.empty(bs, d_in, device=device)
        switch_sae.forward_descriptive(input)

        config_results = []

        for i in range(n_trials):
            torch.cuda.synchronize()
            start = time.time()

            input = torch.empty(bs, d_in, device=device) * i
            switch_sae.forward_descriptive(input)

            torch.cuda.synchronize()
            end = time.time()

            config_results.append(end-start)

        results[str(n_experts)].append(config_results)

for k in results.keys():
    rk = results[k]
    means = [round(np.mean(trials).item(), 2) for trials in rk]
    stds = [round(np.std(trials).item(), 2) for trials in rk]

    plt.errorbar(bss, means, yerr=stds, fmt='-o', label=k)
    plt.legend()
    plt.xlabel('Batch size')
    plt.ylabel('Time (s)')
    plt.title('Performance by Batch Size')

    print('k')

    for i in range(len(means)):
        print(means[i], stds[i])

