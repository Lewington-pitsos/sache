import matplotlib.pyplot as plt

def vanilla_ram(ft, d_in):
    return 2 * ft * d_in + d_in

def switch_ram(ft, d_in, n):
    return 2* ft * d_in + n*d_in + 2 * d_in


ft_values = [768, 1024, 24576]
d_in = [768]
n_values = [16, 32, 64]

vanilla_rams = []
switch_rams = []

for ft in ft_values:
    for input_dim in d_in:
        for n in n_values:
            vanilla_rams.append(vanilla_ram(ft, input_dim) / 1024 / 1024 * 4)
            switch_rams.append(switch_ram(ft, input_dim, n) / 1024 / 1024 * 4)

plt.plot(vanilla_rams, label='Vanilla')
plt.plot(switch_rams, label='Switch')

def vanilla_params(ft, d_in, k):
    return ft * d_in + k * d_in + d_in

def switch_params(ft, d_in, k, n):
    return (ft / n) * d_in + n * d_in  + k * d_in + d_in * 2


ft_values = [1024, 24576]
d_in = [768]
k_values = [32]
n_values = [4, 8, 16]


switch_totals = []
vanilla_totals = []
for ft in ft_values:
    for input_dim in d_in:
        for k in k_values:
            for n in n_values:
                vanilla = vanilla_params(ft, input_dim, k)
                switch = switch_params(ft, input_dim, k, n)
                vanilla_totals.append(vanilla)
                switch_totals.append(switch)

plt.plot(vanilla_totals, label='Vanilla')
plt.plot(switch_totals, label='Switch')
