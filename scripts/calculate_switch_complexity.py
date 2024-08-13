import matplotlib.pyplot as plt

def vanilla_ram(ft, hidden):
    return 2 * ft * hidden + hidden

def switch_ram(ft, hidden, n):
    return 2* ft * hidden + n*hidden + 2 * hidden


ft_values = [768, 1024, 24576]
hidden_values = [768]
n_values = [16, 32, 64]

vanilla_rams = []
switch_rams = []

for ft in ft_values:
    for hidden in hidden_values:
        for n in n_values:
            vanilla_rams.append(vanilla_ram(ft, hidden) / 1024 / 1024 * 4)
            switch_rams.append(switch_ram(ft, hidden, n) / 1024 / 1024 * 4)

plt.plot(vanilla_rams, label='Vanilla')
plt.plot(switch_rams, label='Switch')

def vanilla_params(ft, hidden, k):
    return ft * hidden + k * hidden + hidden

def switch_params(ft, hidden, k, n):
    return (ft / n) * hidden + n * hidden  + k * hidden + hidden * 2


ft_values = [1024, 24576]
hidden_values = [768]
k_values = [32]
n_values = [4, 8, 16]


switch_totals = []
vanilla_totals = []
for ft in ft_values:
    for hidden in hidden_values:
        for k in k_values:
            for n in n_values:
                vanilla = vanilla_params(ft, hidden, k)
                switch = switch_params(ft, hidden, k, n)
                vanilla_totals.append(vanilla)
                switch_totals.append(switch)

plt.plot(vanilla_totals, label='Vanilla')
plt.plot(switch_totals, label='Switch')
