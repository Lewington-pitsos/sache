import torch

input = torch.tensor([
    [1, 2],
    [1, 2],
    [1, 2],
]) # (bs, input_dim)


enc_expert_1 = torch.tensor([
    [1, 1, 1, 1],
    [1, 1, 1, 1],

])
enc_expert_2 = torch.tensor([
    [3, 3, 0, 0],
    [0, 0, 2, 0],
])



dec_expert_1 = torch.tensor([
    [ -1, -1],
    [ -1, -1],
    [ -1, -1],
    [ -1, -1],
])

dec_expert_2 = torch.tensor([
    [-10, -10,],
    [-10, -10,],
    [-10, -10,],
    [-10, -10,],

])

def moe(input, enc, dec, nonlinearity):
    input = input.unsqueeze(1)
    latent = torch.bmm(input, enc)

    recon = nonlinearity(torch.bmm(latent, dec))

    return recon.squeeze(1), latent.squeeze(1)


# not this! some kind of actual routing algorithm, but you end up with something similar
enc = torch.stack([enc_expert_2, enc_expert_1, enc_expert_2])
dec = torch.stack([dec_expert_1, dec_expert_2, dec_expert_1])

nonlinearity = torch.nn.ReLU()
recons, latent = moe(input, enc, dec, nonlinearity)