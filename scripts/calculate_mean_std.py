print(len(info['input_mean']))
print(len(info['input_std']))

for key in ['mean', 'std']:
    tensor = torch.mean(torch.tensor(info['input_' + key]), dim=0)
    print('shape', key, tensor.shape)
    print(torch.topk(tensor, 10))
    print(torch.topk(tensor, 10, largest=False))

    torch.save(tensor, 'normalize/merciless-citadel/' + key + '.pt')
