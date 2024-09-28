from sorteval import construct_prompt
from transformers import GPT2Tokenizer

p = construct_prompt(
    'cruft/ViT-3mil-topkk-32-experts-None_1aaa89/latents-2969600/images_old/feature_0/0_grid.png',
    'cruft/ViT-3mil-topkk-32-experts-None_1aaa89/latents-2969600/images_old/feature_5/5_grid.png',
    'cruft/ViT-3mil-topkk-32-experts-None_1aaa89/latents-2969600/images_old/feature_0/0_top9_1.png'
)

txt_count = 0
img_count = 0
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
for a in p:
    print(a.keys())

    if 'text' in a:
        token_count = len(tokenizer.encode(a['text']))
        txt_count += token_count


print(txt_count, img_count)



print(len(tokenizer.encode("""To determine which neuron is more likely to be activated by the given example, we need to analyze the concepts represented by each neuron based on the provided images.\n\nNeuron 1 Examples:\n- Power tools (Makita drills, grinders)\n- Software (Kaspersky Internet Security)\n- Jewelry (Tiffany & Co. necklace)\n- Racing car\n\nNeuron 2 Examples:\n- Radio logos\n- Hotel rooms\n- Food (pie crusts)\n- News broadcast\n- Political scene\n\nThe specific example image provided shows a racing car in a pit stop. This is similar to the racing car image in Neuron 1's examples.\n\nGiven this analysis, the specific example is more closely related to the concept represented by Neuron 1.\n\nANSWER: 1",
                           """)))