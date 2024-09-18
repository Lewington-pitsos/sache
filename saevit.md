# plan


Laion-5B -> still has the faces in, but no porn/harm https://arxiv.org/pdf/2210.08402

https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K trained on the LAION-2B english dataset. We will use this, only 75% acc on imagenet, but like imagenet is for nerds anyway, it probably does not map to good ability to represent the world.

weirdly hugo only used the cls token from the end of the CLIP thingo and doesn't give an excellent justification. Let's try all 256 at some point.

- profile why the generation is 4x as slow as it ought to be

## Random notes




Check why the other guy used Vit-BigG - He gives no reason

Comparr vit-bigG and vit-L/14, see what the difference in model size actually is

commonpool 12.8m, explained here: https://arxiv.org/pdf/2304.14108 , turns out it has all faces blurred :(, fucking LAME

