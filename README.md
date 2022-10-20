# Vision Model Implementations
This repository contains a growing list of computer vision models implemented in PyTorch. I am mostly focusing on recent-ish models, especially those based on attention. My goal is to write my own implementation based on the paper, but also learn tricks and check my work by referencing existing implementations.

# Models
* __MLP Mixer:__ MLP-based architecture based on a [paper from Google AI Research](https://arxiv.org/abs/2105.01601), which shows that you don't _have_ to use convolutions or attention to get good performance on computer vision tasks.
* __Vision Transformer:__ Attention-based architecture, adapted from the paper "[Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)". The authors split images into patches, and feed the resulting sequence into a transformer encoder.
* __Masked Autoencoder Vision Transformer:__ Using the Vision Transformer architecture above, the authors of the paper "[Masked Autoencoders Are Scalable Vision Learners](Masked Autoencoders Are Scalable Vision Learners)" adopt a self-supervised pretraining objective similar to masked language modeling ([Devlin et al., 2018](https://arxiv.org/abs/1810.04805)), where image patches are masked before they are fed into a transformer encoder, and a lightweight decoder must reconstruct them.

# Data & Training
My initial tests for these models are still in progress, involve training them on CIFAR-100. I will release code and results of some of these experiments soon.

# References

### Papers Implemented
* __MLP Mixer:__ [Tolstikhin, I. O., Houlsby, N., Kolesnikov, A., Beyer, L., Zhai, X., Unterthiner, T., ... & Dosovitskiy, A. (2021). MLP-Mixer: An All-MLP Architecture for Vision. Advances in Neural Information Processing Systems, 34, 24261-24272.](https://arxiv.org/abs/2105.01601)
* __Vision Transformer:__ [Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.](https://arxiv.org/abs/2010.11929)
* __Masked Autoencoder:__ [He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 16000-16009)](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf)

### Code References
* Andrej Karpathy's [`mingpt`](https://github.com/karpathy/minGPT): Referenced for some tricks related to implementation of multi-head attention.
* [Einops Documentation](https://einops.rocks/pytorch-examples.html): Referenced for more tricks related to multi-head attention, namely, Einstein notation.
* Phil Wang's [ViT repository](https://github.com/lucidrains/vit-pytorch): Referenced for Vision Transformer and masked autoencoder, and subsequent proposed modifications/improvements in the literature, including replacing CLS token with average pooling after the final transformer block. I borrowed the elegant approach of wrapping the attention and FFN blocks in a "PreNorm" layer that handles normalization, tweaking it slightly to include the residual connection. This results in a much cleaner transformer block implementation.
* Google Research [MLP Mixer & ViT implementations](https://github.com/google-research/vision_transformer): Referenced for my MLP Mixer and Vision Transformer implementations.
* Facebook Research [MAE implementation](https://github.com/facebookresearch/mae): Referenced for my Masked Autoencoder implementation.