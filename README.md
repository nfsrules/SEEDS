# SEEDS: Exponential SDE Solvers for Fast High-Quality Sampling from Diffusion Models
## Neurips 2023 accepted paper [[arXiv Link](https://arxiv.org/abs/2305.14267)]
### Official PyTorch implementation 

<p float="left">
  <img src="/assets/grid_bedroom.png" width="240" />
  <img src="/assets/grid_celeba.png" width="240" />
  <img src="/assets/grid_cifar10.png" width="240" /> 
</p>


Martin Gonzalez, Nelson Fernandez, Thuy Tran, Elies Gherbi, Hatem Hajri, Nader Masmoudi.

Abstract: *A potent class of generative models known as Diffusion Probabilistic Models (DPMs) has become prominent. A forward diffusion process adds gradually noise to data, while a model learns to gradually denoise. Sampling from pre-trained DPMs is obtained by solving differential equations (DE) defined by the learnt model, a process which has shown to be prohibitively slow. Numerous efforts on speeding-up this process have consisted on crafting powerful ODE solvers. Despite being quick, such solvers do not usually reach the optimal quality achieved by available slow SDE solvers. Our goal is to propose SDE solvers that reach optimal quality without requiring several hundreds or thousands of NFEs to achieve that goal. In this work, we propose Stochastic Exponential Derivative-free Solvers (SEEDS), improving and generalizing Exponential Integrator approaches to the stochastic case on several frameworks. After carefully analyzing the formulation of exact solutions of diffusion SDEs, we craft SEEDS to analytically compute the linear part of such solutions. Inspired by the Exponential Time-Differencing method, SEEDS uses a novel treatment of the stochastic components of solutions, enabling the analytical computation of their variance, and contains high-order terms allowing to reach optimal quality sampling ∼3-5× faster than previous SDE methods. We validate our approach on several image generation benchmarks, showing that SEEDS outperforms or is competitive with previous SDE solvers. Contrary to the latter, SEEDS are derivative and training free, and we fully prove strong convergence guarantees for them.*


## Requirements
* We recommend running on Linux for performance and compatibility reasons. 
* 1+ GPU should be used for sampling. We have done all testing and development using Tesla V100S GPUs.
* 64-bit Python 3.9 and PyTorch 1.13.0 (or later) which includes the function `torch.torch.distributed.all_gather_into_tensor()` required to compute the **Inception score**. See https://pytorch.org for PyTorch install instructions.
* Python libraries: The required libraries are described in the file [requirements.txt](./requirements.txt). You can easily install using the following `pip` command:
```.bash
pip install -r requirements
```

## Getting started
To reproduce the main results of our article, you can simply run:
```.bash
sh curves.sh
```
After successfully running, the directory [output](./output) contains 100 images sampling from [Flickr-Faces-HQ dataset](https://github.com/NVlabs/ffhq-dataset) of shape 64x64 in the **variance preserving (VP)** setting. You can explore more results by changing the **schedule**, **scaling**, **discretization** scheme or **preconditioning** params. Moreover, you can also choose the solvers as *ETD-ERK* or *ETD-SERK* in low or high orders (`order=1,2,3,4`) such that we have a trade-off between the speed and the image quality.

## License

Part of the code was derived from Karras and Al. [Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://github.com/NVlabs/edm) under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

All material provided in this repository, including source code and pre-trained models, is therefore licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

`baseline-cifar10-32x32-uncond-vp.pkl` and `baseline-cifar10-32x32-uncond-ve.pkl` are derived from the [pre-trained models](https://github.com/yang-song/score_sde_pytorch) by Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. The models were originally shared under the [Apache 2.0 license](https://github.com/yang-song/score_sde_pytorch/blob/main/LICENSE).

`baseline-imagenet-64x64-cond-adm.pkl` is derived from the [pre-trained model](https://github.com/openai/guided-diffusion) by Prafulla Dhariwal and Alex Nichol. The model was originally shared under the [MIT license](https://github.com/openai/guided-diffusion/blob/main/LICENSE).

`imagenet-64x64-baseline.npz` is derived from the [precomputed reference statistics](https://github.com/openai/guided-diffusion/tree/main/evaluations) by Prafulla Dhariwal and Alex Nichol. The statistics were
originally shared under the [MIT license](https://github.com/openai/guided-diffusion/blob/main/LICENSE).
