# SEEDS: Exponential SDE Solvers for Fast High-Quality Sampling from Diffusion Models
### The official code of [SEEDS: Exponential SDE Solvers for Fast High-Quality Sampling from Diffusion Models](https://arxiv.org/abs/2305.14267) (Neurips 2023).

<figure style="text-align: center;">
  <img src="/assets/seeds-stable-diffusion-xl.png" width="612" alt="Image Description" />
  <figcaption>Image sampled with SEEDS2 from Stable Diffusion XL at 200 NFE.</figcaption>
</figure>


Martin Gonzalez, Nelson Fernandez, Thuy Tran, Elies Gherbi, Hatem Hajri, Nader Masmoudi.

## Overview
We address the challenges of slow SDE sampling in Diffusion Probabilistic Models (DPMs) by introducing Stochastic Exponential Derivative-free Solvers (SEEDS). SEEDS are designed to provide optimal quality sampling without the need for a large number of evaluations (NFEs). Previous efforts have focused on improving speed by crafting powerful ODE solvers, but they often fall short of achieving the optimal quality obtained by slower SDE solvers. We accomplish this by analytically computing the linear part of solutions in diffusion SDEs and incorporating innovative techniques for handling stochastic components. Inspired by the Stochastic Exponential Time-Differencing method, SEEDS include Markov preserving high-order terms that significantly accelerate the sampling process. Importantly, SEEDS do not require derivatives or training. This research provides the first set of scalable and efficient solvers with fully proven strong & weak convergence guarantees. We validate our approach on multiple image generation benchmarks, demonstrating that SEEDS is about 3-5 times faster than previous SDE methods while either outperforming or remaining competitive with their quality.


## Requirements
* We recommend running on Linux for performance and compatibility reasons. 
* 1+ GPU should be used for sampling. We have done all testing and development using Tesla V100S GPUs.
* 64-bit Python 3.9 and PyTorch 1.13.0 (or later) which includes the function `torch.torch.distributed.all_gather_into_tensor()` required to compute the **Inception score**. See https://pytorch.org for PyTorch install instructions.
* Python libraries: The required libraries are described in the file [requirements.txt](./requirements.txt). You can easily install using the following `pip` command:
```.bash
pip install -r requirements
```

## More details coming soon!
Stay tuned for more useful details on how to run this repository.


## License

The code is heavily derived from Karras and Al. [Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://github.com/NVlabs/edm) under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).


## Citation
If you find our work useful, please consider citing:

```
@inproceedings{Gonzalez2023seeds,
  author    = {Martin Gonzalez, Nelson Fernandez Pinto, Thuy Tran, Elies Gherbi, Hatem Hajri, Nader Masmoudi},
  title     = {SEEDS: Exponential SDE Solvers for Fast High-Quality Sampling from Diffusion Models},
  booktitle = {Proc. NeurIPS},
  year      = {2023}
}
```

