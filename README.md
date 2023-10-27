# SEEDS: Exponential SDE Solvers for Fast High-Quality Sampling from Diffusion Models
### The official code of [SEEDS: Exponential SDE Solvers for Fast High-Quality Sampling from Diffusion Models](https://arxiv.org/abs/2305.14267) (Neurips 2023).

<figure style="text-align: center;">
  <img src="/assets/seeds-stable-diffusion-xl.png" width="612" alt="Image Description" />
  <figcaption>Image sampled with SEEDS2 from Stable Diffusion XL at 200 NFE.</figcaption>
</figure>


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

