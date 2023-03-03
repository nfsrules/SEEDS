# SEEDS: A Family of Exponencial SDE Solvers for Fast-High Quality Sampling from Diffusion models
### Official PyTorch implementation 

<p float="left">
  <img src="/assets/grid_celeba.png" width="300" />
  <img src="/assets/grid_cifar10.png" width="300" /> 
</p>

Abstract: *We introduce SEEDS, a family of training-free exponential SDE solvers for Diffusion Probabilistic Models (DPMs) with guaranteed convergence. Unlike other stochastic solvers, SEEDS optimally exploits the linear analytic computation on semi-linear SDEs appearing in DPMs while preserving thestochasticity of the solver, reducing the number of function evaluations (NFEs) needed to sample high-quality images. Moreover, the formulation of the analytic framework, from which we derive SEEDS, is flexible enough to take into account many choices in the design space of DPMs. SEEDS is suitable for discretely and continuously trained DPMs, in the VP, VE, iDDPM, and EDM regimes, without the need for further training or expensive parameter optimizations. Experiments show that one of our SEEDS solvers achieve FID scores of 1.93 and 2.24 on CelebA (VP uncond.) and CIFAR-10 (VP cond.) respectively, outperforming previous SDE and ODE samplers. Finally, different SEEDS solvers offer competitive performances on various datasets, considerablyfaster than other available stochastic methods*



## License

Part of the code was derived from Karras and Al [Elucidating the Design Space of Diffusion-Based Generative]([http://creativecommons.org/licenses/by-nc-sa/4.0/](https://github.com/NVlabs/edm)

All material, including source code and pre-trained models, is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

`baseline-cifar10-32x32-uncond-vp.pkl` and `baseline-cifar10-32x32-uncond-ve.pkl` are derived from the [pre-trained models](https://github.com/yang-song/score_sde_pytorch) by Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. The models were originally shared under the [Apache 2.0 license](https://github.com/yang-song/score_sde_pytorch/blob/main/LICENSE).

`baseline-imagenet-64x64-cond-adm.pkl` is derived from the [pre-trained model](https://github.com/openai/guided-diffusion) by Prafulla Dhariwal and Alex Nichol. The model was originally shared under the [MIT license](https://github.com/openai/guided-diffusion/blob/main/LICENSE).

`imagenet-64x64-baseline.npz` is derived from the [precomputed reference statistics](https://github.com/openai/guided-diffusion/tree/main/evaluations) by Prafulla Dhariwal and Alex Nichol. The statistics were
originally shared under the [MIT license](https://github.com/openai/guided-diffusion/blob/main/LICENSE).
