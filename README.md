# SEEDS: A Family of Exponencial SDE Solvers for Fast-High Quality Sampling from Diffusion models
### Official PyTorch implementation 

<p float="left">
  <img src="/assets/grid_celeba.png" width="300" />
  <img src="/assets/grid_cifar10.png" width="300" /> 
</p>

Abstract: *We introduce SEEDS, a family of training-free exponential SDE solvers for Diffusion Probabilistic Models (DPMs) with guaranteed convergence. Unlike other stochastic solvers, SEEDS optimally exploits the linear analytic computation on semi-linear SDEs appearing in DPMs while preserving thestochasticity of the solver, reducing the number of function evaluations (NFEs) needed to sample high-quality images. Moreover, the formulation of the analytic framework, from which we derive SEEDS, is flexible enough to take into account many choices in the design space of DPMs. SEEDS is suitable for discretely and continuously trained DPMs, in the VP, VE, iDDPM, and EDM regimes, without the need for further training or expensive parameter optimizations. Experiments show that one of our SEEDS solvers achieve FID scores of 1.93 and 2.24 on CelebA (VP uncond.) and CIFAR-10 (VP cond.) respectively, outperforming previous SDE and ODE samplers. Finally, different SEEDS solvers offer competitive performances on various datasets, considerablyfaster than other available stochastic methods*
