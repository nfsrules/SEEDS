import os
import pickle
import re

import click
import numpy as np
import PIL.Image
import torch
import tqdm

import dnnlib
from torch_utils import distributed as dist


def sampler(
    net,
    latents,
    class_labels=None,
    randn_like=torch.randn_like,
    num_steps=18,
    sigma_min=None,
    sigma_max=None,
    rho=7,
    solver="rk",
    discretization="edm",
    schedule="linear",
    scaling="none",
    epsilon_s=1e-3,
    C_1=0.001,
    C_2=0.008,
    M=1000,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
    order=1,
    c2=None,
    noise_pred=True,
    noise_type="gaussian",
    butcher_type=None,
):
    assert solver in ["rk", "etd-erk", "etd-serk"]
    assert discretization in ["vp", "ve", "iddpm", "edm"]
    assert schedule in ["vp", "ve", "linear"]
    assert scaling in ["vp", "none"]

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = (
        lambda beta_d, beta_min: lambda t: (
            np.e ** (0.5 * beta_d * (t**2) + beta_min * t) - 1
        )
        ** 0.5
    )
    vp_sigma_deriv = (
        lambda beta_d, beta_min: lambda t: 0.5
        * (beta_min + beta_d * t)
        * (sigma(t) + 1 / sigma(t))
    )
    vp_sigma_inv = (
        lambda beta_d, beta_min: lambda sigma: (
            (beta_min**2 + 2 * beta_d * (sigma**2 + 1).log()).sqrt() - beta_min
        )
        / beta_d
    )
    vp_log_alpha = (
        lambda beta_d, beta_min: lambda t: -0.25 * t**2 * beta_d - 0.5 * t * beta_min
    )
    vp_lambd = lambda beta_d, beta_min: lambda t: vp_log_alpha(beta_d, beta_min)(
        t
    ) - 0.5 * torch.log(1.0 - torch.exp(2.0 * vp_log_alpha(beta_d, beta_min)(t)))
    vp_lambd_inv = (
        lambda beta_d, beta_min: lambda lamb: 2.0
        * beta_d
        * torch.logaddexp(-2.0 * lamb, torch.zeros((1,)).to(lamb))
        / (
            torch.sqrt(
                beta_min**2
                + 2.0
                * beta_d
                * torch.logaddexp(-2.0 * lamb, torch.zeros((1,)).to(lamb))
            )
            + beta_min
        )
        / beta_d
    )

    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma**2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.1, beta_min=0.1)(t=epsilon_s)
        sigma_min = {"vp": vp_def, "ve": 0.02, "iddpm": 0.002, "edm": 0.002}[
            discretization
        ]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.1, beta_min=0.1)(t=1)
        sigma_max = {"vp": vp_def, "ve": 100, "iddpm": 81, "edm": 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = (
        2
        * (np.log(sigma_min**2 + 1) / epsilon_s - np.log(sigma_max**2 + 1))
        / (epsilon_s - 1)
    )
    vp_beta_min = np.log(sigma_max**2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == "vp":
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == "ve":
        orig_t_steps = (sigma_max**2) * (
            (sigma_min**2 / sigma_max**2) ** (step_indices / (num_steps - 1))
        )
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == "iddpm":
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device):  # M, ..., 1
            u[j - 1] = (
                (u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1
            ).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[
            ((len(u_filtered) - 1) / (num_steps - 1) * step_indices)
            .round()
            .to(torch.int64)
        ]
    else:
        assert discretization == "edm"
        if rho is None:
            rho = 7
        sigma_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho

    # Define noise level schedule.
    if schedule == "vp":
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
        lamb = vp_lambd(vp_beta_d, vp_beta_min)
        lamb_inv = vp_lambd_inv(vp_beta_d, vp_beta_min)
    elif schedule == "ve":
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
        lamb = lambda t: -torch.log(sigma(t))
        lamb_inv = lambda t: torch.exp(-2.0 * t)
    else:
        assert schedule == "linear"
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma
        lamb = lambda t: -torch.log(t)
        lamb_inv = lambda t: torch.exp(-t)

    # Define scaling schedule.
    if scaling == "vp":
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == "none"
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))

    from seeds_lib import SEEDS_SOLVER

    solver_fn = SEEDS_SOLVER(
        solver,
        net,
        lamb=lamb,
        lamb_inv=lamb_inv,
        class_labels=class_labels,
        sigma=sigma,
        sigma_deriv=sigma_deriv,
        s=s,
        s_deriv=s_deriv,
        order=order,
        c2=c2,
        noise_pred=noise_pred,
        noise_type=noise_type,
        butcher_type=butcher_type,
    )

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next
        if not S_churn == 0:
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= sigma(t_cur) <= S_max
                else 0
            )
            t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
            x_hat = s(t_hat) / s(t_cur) * x_cur + (
                sigma(t_hat) ** 2 - sigma(t_cur) ** 2
            ).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        x_next = solver_fn.update(x_hat, t_hat, t_next, num_steps, i=i)
    return x_next


# ----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds
        ]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators]
        )

    def randn_like(self, input):
        return self.randn(
            input.shape, dtype=input.dtype, layout=input.layout, device=input.device
        )

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [
                torch.randint(*args, size=size[1:], generator=gen, **kwargs)
                for gen in self.generators
            ]
        )


# ----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]


def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------
@click.command()
@click.option(
    "--network",
    "network_pkl",
    help="Network pickle filename",
    metavar="PATH|URL",
    type=str,
    required=True,
)
@click.option(
    "--outdir",
    help="Where to save the output images",
    metavar="DIR",
    type=str,
    required=True,
)
@click.option(
    "--seeds",
    help="Random seeds (e.g. 1,2,5-10)",
    metavar="LIST",
    type=parse_int_list,
    default="1-64",
    show_default=True,
)
@click.option(
    "--subdirs", help="Create subdirectory for every 1000 seeds", is_flag=True
)
@click.option(
    "--class",
    "class_idx",
    help="Class label  [default: random]",
    metavar="INT",
    type=click.IntRange(min=0),
    default=None,
)
@click.option(
    "--batch",
    "max_batch_size",
    help="Maximum batch size",
    metavar="INT",
    type=click.IntRange(min=1),
    default=64,
    show_default=True,
)
@click.option(
    "--steps",
    "num_steps",
    help="Number of steps",
    metavar="INT",
    type=click.IntRange(min=1),
    default=18,
    show_default=True,
)
@click.option(
    "--sigma_min",
    help="Lowest noise level  [default: varies]",
    metavar="FLOAT",
    type=click.FloatRange(min=0, min_open=True),
)
@click.option(
    "--sigma_max",
    help="Highest noise level  [default: varies]",
    metavar="FLOAT",
    type=click.FloatRange(min=0, min_open=True),
)
@click.option(
    "--rho",
    help="Time step exponent",
    metavar="FLOAT",
    type=click.FloatRange(min=0, min_open=True),
)
@click.option(
    "--S_churn",
    "S_churn",
    help="Stochasticity strength",
    metavar="FLOAT",
    type=click.FloatRange(min=0),
)
@click.option(
    "--S_min",
    "S_min",
    help="Stoch. min noise level",
    metavar="FLOAT",
    type=click.FloatRange(min=0),
)
@click.option(
    "--S_max",
    "S_max",
    help="Stoch. max noise level",
    metavar="FLOAT",
    type=click.FloatRange(min=0),
)
@click.option(
    "--S_noise", "S_noise", help="Stoch. noise inflation", metavar="FLOAT", type=float
)
@click.option(
    "--c2",
    help="Solver free param",
    metavar="FLOAT",
    type=click.FloatRange(min=0, min_open=True, max=1),
)
@click.option(
    "--butcher_type",
    help="Butcher Type",
    metavar="STR",
    type=click.Choice(["2a", "2b", "3a", "3b", "3c"]),
)
@click.option(
    "--noise_type",
    help="Noise Type",
    metavar="STR",
    type=click.Choice(["gaussian", "discrete"]),
)
@click.option(
    "--solver",
    help="Type of  solver",
    metavar="rk|etd-erk",
    type=click.Choice(["rk", "etd-erk", "etd-serk"]),
)
@click.option(
    "--disc",
    "discretization",
    help="Time step discretization {t_i}",
    metavar="vp|ve|iddpm|edm",
    type=click.Choice(["vp", "ve", "iddpm", "edm"]),
)
@click.option(
    "--schedule",
    help="Noise schedule sigma(t)",
    metavar="vp|ve|linear",
    type=click.Choice(["vp", "ve", "linear"]),
)
@click.option(
    "--scaling",
    help="Signal scaling s(t)",
    metavar="vp|none",
    type=click.Choice(["vp", "none"]),
)
@click.option(
    "--order",
    help="Deterministic order of the solver",
    metavar="INT",
    type=click.IntRange(min=1, max=4),
    default=1,
    show_default=True,
)
@click.option("--noise_pred", help="Noise prediction mode", type=bool, default=True)
def main(
    network_pkl,
    outdir,
    subdirs,
    seeds,
    class_idx,
    max_batch_size,
    device=torch.device("cuda"),
    **sampler_kwargs,
):
    dist.init()
    num_batches = (
        (len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1
    ) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)["ema"].to(device)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Settings print
    sampler_args = {
        key: value for key, value in sampler_kwargs.items() if value is not None
    }
    dist.print0(f"sampler_args {sampler_args}")

    sol = sampler_args["solver"]
    ords = sampler_args["order"]
    nsteps = sampler_args["num_steps"]
    if sol == "rk" and ords == 1:
        nfes = nsteps
    elif sol == "rk" and ords == 2:
        nfes = nsteps * ords - 1
    else:
        nfes = ords * (nsteps - 1)
    # dist.print0(f'Ords = {ot}')
    dist.print0(f"NFEs = {nfes}")
    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(
        rank_batches, unit="batch", disable=(dist.get_rank() != 0)
    ):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn(
            [batch_size, net.img_channels, net.img_resolution, net.img_resolution],
            device=device,
        )
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[
                rnd.randint(net.label_dim, size=[batch_size], device=device)
            ]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # Generate images.
        sampler_kwargs = {
            key: value for key, value in sampler_kwargs.items() if value is not None
        }
        # have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
        # sampler_fn = ablation_sampler #if have_ablation_kwargs else edm_sampler
        images = sampler(
            net, latents, class_labels, randn_like=rnd.randn_like, **sampler_kwargs
        )

        # Save images.
        images_np = (
            (images * 127.5 + 128)
            .clip(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = (
                os.path.join(outdir, f"{seed-seed%1000:06d}") if subdirs else outdir
            )
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f"{seed:06d}.png")
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], "L").save(image_path)
            else:
                PIL.Image.fromarray(image_np, "RGB").save(image_path)

    # Done.
    torch.distributed.barrier()


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
