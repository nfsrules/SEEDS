import torch
from ewok_solver.noise.brownian import Discrete, Standard


class M_SERKW1(object):
    def __init__(self, model_fn, noise_schedule, noise_type="s", random_seed=77):
        """Construct Mixed SERKW1 solvers.
        Args:
            model_fn: A noise prediction model function which accepts the continuous-time input
                (t in [epsilon, T]):
                ``
                def model_fn(x, t_continuous):
                    return noise
                ``
            noise_schedule: A noise schedule object, such as NoiseScheduleVP.
        """
        self.model_fn = model_fn
        self.noise_schedule = noise_schedule
        if noise_type == "s":
            self.random = Standard()
        elif noise_type == "d":
            self.random = Discrete()
        else:
            raise NameError(
                "{} method not implemented, only s (standard) or d (discrete).".format(
                    random_type
                )
            )

    def first_update(self, x, s, t, noise_s=None, return_noise=False):
        """
        A single step for M1 SERKW1-D1 (ETD in the deterministic part, IF in the stochastic part)
        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (x.shape[0],).
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            return_noise: A `bool`. If true, also return the predicted noise at time `s`.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """

        dims = len(x.shape) - 1
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        alpha_s, alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_t, sigma_s = ns.marginal_std(t), ns.marginal_std(s)
        a1 = torch.exp(alpha_t - alpha_s)
        h = lambda_t - lambda_s
        z = self.random(x)

        if noise_s is None:
            noise_s = self.model_fn(x, s)

        x_t = (
            a1[(...,) + (None,) * dims].mul(x)
            - (a1 * sigma_s.mul(torch.sqrt(2 * h)))[(...,) + (None,) * dims].mul(z)
            - (2 * sigma_t.mul(torch.expm1(h)))[(...,) + (None,) * dims].mul(noise_s)
        )

        if return_noise:
            return x_t, {"noise_s": noise_s}
        else:
            return x_t

    def second_update(self, x, s, t, noise_s=None, return_noise=False):
        """
        A single step for M2 SERKW1-D1 (ETD in the stochastic part, IF in the deterministic part)
        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (x.shape[0],).
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            return_noise: A `bool`. If true, also return the predicted noise at time `s`.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """

        dims = len(x.shape) - 1
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        alpha_s, alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_t, sigma_s = ns.marginal_std(t), ns.marginal_std(s)
        a1 = torch.exp(alpha_t - alpha_s)
        h = lambda_t - lambda_s
        z = self.random(x)

        if noise_s is None:
            noise_s = self.model_fn(x, s)

        x_t = (
            a1[(...,) + (None,) * dims].mul(x)
            - (a1 * 2 * h * sigma_s)[(...,) + (None,) * dims].mul(noise_s)
            - (sigma_t.mul(torch.sqrt(torch.expm1(2 * h))))[
                (...,) + (None,) * dims
            ].mul(z)
        )

        if return_noise:
            return x_t, {"noise_s": noise_s}
        else:
            return x_t

    def third_update(self, x, s, t, noise_s=None, return_noise=False):
        """
        A single step for M3 SERKW1-D1 (ETDb in the stochastic part, IF in the deterministic part)
        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (x.shape[0],).
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            return_noise: A `bool`. If true, also return the predicted noise at time `s`.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """

        dims = len(x.shape) - 1
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        alpha_s, alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_t, sigma_s = ns.marginal_std(t), ns.marginal_std(s)
        h = lambda_t - lambda_s
        a1 = torch.exp(alpha_t - alpha_s)
        phi1 = torch.expm1(h)

        z = self.random(x)

        if noise_s is None:
            noise_s = self.model_fn(x, s)

        x_t = (
            a1[(...,) + (None,) * dims].mul(x)
            - (a1 * 2 * h * sigma_s)[(...,) + (None,) * dims].mul(noise_s)
            - (sigma_t.mul(torch.sqrt(torch.tensor(2.0)) * phi1))[
                (...,) + (None,) * dims
            ].mul(z)
        )

        if return_noise:
            return x_t, {"noise_s": noise_s}
        else:
            return x_t
