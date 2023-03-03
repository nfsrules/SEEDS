import torch
from ewok_solver.noise.brownian import Discrete, Standard


class ALT_SERKW1(object):
    def __init__(self, model_fn, noise_schedule, noise_type="s", random_seed=77):
        """Construct ALTERNATIVE SERKW1 solvers.
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
        A single step for ALTERNATE SERKW1D2 : stochastic contributions only in final steps
        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (x.shape[0],).
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            return_noise: A `bool`. If true, also return the predicted noise at time `s`.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """

        ns = self.noise_schedule
        n_channels = len(x.shape) - 1
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(
            s
        ), ns.marginal_log_mean_coeff(t)
        sigma_t = ns.marginal_std(t)

        s1 = ns.inverse_lambda(lambda_s + h * 0.5)
        log_alpha_s1 = ns.marginal_log_mean_coeff(s1)
        sigma_s1 = ns.marginal_std(s1)

        z = self.random(x)

        expm1 = torch.expm1(h)
        expm2 = torch.expm1(0.5 * h)

        if noise_s is None:
            noise_s = self.model_fn(x, s)

        u = torch.exp(log_alpha_s1 - log_alpha_s)[(...,) + (None,) * n_channels].mul(
            x
        ) - 2 * sigma_s1.mul(torch.expm1(h * 0.5))[(...,) + (None,) * n_channels].mul(
            noise_s
        )

        noise_s1 = self.model_fn(u, s1)

        x_t = (
            torch.exp(log_alpha_t - log_alpha_s)[(...,) + (None,) * n_channels].mul(x)
            - 2 * sigma_t.mul(expm1)[(...,) + (None,) * n_channels].mul(noise_s1)
            - (sigma_t.mul(torch.sqrt(torch.expm1(2 * h))))[
                (...,) + (None,) * n_channels
            ].mul(z)
        )
        if return_noise:
            return x_t, {"noise_s": noise_s, "noise_s1": noise_s1}
        else:
            return x_t

    def second_update(self, x, s, t, noise_s=None, return_noise=False):
        """
        A single step for ALT SERKW1D2 : the random var are inverted
        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (x.shape[0],).
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            r1: A `float`. The hyperparameter of the second-order solver. We recommend the default setting `0.5`.
            noise_s: A pytorch tensor. The predicted noise at time `s`.
                If `noise_s` is None, we compute the predicted noise by `x` and `s`; otherwise we directly use it.
            return_noise: A `bool`. If true, also return the predicted noise at time `s` and `s1` (the intermediate time).
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        n_channels = len(x.shape) - 1
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(
            s
        ), ns.marginal_log_mean_coeff(t)
        sigma_t = ns.marginal_std(t)

        s1 = ns.inverse_lambda(lambda_s + h * 0.5)
        log_alpha_s1 = ns.marginal_log_mean_coeff(s1)
        sigma_s1 = ns.marginal_std(s1)

        z, v = self.random(x), self.random(x)

        expm1 = torch.expm1(h)
        expm2 = torch.expm1(0.5 * h)

        if noise_s is None:
            noise_s = self.model_fn(x, s)

        u = (
            torch.exp(log_alpha_s1 - log_alpha_s)[(...,) + (None,) * n_channels].mul(x)
            - 2
            * sigma_s1.mul(torch.expm1(h * 0.5))[(...,) + (None,) * n_channels].mul(
                noise_s
            )
            - sigma_s1.mul(torch.sqrt(expm1))[(...,) + (None,) * n_channels].mul(z)
        )

        noise_s1 = self.model_fn(u, s1)

        x_t = (
            torch.exp(log_alpha_t - log_alpha_s)[(...,) + (None,) * n_channels].mul(x)
            - 2 * sigma_t.mul(expm1)[(...,) + (None,) * n_channels].mul(noise_s1)
            - sigma_t[(...,) + (None,) * n_channels].mul(
                (torch.sqrt(expm1).mul(expm2 + 1))[(...,) + (None,) * n_channels].mul(z)
                + torch.sqrt(expm1)[(...,) + (None,) * n_channels].mul(v)
            )
        )
        if return_noise:
            return x_t, {"noise_s": noise_s, "noise_s1": noise_s1}
        else:
            return x_t

    def third_update(self, x, s, t, noise_s=None, return_noise=False):
        """
        A single step for ALT SERKW1D2 : the random var are independent
        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (x.shape[0],).
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            r1: A `float`. The hyperparameter of the second-order solver. We recommend the default setting `0.5`.
            noise_s: A pytorch tensor. The predicted noise at time `s`.
                If `noise_s` is None, we compute the predicted noise by `x` and `s`; otherwise we directly use it.
            return_noise: A `bool`. If true, also return the predicted noise at time `s` and `s1` (the intermediate time).
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        n_channels = len(x.shape) - 1
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(
            s
        ), ns.marginal_log_mean_coeff(t)
        sigma_t = ns.marginal_std(t)

        s1 = ns.inverse_lambda(lambda_s + h * 0.5)
        log_alpha_s1 = ns.marginal_log_mean_coeff(s1)
        sigma_s1 = ns.marginal_std(s1)

        z, v = self.random(x), self.random(x)

        expm1 = torch.expm1(h)
        expm2 = torch.expm1(0.5 * h)

        if noise_s is None:
            noise_s = self.model_fn(x, s)

        u = (
            torch.exp(log_alpha_s1 - log_alpha_s)[(...,) + (None,) * n_channels].mul(x)
            - 2
            * sigma_s1.mul(torch.expm1(h * 0.5))[(...,) + (None,) * n_channels].mul(
                noise_s
            )
            - sigma_s1.mul(torch.sqrt(expm1))[(...,) + (None,) * n_channels].mul(z)
        )

        noise_s1 = self.model_fn(u, s1)

        x_t = (
            torch.exp(log_alpha_t - log_alpha_s)[(...,) + (None,) * n_channels].mul(x)
            - 2 * sigma_t.mul(expm1)[(...,) + (None,) * n_channels].mul(noise_s1)
            - (sigma_t.mul(torch.sqrt(torch.expm1(2 * h))))[
                (...,) + (None,) * n_channels
            ].mul(v)
        )

        if return_noise:
            return x_t, {"noise_s": noise_s, "noise_s1": noise_s1}
        else:
            return x_t
