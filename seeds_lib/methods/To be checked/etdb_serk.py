import torch
from ewok_solver.noise.brownian import Discrete, Standard


class ETDB_SERKW1(object):
    def __init__(self, model_fn, noise_schedule, noise_type="s", random_seed=77):
        """Construct ETDbis SERKW1 solvers which are variants of ETD SERKW1 ONLY in terms of phi functions.
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
        A single step for ETDbis SERKW1D1
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
        sigma_t = ns.marginal_std(t)
        h = lambda_t - lambda_s
        phi1 = torch.expm1(h) / h
        z = self.random(x)

        if noise_s in None:
            noise_s = self.model_fn(x, s)

        x_t = (
            torch.exp(alpha_t - alpha_s)[(...,) + (None,) * dims].mul(x)
            - (2 * sigma_t.mul(torch.expm1(h)))[(...,) + (None,) * dims].mul(noise_s)
            - (phi1 * sigma_t.mul(torch.sqrt(2)))[(...,) + (None,) * dims].mul(z)
        )

        if return_noise:
            return x_t, {"noise_s": noise_s}
        else:
            return x_t

    def second_update(self, x, s, t, noise_s=None, return_noise=False):
        return x
        """
        A single step for ETD SERKW1D2
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
        
        ns = self.noise_schedule
        n_channels = len(x.shape) - 1
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_t = ns.marginal_std(t)

        s1 = ns.inverse_lambda(lambda_s + h*0.5)
        log_alpha_s1 = ns.marginal_log_mean_coeff(s1)
        sigma_s1 = ns.marginal_std(s1)

        z, v = self.random(x), self.random(x)

        expm1 = torch.expm1(h)
        expm2 = torch.expm1(0.5*h)

        if noise_s is None:
            noise_s = self.model_fn(x, s)

        u = (torch.exp(log_alpha_s1 - log_alpha_s)[(...,) + (None,)*n_channels].mul(x)
                - 2*sigma_s1.mul(torch.expm1(h*0.5))[(...,) + (None,)*n_channels].mul(noise_s)
                - sigma_s1.mul(torch.sqrt(expm1))[(...,) + (None,)*n_channels].mul(z))

        noise_s1 = self.model_fn(u, s1)

        x_t = (torch.exp(log_alpha_t - log_alpha_s)[(...,) + (None,)*n_channels].mul(x)
                - 2*sigma_t.mul(expm1)[(...,) + (None,)*n_channels].mul(noise_s1)
                - sigma_t[(...,) + (None,)*n_channels].mul((torch.sqrt(expm1).mul(expm2+1))[(...,) + (None,)*n_channels].mul(z)
                                                          + torch.sqrt(expm1)[(...,) + (None,)*n_channels].mul(v)))   
        if return_noise:
            return x_t, {'noise_s': noise_s, 'noise_s1': noise_s1}
        else:
            return x_t
        """

    def third_update(
        self,
        x,
        s,
        t,
        r1=1.0 / 3.0,
        r2=2.0 / 3.0,
        noise_s=None,
        noise_s1=None,
        noise_s2=None,
    ):
        return x
        """
        A single step for ETD SERKW1-D3
        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (x.shape[0],).
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            r1: A `float`. The hyperparameter of the third-order solver. We recommend the default setting `1 / 3`.
            r2: A `float`. The hyperparameter of the third-order solver. We recommend the default setting `2 / 3`.
            noise_s: A pytorch tensor. The predicted noise at time `s`.
                If `noise_s` is None, we compute the predicted noise by `x` and `s`; otherwise we directly use it.
            noise_s1: A pytorch tensor. The predicted noise at time `s1` (the intermediate time given by `r1`).
                If `noise_s1` is None, we compute the predicted noise by `s1`; otherwise we directly use it.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        
        ns = self.noise_schedule
        n_channels = len(x.shape) - 1
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h
        s1 = ns.inverse_lambda(lambda_s1)
        s2 = ns.inverse_lambda(lambda_s2)
        log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(s2), ns.marginal_log_mean_coeff(t)
        sigma_s1, sigma_s2, sigma_t = ns.marginal_std(s1), ns.marginal_std(s2), ns.marginal_std(t)

        phi_11 = torch.expm1(r1 * h)
        phi_111 = torch.expm1(2*r1 * h)
        phi_12 = torch.expm1(r2 * h)
        phi_1 = torch.expm1(h)
        phi_22 = torch.expm1(r2 * h) / (r2 * h) - 1.
        phi_2 = torch.expm1(h) / h - 1.
        
        z1, z2, z3 = self.random(x), self.random(x), self.random(x)
        
        if noise_s is None:
            noise_s = self.model_fn(x, s)
            
        if noise_s1 is None:
            x_s1 = (
                torch.exp(log_alpha_s1 - log_alpha_s)[(...,) + (None,)*n_channels] * x
                - 2 * (sigma_s1 * phi_11)[(...,) + (None,)*n_channels] * noise_s
                - sigma_s1.mul(torch.sqrt(phi_12))[(...,) + (None,)*n_channels].mul(z1)
            )
            noise_s1 = self.model_fn(x_s1, s1)
            
        if noise_s2 is None:
            x_s2 = (torch.exp(log_alpha_s2 - log_alpha_s)[(...,) + (None,)*n_channels] * x
                - 2 * (sigma_s2 * phi_12)[(...,) + (None,)*n_channels] * noise_s
                - 4 * (sigma_s2 * phi_22)[(...,) + (None,)*n_channels] * (noise_s1 - noise_s)
                - sigma_s2[(...,) + (None,)*n_channels].mul((torch.sqrt(phi_12).mul(phi_11+1))[(...,) + (None,)*n_channels].mul(z1)
                                                          + torch.sqrt(phi_12)[(...,) + (None,)*n_channels].mul(z2)))
            
            noise_s2 = self.model_fn(x_s2, s2)
            
        x_t = (torch.exp(log_alpha_t - log_alpha_s)[(...,) + (None,)*n_channels] * x
               - 2 * (sigma_t * phi_1)[(...,) + (None,)*n_channels] * noise_s
               - 3 * (sigma_t * phi_2)[(...,) + (None,)*n_channels] * (noise_s2 - noise_s)
               - sigma_t[(...,) + (None,)*n_channels].mul((torch.sqrt(phi_12).mul(phi_12+1))[(...,) + (None,)*n_channels].mul(z1)
                                                          + (torch.sqrt(phi_12).mul(phi_11+1))[(...,) + (None,)*n_channels].mul(z2)
                                                          + torch.sqrt(phi_12)[(...,) + (None,)*n_channels].mul(z3)))
        return x_t
        """
