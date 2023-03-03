import torch

from seeds_lib.noise.brownian import Discrete, Standard


class ETD_SERK_SOLVER:
    def __init__(
        self,
        net,
        lamb,
        lamb_inv,
        class_labels,
        sigma,
        s,
        order,
        c2,
        noise_pred,
        butcher_type,
        noise_type,
    ):
        self.net = net
        self.lamb = lamb
        self.lamb_inv = lamb_inv
        self.class_labels = class_labels
        self.s = s
        self.sigma = sigma
        self.noise_pred = noise_pred

        if noise_type is None or noise_type == "gaussian":
            self.randn_like = Standard()
        else:
            self.randn_like = Discrete()
        if order == 1:
            self.update = self.first_update
        elif order == 2:
            if c2 is None:
                self.c2 = 0.5
            else:
                self.c2 = c2
            if butcher_type is None:
                self.butcher_type = "2a"
            else:
                self.butcher_type = butcher_type
            self.update = self.second_update
        elif order == 3:
            if c2 is None:
                self.c2 = 1 / 3
            else:
                self.c2 = c2
            if butcher_type is None:
                self.butcher_type = "3a"
            else:
                self.butcher_type = butcher_type
            self.update = self.third_update
        else:
            raise NameError("Order {} not available in ETD_SERK method.".format(order))

    def first_update(self, x, s, t, num_steps, i):
        std = lambda u: self.sigma(u) * self.s(u)
        h = self.lamb(t) - self.lamb(s)
        n_pred = lambda data, x, u: (x - self.s(u) * data) / std(u)
        z = self.randn_like(x)
        gamma = lambda t, s: ((std(t) ** 2) / (std(s) ** 2)) * (self.s(s) / self.s(t))

        if i != num_steps - 1:
            if self.noise_pred:
                phi_1 = torch.expm1(h)
                psi_1 = torch.sqrt(torch.expm1(2.0 * h))
                model_s = self.net(x / self.s(s), self.sigma(s), self.class_labels).to(
                    torch.float64
                )
                model_s = n_pred(model_s, x, s)
                x_t = (
                    self.s(t) / self.s(s) * x
                    - 2.0 * std(t) * phi_1 * model_s
                    - std(t) * psi_1 * z
                )
            else:
                model_s = self.net(x / self.s(s), self.sigma(s), self.class_labels).to(
                    torch.float64
                )
                phi_1 = torch.expm1(-2.0 * h)
                # psi_1 = torch.sqrt(torch.expm1(2. * h).clip(min=0))
                psi_1 = torch.sqrt(-torch.expm1(-2.0 * h))
                x_t = gamma(t, s) * x - self.s(t) * phi_1 * model_s - std(t) * psi_1 * z
        else:
            x_t = x
        return x_t

    def second_update(self, x, s, t, num_steps, i):
        assert self.butcher_type in ["2a", "2b"]

        std = lambda u: self.sigma(u) * self.s(u)
        h = self.lamb(t) - self.lamb(s)
        n_pred = lambda data, x, u: (x - self.s(u) * data) / std(u)
        lambda_s2 = self.lamb(s) + self.c2 * h
        s2 = self.lamb_inv(lambda_s2)
        gamma = lambda t, s: ((std(t) ** 2) / (std(s) ** 2)) * (self.s(s) / self.s(t))
        z1, z2 = self.randn_like(x), self.randn_like(x)

        if i != num_steps - 1:
            if self.noise_pred:
                phi_1 = torch.expm1(h)
                phi_12 = torch.expm1(self.c2 * h)

                psi_1 = torch.sqrt(torch.expm1(2.0 * h * self.c2))
                psi_2 = torch.sqrt(torch.expm1(2.0 * h * (1.0 - self.c2)))
                A_t1 = (phi_12 + 1.0) * psi_2
                noise1 = std(s2) * psi_1 * z1
                noise2 = std(t) * A_t1 * z1 + std(t) * psi_1 * z2

                model_s = self.net(x / self.s(s), self.sigma(s), self.class_labels).to(
                    torch.float64
                )
                model_s = n_pred(model_s, x, s)

                x_s2 = (
                    self.s(s2) / self.s(s) * x
                    - 2.0 * std(s2) * phi_12 * model_s
                    - noise1
                )

                model_s2 = self.net(
                    x_s2 / self.s(s2), self.sigma(s2), self.class_labels
                ).to(torch.float64)
                model_s2 = n_pred(model_s2, x_s2, s2)

                if self.butcher_type == "2a":
                    x_t = (
                        self.s(t) / self.s(s) * x
                        - 2.0 * std(t) * phi_1 * model_s
                        - (1.0 / self.c2) * std(t) * phi_1 * (model_s2 - model_s)
                        - noise2
                    )
                elif self.butcher_type == "2b":
                    phi_2 = phi_1 / h - 1.0
                    x_t = (
                        self.s(t) / self.s(s) * x
                        - 2.0 * std(t) * phi_1 * model_s
                        - (2.0 / self.c2) * std(t) * phi_2 * (model_s2 - model_s)
                        - noise2
                    )
            else:
                phi_1 = torch.expm1(-2.0 * h)
                phi_12 = torch.expm1(-self.c2 * 2.0 * h)
                psi = lambda u: torch.sqrt(-torch.expm1(-2.0 * h * u))
                phi12 = torch.expm1(-self.c2 * h)
                A_t1 = psi(1.0 - self.c2) * (phi12 + 1.0)
                noise1 = std(s2) * psi(self.c2) * z1
                noise2 = std(t) * psi(self.c2) * z2 + std(t) * A_t1 * z1

                model_s = self.net(x / self.s(s), self.sigma(s), self.class_labels).to(
                    torch.float64
                )

                x_s2 = gamma(s2, s) * x - self.s(s2) * phi_12 * model_s - noise1

                model_s2 = self.net(
                    x_s2 / self.s(s2), self.sigma(s2), self.class_labels
                ).to(torch.float64)

                if self.butcher_type == "2a":
                    x_t = (
                        gamma(t, s) * x
                        - self.s(t) * phi_1 * model_s
                        - (0.5 / self.c2) * self.s(t) * phi_1 * (model_s2 - model_s)
                        - noise2
                    )
                elif self.butcher_type == "2b":
                    # TO BE CORRECTED
                    phi_2 = phi_1 / (2 * h) + 1.0
                    x_t = (
                        gamma(t, s) * x
                        - self.s(t) * phi_1 * model_s
                        + (1.0 / self.c2) * self.s(t) * phi_2 * (model_s2 - model_s)
                        - noise2
                    )
        else:
            x_t = x
        return x_t

    def third_update(self, x, s, t, num_steps, i):
        assert self.butcher_type in ["3a", "3b"]

        std = lambda u: self.sigma(u) * self.s(u)
        h = self.lamb(t) - self.lamb(s)
        n_pred = lambda data, x, u: (x - self.s(u) * data) / std(u)
        c3 = 2 / 3
        lambda_s2 = self.lamb(s) + self.c2 * h
        lambda_s3 = self.lamb(s) + c3 * h
        s2 = self.lamb_inv(lambda_s2)
        s3 = self.lamb_inv(lambda_s3)

        gamma = lambda t, s: ((std(t) ** 2) / (std(s) ** 2)) * (self.s(s) / self.s(t))

        z1, z2, z3 = self.randn_like(x), self.randn_like(x), self.randn_like(x)

        if i != num_steps - 1:
            if self.noise_pred:
                phi_12 = torch.expm1(self.c2 * h)
                phi_13 = torch.expm1(c3 * h)
                phi_1 = torch.expm1(h)
                phi_23 = torch.expm1(c3 * h) / (c3 * h) - 1.0
                phi_2 = phi_1 / h - 1.0
                phi_3 = phi_2 / h - 0.5

                psi_1 = torch.sqrt(torch.expm1(2.0 * h * self.c2))
                psi_2 = torch.sqrt(torch.expm1(2.0 * h * (c3 - self.c2)))
                psi_3 = torch.sqrt(torch.expm1(2.0 * h * (1.0 - c3)))
                A_t1 = (phi_12 + 1.0) * psi_2
                A_t2 = (phi_13 + 1.0) * psi_3
                noise1 = std(s2) * psi_1 * z1
                noise2 = std(t) * A_t1 * z1 + std(t) * psi_1 * z2
                noise3 = std(t) * A_t1 * z1 + std(t) * A_t2 * z2 + std(t) * psi_1 * z3

                model_s = self.net(x / self.s(s), self.sigma(s), self.class_labels).to(
                    torch.float64
                )
                model_s = n_pred(model_s, x, s)

                x_s2 = (
                    self.s(s2) / self.s(s) * x
                    - 2.0 * std(s2) * phi_12 * model_s
                    - noise1
                )
                model_s2 = self.net(
                    x_s2 / self.s(s2), self.sigma(s2), self.class_labels
                ).to(torch.float64)
                model_s2 = n_pred(model_s2, x_s2, s2)

                x_s3 = (
                    self.s(s3) / self.s(s) * x
                    - 2.0 * std(s3) * phi_13 * model_s
                    - 2.0 * c3 / self.c2 * std(s3) * phi_23 * (model_s2 - model_s)
                    - noise2
                )
                model_s3 = self.net(
                    x_s3 / self.s(s3), self.sigma(s3), self.class_labels
                ).to(torch.float64)
                model_s3 = n_pred(model_s3, x_s3, s3)

                if self.butcher_type == "3a":
                    x_t = (
                        self.s(t) / self.s(s) * x
                        - 2.0 * std(t) * phi_1 * model_s
                        - 3.0 * std(t) * phi_2 * (model_s3 - model_s)
                        - noise3
                    )
                elif self.butcher_type == "3b":
                    assert self.c2 == 1 / 3
                    D1 = -1.5 * model_s3 + 6.0 * model_s2 - 4.5 * model_s
                    D2 = 9.0 * (model_s3 - 2.0 * model_s2 + model_s)
                    x_t = (
                        self.s(t) / self.s(s) * x
                        - 2.0 * std(t) * phi_1 * model_s
                        - 2.0 * std(t) * phi_2 * D1
                        - 2.0 * std(t) * phi_3 * D2
                        - noise3
                    )
            else:
                phi_12 = torch.expm1(-self.c2 * 2.0 * h)
                phi_13 = torch.expm1(-c3 * 2.0 * h)
                phi_1 = torch.expm1(-2.0 * h)
                psi = lambda u: torch.sqrt(-torch.expm1(-2.0 * h * u))
                phi12 = torch.expm1(-self.c2 * h)
                phi13 = torch.expm1(-c3 * h)
                A_t1 = psi(c3 - self.c2) * (phi12 + 1.0)
                A_t2 = psi(1 - c3) * (phi13 + 1.0)
                noise1 = std(s2) * psi(self.c2) * z1
                noise2 = std(s3) * psi(self.c2) * z2 + std(s3) * A_t1 * z1
                noise3 = (
                    std(t) * psi(self.c2) * z3 + std(t) * A_t1 * z2 + std(t) * A_t2 * z1
                )
                # either this, or the role of 'A' in Luan is not taken into account
                phi_2 = phi_1 / (2 * h) + 1.0
                phi_23 = torch.expm1(-2.0 * self.c2 * h) / (self.c2 * h) + 1.0
                phi_3 = phi_2 / (2 * h) - 0.5
                model_s = self.net(x / self.s(s), self.sigma(s), self.class_labels).to(
                    torch.float64
                )
                # --------------------------------------------------------------
                x_s2 = gamma(s2, s) * x - self.s(s2) * phi_12 * model_s - noise1
                model_s2 = self.net(
                    x_s2 / self.s(s2), self.sigma(s2), self.class_labels
                ).to(torch.float64)
                x_s3 = (
                    gamma(s3, s) * x
                    - self.s(s3) * phi_13 * model_s
                    + c3 / self.c2 * self.s(s3) * phi_23 * (model_s2 - model_s)
                    - noise2
                )
                model_s3 = self.net(
                    x_s3 / self.s(s3), self.sigma(s3), self.class_labels
                ).to(torch.float64)
                if self.butcher_type == "3a":
                    x_t = (
                        gamma(t, s) * x
                        - self.s(t) * phi_1 * model_s
                        + 1.5 * self.s(t) * phi_2 * (model_s3 - model_s)
                        - noise3
                    )
                elif self.butcher_type == "3b":
                    D1 = -1.5 * model_s3 + 6.0 * model_s2 - 4.5 * model_s
                    D2 = 9.0 * (model_s3 - 2.0 * model_s2 + model_s)
                    x_t = (
                        gamma(t, s) * x
                        - self.s(t) * phi_1 * model_s
                        + self.s(t) * phi_2 * D1
                        - self.s(t) * phi_3 * D2
                        - noise3
                    )

        else:
            x_t = x
        return x_t
