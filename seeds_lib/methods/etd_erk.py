import torch


class ETD_ERK_SOLVER:
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
    ):
        self.net = net
        self.lamb = lamb
        self.lamb_inv = lamb_inv
        self.class_labels = class_labels
        self.s = s
        self.sigma = sigma
        self.noise_pred = noise_pred

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
        elif order == 4:
            self.update = self.fourth_update
        else:
            raise NameError("Order {} not available in ETD_ERK method.".format(order))

    def first_update(self, x, s, t, num_steps, i):
        std = lambda u: self.sigma(u) * self.s(u)
        h = self.lamb(t) - self.lamb(s)
        n_pred = lambda data, x, u: (x - self.s(u) * data) / std(u)

        if i != num_steps - 1:  # exit contidion
            if self.noise_pred:  # Noise prediction mode : F_theta in EDM
                phi_1 = torch.expm1(h)
                model_s = self.net(x / self.s(s), self.sigma(s), self.class_labels).to(
                    torch.float64
                )
                model_s = n_pred(model_s, x, s)
                x_t = self.s(t) / self.s(s) * x - std(t) * phi_1 * model_s
            else:  # Data prediction mode : D_theta in EDM
                phi_1 = torch.expm1(-h)
                model_s = self.net(x / self.s(s), self.sigma(s), self.class_labels).to(
                    torch.float64
                )
                x_t = std(t) / std(s) * x - self.s(t) * phi_1 * model_s
        else:  # Do nothing
            x_t = x
        return x_t

    def second_update(self, x, s, t, num_steps, i):
        assert self.butcher_type in ["2a", "2b"]

        std = lambda u: self.sigma(u) * self.s(u)
        h = self.lamb(t) - self.lamb(s)
        n_pred = lambda data, x, u: (x - self.s(u) * data) / std(u)
        lambda_s2 = self.lamb(s) + self.c2 * h
        s2 = self.lamb_inv(lambda_s2)

        if i != num_steps - 1:  # exit contidion
            if self.noise_pred:  # Noise prediction mode
                phi_12 = torch.expm1(self.c2 * h)
                phi_1 = torch.expm1(h)
                model_s = self.net(x / self.s(s), self.sigma(s), self.class_labels).to(
                    torch.float64
                )
                model_s = n_pred(model_s, x, s)
                x_s2 = self.s(s2) / self.s(s) * x - std(s2) * phi_12 * model_s
                model_s2 = self.net(
                    x_s2 / self.s(s2), self.sigma(s2), self.class_labels
                ).to(torch.float64)
                model_s2 = n_pred(model_s2, x_s2, s2)
                if self.butcher_type == "2a":
                    x_t = (
                            self.s(t) / self.s(s) * x
                            - std(t) * phi_1 * model_s
                            - (0.5 / self.c2) * std(t) * phi_1 * (model_s2 - model_s)
                    )
                elif self.butcher_type == "2b":
                    phi_2 = phi_1 / h - 1.0
                    x_t = (
                            self.s(t) / self.s(s) * x
                            - std(t) * phi_1 * model_s
                            - (1.0 / self.c2) * std(t) * phi_2 * (model_s2 - model_s)
                    )
            else:
                phi_12 = torch.expm1(-self.c2 * h)
                phi_1 = torch.expm1(-h)
                model_s = self.net(x / self.s(s), self.sigma(s), self.class_labels).to(
                    torch.float64
                )
                x_s2 = std(s2) / std(s) * x - self.s(s2) * phi_12 * model_s
                model_s2 = self.net(
                    x_s2 / self.s(s2), self.sigma(s2), self.class_labels
                ).to(torch.float64)
                if self.butcher_type == "2a":
                    x_t = (
                            std(t) / std(s) * x
                            - self.s(t) * phi_1 * model_s
                            - (0.5 / self.c2) * self.s(t) * phi_1 * (model_s2 - model_s)
                    )
                elif self.butcher_type == "2b":
                    phi_2 = phi_1 / h + 1.0
                    x_t = (
                            std(t) / std(s) * x
                            - self.s(t) * phi_1 * model_s
                            + (1.0 / self.c2) * self.s(t) * phi_2 * (model_s2 - model_s)
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

        if i != num_steps - 1:  # exit contidion
            if self.noise_pred:  # Noise prediction mode
                phi_12 = torch.expm1(self.c2 * h)
                phi_13 = torch.expm1(c3 * h)
                phi_1 = torch.expm1(h)
                phi_23 = torch.expm1(c3 * h) / (c3 * h) - 1.0
                phi_2 = phi_1 / h - 1.0
                phi_3 = phi_2 / h - 0.5
                model_s = self.net(x / self.s(s), self.sigma(s), self.class_labels).to(
                    torch.float64
                )
                model_s = n_pred(model_s, x, s)
                x_s2 = self.s(s2) / self.s(s) * x - std(s2) * phi_12 * model_s
                model_s2 = self.net(
                    x_s2 / self.s(s2), self.sigma(s2), self.class_labels
                ).to(torch.float64)
                model_s2 = n_pred(model_s2, x_s2, s2)
                x_s3 = (
                        self.s(s3) / self.s(s) * x
                        - std(s3) * phi_13 * model_s
                        - c3 / self.c2 * std(s3) * phi_23 * (model_s2 - model_s)
                )
                model_s3 = self.net(
                    x_s3 / self.s(s3), self.sigma(s3), self.class_labels
                ).to(torch.float64)
                model_s3 = n_pred(model_s3, x_s3, s3)

                if self.butcher_type == "3a":
                    x_t = (
                            self.s(t) / self.s(s) * x
                            - std(t) * phi_1 * model_s
                            - 1.5 * std(t) * phi_2 * (model_s3 - model_s)
                    )
                elif self.butcher_type == "3b":
                    assert self.c2 == 1 / 3
                    D1 = -1.5 * model_s3 + 6.0 * model_s2 - 4.5 * model_s
                    D2 = 9.0 * (model_s3 - 2.0 * model_s2 + model_s)
                    x_t = (
                            self.s(t) / self.s(s) * x
                            - std(t) * phi_1 * model_s
                            - std(t) * phi_2 * D1
                            - std(t) * phi_3 * D2
                    )
            else:
                phi_12 = torch.expm1(-self.c2 * h)
                phi_13 = torch.expm1(-c3 * h)
                phi_1 = torch.expm1(-h)
                phi_23 = torch.expm1(-c3 * h) / (c3 * h) + 1.0
                phi_2 = phi_1 / h + 1.0
                phi_3 = phi_2 / h - 0.5
                model_s = self.net(x / self.s(s), self.sigma(s), self.class_labels).to(
                    torch.float64
                )
                x_s2 = std(s2) / std(s) * x - self.s(s2) * phi_12 * model_s
                model_s2 = self.net(
                    x_s2 / self.s(s2), self.sigma(s2), self.class_labels
                ).to(torch.float64)
                x_s3 = (
                        std(s3) / std(s) * x
                        - self.s(s3) * phi_13 * model_s
                        + (c3 / self.c2) * self.s(s3) * phi_23 * (model_s2 - model_s)
                )
                model_s3 = self.net(
                    x_s3 / self.s(s3), self.sigma(s3), self.class_labels
                ).to(torch.float64)

                if self.butcher_type == "3a":
                    x_t = (
                            std(t) / std(s) * x
                            - self.s(t) * phi_1 * model_s
                            + 1.5 * self.s(t) * phi_2 * (model_s3 - model_s)
                    )
                elif self.butcher_type == "3b":
                    assert self.c2 == 1 / 3
                    D1 = -1.5 * model_s3 + 6.0 * model_s2 - 4.5 * model_s
                    D2 = 9.0 * (model_s3 - 2.0 * model_s2 + model_s)
                    x_t = (
                            std(t) / std(s) * x
                            - self.s(t) * phi_1 * model_s
                            + self.s(t) * phi_2 * D1
                            - self.s(t) * phi_3 * D2
                    )
        else:
            x_t = x
        return x_t

    def fourth_update(self, x, s, t, num_steps, i):
        std = lambda u: self.sigma(u) * self.s(u)
        h = self.lamb(t) - self.lamb(s)
        n_pred = lambda data, x, u: (x - self.s(u) * data) / std(u)
        c2 = 0.5
        lambda_s2 = self.lamb(s) + c2 * h
        s2 = self.lamb_inv(lambda_s2)
        if i != num_steps - 1:
            if self.noise_pred:
                phi_12 = torch.expm1(c2 * h)
                phi_1 = torch.expm1(h)
                phi_23 = 4.0 * (phi_12 / h) - 2.0
                phi_2 = phi_1 / h - 1.0
                phi_3 = phi_2 / h - 0.5
                phi_35 = 8.0 * (phi_12 / (h * h)) - (4.0 / h) - 1.0
                a_52 = (c2 * phi_23) - phi_3 - (c2 * c2 * phi_2) - (c2 * phi_35)
                phi_25 = phi_12 / h - c2
                a_54 = phi_25 + a_52
                int1 = self.s(s2) / self.s(s)
                int2 = self.s(t) / self.s(s)
                model_s = self.net(x / self.s(s), self.sigma(s), self.class_labels).to(
                    torch.float64
                )
                model_s = n_pred(model_s, x, s)
                x_s2 = int1 * x - std(s2) * phi_12 * model_s
                model_s2 = self.net(
                    x_s2 / self.s(s2), self.sigma(s2), self.class_labels
                ).to(torch.float64)
                model_s2 = n_pred(model_s2, x_s2, s2)
                x_s3 = (
                        int1 * x
                        - std(s2) * phi_12 * model_s
                        - std(s2) * phi_23 * (model_s2 - model_s)
                )
                model_s3 = self.net(
                    x_s3 / self.s(s2), self.sigma(s2), self.class_labels
                ).to(torch.float64)
                model_s3 = n_pred(model_s3, x_s3, s2)
                x_s4 = (
                        int2 * x
                        - std(t) * phi_1 * model_s
                        - std(t) * phi_2 * (model_s2 + model_s3 - (2.0 * model_s))
                )
                model_s4 = self.net(
                    x_s4 / self.s(t), self.sigma(t), self.class_labels
                ).to(torch.float64)
                model_s4 = n_pred(model_s4, x_s4, t)
                x_s5 = (
                        int1 * x
                        - std(s2) * phi_12 * model_s
                        - std(s2) * a_52 * (model_s2 + model_s3 - (2.0 * model_s))
                        - std(s2) * a_54 * (model_s4 - model_s)
                )
                model_s5 = self.net(
                    x_s5 / self.s(s2), self.sigma(s2), self.class_labels
                ).to(torch.float64)
                model_s5 = n_pred(model_s5, x_s5, s2)
                x_t = (
                        int2 * x
                        - std(t) * phi_1 * model_s
                        - std(t) * phi_2 * ((4.0 * model_s5) - model_s4 - (3.0 * model_s))
                        - 4.0 * std(t) * phi_3 * (model_s + model_s4 - (2.0 * model_s5))
                )

            else:
                print("TBA")
        else:
            x_t = x
        return x_t
