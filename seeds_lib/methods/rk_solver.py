import torch


class RK_SOLVER():
    def __init__(self, net, order, sigma, sigma_deriv, s_deriv, s, class_labels, c2):
        self.net = net
        self.class_labels = class_labels 
        self.sigma = sigma
        self.sigma_deriv = sigma_deriv
        self.s_deriv = s_deriv
        self.s = s
        
        if order == 1:
            self.update = self.first_update
        elif order == 2:
            if c2 is None:
                self.c2 = 1.
            else:
                self.c2 = c2
            self.update = self.second_update

    
    def first_update(self, x, s, t, num_steps, i):
        # Euler step.
        h = t - s
        denoised = self.net(x / self.s(s), self.sigma(s), self.class_labels).to(torch.float64)
        d_s = (self.sigma_deriv(s) / self.sigma(s) + self.s_deriv(s) / self.s(s)) * x - self.sigma_deriv(s) * self.s(s) / self.sigma(s) * denoised
        x_t = x + h * d_s
        return x_t
    
    
    def second_update(self, x, s, t, num_steps, i):
        ############ This is the Karras update:
        # Last step do Euler and before do:
        # 0 | 
        # c | c
        # ----------------------
        #   | 1-1/(2c)   1/(2c)
        ###################################
        h = t - s
        denoised = self.net(x / self.s(s), self.sigma(s), self.class_labels).to(torch.float64)
        d_s = (self.sigma_deriv(s) / self.sigma(s) + self.s_deriv(s) / self.s(s)) * x - self.sigma_deriv(s) * self.s(s) / self.sigma(s) * denoised
        x1 = x + self.c2 * h * d_s
        s1 = s + self.c2 * h

        if i == num_steps - 1:
            x_t = x + h * d_s
        else:
            denoised = self.net(x1 / self.s(s1), self.sigma(s1), self.class_labels).to(torch.float64)
            d_s1 = (self.sigma_deriv(s1) / self.sigma(s1) + self.s_deriv(s1) / self.s(s1)) * x1 - self.sigma_deriv(s1) * self.s(s1) / self.sigma(s1) * denoised
            x_t = x + h * ((1 - 1 / (2 * self.c2)) * d_s + 1 / (2 * self.c2) * d_s1)
        return x_t
    
    
    """     
    ######## This is the stochastic analog of Karras update
    def update(self, x_hat, t_hat, t_next, num_steps, i):
        # Euler step.
        denoised = self.net(x_hat, self.sigma(t_hat), self.class_labels).to(torch.float64)
        d_cur = 2 / self.sigma(t_hat) * (x_hat - denoised)
        h = t_next - t_hat
        z = torch.randn_like(x_hat)
        # Apply 2nd order correction.
        if self.solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur + (self.sigma(t_hat) ** 2 - self.sigma(t_next) ** 2).clip(min=0).sqrt() * z
        else:
            assert self.solver == 'heun'
            x_prime = x_hat + self.alpha * h * d_cur + (self.sigma(t_hat) ** 2 - self.sigma(t_next) ** 2).clip(min=0).sqrt() * z
            z1 = torch.randn_like(x_hat)
            t_prime = t_hat + self.alpha * h
            denoised = self.net(x_prime, self.sigma(t_prime), self.class_labels).to(torch.float64)
            d_prime = 2 / self.sigma(t_prime) * (x_prime - denoised)
            x_next = x_hat + h * ((1 - 1 / (2 * self.alpha)) * d_cur + 1 / (2 * self.alpha) * d_prime) + (self.sigma(t_hat) ** 2 - self.sigma(t_next) ** 2).clip(min=0).sqrt() * z1
        return x_next    
    """