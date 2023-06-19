import torch
import numpy as np
from seeds_lib.noise.brownian import Standard, Discrete

class ETD_ERK_EDM_SOLVER():
    def __init__(self, net, lamb, lamb_inv, class_labels, sigma, s, order, c2, noise_pred, butcher_type, sigma_d):
        self.net = net
        self.class_labels = class_labels
        self.s = s
        self.sigma = sigma
        self.sigma_d = sigma_d
        self.noise_pred = noise_pred
        self.lamb = lamb
        self.lamb_inv = lamb_inv
        
        if order == 1:
            self.update = self.first_update
        elif order == 2:
            if c2 is None:
                self.c2=0.5
            else:
                self.c2 = c2
            if butcher_type is None:
                self.butcher_type = '2a'
            else: 
                self.butcher_type = butcher_type
            self.update = self.second_update
        elif order == 3:
            if c2 is None:
                self.c2 = 1/3
            else:
                self.c2 = c2
            if butcher_type is None:
                self.butcher_type = '3a'
            else: 
                self.butcher_type = butcher_type
            self.c3 = 2./3.
            self.update = self.third_update
        else:
            raise NameError('Order {} not available in ETD_SERK_EDM method.'.format(order))
    
    def data_pred_fn_edm(self, model, x, t, sigma_data):
        c_skip = sigma_data ** 2 / (t ** 2 + sigma_data ** 2)
        c_out = t * sigma_data / (t ** 2 + sigma_data ** 2).sqrt()
        return (model - c_skip * x) / c_out

    def first_update(self, x, s, t, num_steps, i):
        # Get coefficients of noise schedule
        lambda_s, lambda_t = self.lamb(s), self.lamb(t)
        h = lambda_t - lambda_s
        if i != num_steps - 1: # exit contidion
            if not self.noise_pred: # Data prediction mode : D_theta in EDM
                raise NotImplementedError("The same as VE case")
            else: # Noise prediction mode : F_theta in EDM
                phi_1 = torch.expm1(h)
                model_s = self.net(x / self.s(s), self.sigma(s), self.class_labels).to(torch.float64)
                model_s = self.data_pred_fn_edm(model_s, x, self.sigma(s), self.sigma_d) 
                x_t = (
                    torch.sqrt(torch.tensor([(t**2+self.sigma_d**2)/(s**2+self.sigma_d**2)], device=x.device))*x 
                    + 2*torch.sqrt(torch.tensor([t**2+self.sigma_d**2],device=x.device)) / 2 * torch.arctan(t/self.sigma_d) * phi_1 * model_s)
        else: # Euler step
            x_t = x
        return x_t
    
    def second_update(self, x, s, t, num_steps, i):
        lambda_s, lambda_t = self.lamb(s), self.lamb(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + self.c2*h
        s1 = self.lamb_inv(lambda_s1)
        assert (s1 <= s and s1 >= t)
        
        if i != num_steps - 1: # exit contidion
            if not self.noise_pred: # Data prediction mode : D_theta in EDM
                raise NotImplementedError("The same as VE case")
            else: # Noise prediction mode : F_theta in EDM
                phi_1 = torch.expm1(h)
                phi_11 = torch.expm1(self.c2 * h)
                model_s = self.net(x / self.s(s), self.sigma(s), self.class_labels).to(torch.float64)
                model_s = self.data_pred_fn_edm(model_s, x, self.sigma(s), self.sigma_d) 
                
                x_s1 = (
                    torch.sqrt(torch.tensor([(s1**2+self.sigma_d**2)/(s**2+self.sigma_d**2)], device=x.device))*x 
                    + 2*torch.sqrt(torch.tensor([s1**2+self.sigma_d**2],device=x.device)) / 2 * torch.arctan(s1/self.sigma_d) * phi_11 * model_s)
                model_s1 = self.net(x_s1 / self.s(s1), self.sigma(s1), self.class_labels).to(torch.float64)
                model_s1 = self.data_pred_fn_edm(model_s1, x_s1, self.sigma(s1), self.sigma_d) 
                
                x_t = (
                    torch.sqrt(torch.tensor([(t**2+self.sigma_d**2)/(s**2+self.sigma_d**2)], device=x.device))*x 
                    + 2*torch.sqrt(torch.tensor([t**2+self.sigma_d**2],device=x.device)) / 2 * torch.arctan(t/self.sigma_d) * phi_1 * model_s
                    + 0.5/self.c2*2*torch.sqrt(torch.tensor([t**2+self.sigma_d**2],device=x.device)) / 2 * torch.arctan(t/self.sigma_d) * phi_1 * (model_s1 - model_s)
                )
        else: # Euler step
            x_t = x
        return x_t
    
    
    def third_update(self, x, s, t, num_steps, i):
        lambda_s, lambda_t = self.lamb(s), self.lamb(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + self.c2*h
        lambda_s2 = lambda_s + self.c3*h
        s1 = self.lamb_inv(lambda_s1)
        s2 = self.lamb_inv(lambda_s2)
        assert (s1 <= s and s1 >= t)
        assert (s2 <= s and s2 >= t)
        
        if i != num_steps - 1: # exit contidion
            if not self.noise_pred: # Data prediction mode : D_theta in EDM
                raise NotImplementedError("The same as VE case")
            else: # Noise prediction mode : F_theta in EDM
                phi_1 = torch.expm1(h)
                phi_11 = torch.expm1(self.c2 * h)
                phi_12 = torch.expm1(self.c3 * h)
                phi_22 = torch.expm1(self.c3 * h) / (self.c3 * h) - 1.
                phi_2 = torch.expm1(h) / h - 1.
                model_s = self.net(x / self.s(s), self.sigma(s), self.class_labels).to(torch.float64)
                model_s = self.data_pred_fn_edm(model_s, x, self.sigma(s), self.sigma_d) 
                
                x_s1 = (
                    torch.sqrt(torch.tensor([(s1**2+self.sigma_d**2)/(s**2+self.sigma_d**2)], device=x.device))*x 
                    + 2*torch.sqrt(torch.tensor([s1**2+self.sigma_d**2],device=x.device)) / 2 * torch.arctan(s1/self.sigma_d) * phi_11 * model_s
                )
                model_s1 = self.net(x_s1 / self.s(s1), self.sigma(s1), self.class_labels).to(torch.float64)
                model_s1 = self.data_pred_fn_edm(model_s1, x_s1, self.sigma(s1), self.sigma_d) 
                
                x_s2 = (
                    torch.sqrt(torch.tensor([(s2**2+self.sigma_d**2)/(s**2+self.sigma_d**2)], device=x.device))*x 
                    + 2*torch.sqrt(torch.tensor([s2**2+self.sigma_d**2],device=x.device)) / 2 * torch.arctan(s2/self.sigma_d) * phi_12 * model_s
                    + 2*self.c3/self.c2*torch.sqrt(torch.tensor([s2**2+self.sigma_d**2],device=x.device))/2*torch.arctan(s2/self.sigma_d)*phi_22*(model_s1 - model_s)
                )
                model_s2 = self.net(x_s2 / self.s(s2), self.sigma(s2), self.class_labels).to(torch.float64)
                model_s2 = self.data_pred_fn_edm(model_s2, x_s2, self.sigma(s2), self.sigma_d) 
                
                x_t = (
                    torch.sqrt(torch.tensor([(t**2+self.sigma_d**2)/(s**2+self.sigma_d**2)], device=x.device))*x 
                    + 2*torch.sqrt(torch.tensor([t**2+self.sigma_d**2],device=x.device)) / 2 * torch.arctan(t/self.sigma_d) * phi_1 * model_s1
                    - 0.5/self.c3*2*torch.sqrt(torch.tensor([t**2+self.sigma_d**2],device=x.device)) / 2 * torch.arctan(t/self.sigma_d) * phi_2 * (model_s2 - model_s1)
                )
        else: # Euler step
            x_t = x
        return x_t