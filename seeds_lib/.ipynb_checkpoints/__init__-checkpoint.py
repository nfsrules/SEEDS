import torch


class SEEDS_SOLVER():
    def __new__(self, solver, net, lamb, lamb_inv, class_labels, sigma, s, order, c2, noise_pred, butcher_type, sigma_deriv=None, s_deriv=None, noise_type='gaussian'):
        assert noise_type in ['gaussian', 'discrete']

        if solver == 'rk':
            from seeds_lib.methods.rk_solver import RK_SOLVER
            return RK_SOLVER(net, order=order, sigma=sigma, sigma_deriv=sigma_deriv, s_deriv=s_deriv, s=s, class_labels=class_labels, c2=c2)
        
        elif solver == 'etd-erk':
            from seeds_lib.methods.etd_erk import ETD_ERK_SOLVER
            return ETD_ERK_SOLVER(net, lamb=lamb, lamb_inv=lamb_inv, class_labels=class_labels, sigma=sigma, s=s, order=order, c2=c2, 
                                  noise_pred=noise_pred, butcher_type=butcher_type)
        
        elif solver == 'etd-serk':
            from seeds_lib.methods.etd_serk import ETD_SERK_SOLVER
            return ETD_SERK_SOLVER(net, lamb=lamb, lamb_inv=lamb_inv, class_labels=class_labels, sigma=sigma, s=s, order=order, c2=c2, 
                                   noise_pred=noise_pred, noise_type=noise_type, butcher_type=butcher_type)
        
        elif solver == 'serk-heun':
            from seeds_lib.methods.edm_serk import SERK_EDM_SOLVER
            return SERK_EDM_SOLVER(net, beta_min=beta_min, beta_max=beta_max, class_labels=class_labels, sigma=sigma, s=s, order=order, c2=c2,
                                  noise_pred=noise_pred)

        else:
            raise NameError('Solver method {} not yet implemented in SEEDS lib.'.format(solver))
