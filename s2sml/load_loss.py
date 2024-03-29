import torch, logging
import math


def load_loss(loss_type):
    logging.info(f"Loading loss {loss_type}")
    if loss_type == "mse":
        return torch.nn.MSELoss()
    if loss_type == "mae":
        return torch.nn.L1Loss()
    if loss_type == "smooth":
        return torch.nn.SmoothL1Loss()
    if loss_type == "logcosh":
        return LogCoshLoss()
    if loss_type == "xtanh":
        return XTanhLoss()
    if loss_type == "xsigmoid":
        return XSigmoidLoss()
    if loss_type == "huber":
        # When delta is set to 1, this loss is equivalent to SmoothL1Loss. 
        # In general, this loss differs from SmoothL1Loss by a factor of delta (AKA beta in Smooth L1).
        return torch.nn.HuberLoss(reduction='mean', delta=0.5)


# https://github.com/tuantle/regression-losses-pytorch?fbclid=IwAR3R5cEsY_dFknBvCP4hHXXxx67WiM9BmRZbY6B55GhXfzjpDpQlEeNH9Tg

#class LogCoshLoss(torch.nn.Module):
#    def __init__(self):
#        super().__init__()

#    def forward(self, y_t, y_prime_t):
#        ey_t = y_t - y_prime_t
#        return torch.mean(torch.log(torch.cosh(ey_t + 1e-08)))
    
    
# https://datascience.stackexchange.com/questions/96271/logcoshloss-on-pytorch

def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
    return torch.mean(_log_cosh(y_pred - y_true))

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)


class XTanhLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t + 1e-08
        return torch.mean(ey_t * torch.tanh(ey_t))


class XSigmoidLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t + 1e-08
        return torch.mean(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t )