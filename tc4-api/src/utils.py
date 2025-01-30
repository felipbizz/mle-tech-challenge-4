import logging
import os
import numpy as np
import torch
from typing import Union
from logging.handlers import TimedRotatingFileHandler

def setLog(logfile: str, 
           format: str = None, 
           level: int = logging.DEBUG, 
           rotation: str = 'd', 
           backupCount: int = 30,
           log_location: str = None):
    
    logger = logging.getLogger(logfile)
    
    if format is None:
        format = os.getenv('DEFAULT_LOG_FORMAT','%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    default_log_level = os.getenv('DEFAULT_LOG_LEVEL','INFO')

    logging.basicConfig(level=default_log_level, format=format)

    if log_location is None:
        log_location = os.getenv('DEFAULT_LOG_LOCATION','logs')
    
    log_location = os.path.join(log_location, logfile)

    if not os.path.exists(log_location):
        os.makedirs(f'{log_location}')

    if rotation not in ['s', 'm', 'h', 'd', 'midnight']:
        rotation = 'midnight'
        logger.error('Período de rotação inválido: %s. Padronizando para rotação diária à meia-noite' % rotation)
    


    logger.setLevel(level)
    formatter = logging.Formatter(format)

    handler = TimedRotatingFileHandler(f'{log_location}/{logfile}.log', when=rotation, backupCount=backupCount)
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger

def wmape(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()

# --------------------------- 

class WMAPE(torch.nn.Module):

    def __init__(self):
        super(WMAPE, self).__init__()
        self.outputsize_multiplier = 1
        self.output_names = [""]
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        return y_hat.squeeze(-1)

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
    ):
        if mask is None:
            mask = torch.ones_like(y_hat)

        num = mask * (y - y_hat).abs()
        den = mask * y.abs()
        return num.sum() / den.sum()