import os
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, MultiStepLR
from typing import Iterator, Any
from torch.nn import Parameter
from torch.optim.optimizer import Optimizer
import pandas as pd
import numpy as np


def set_num_classes(args):
    if args.data.name in ['citation', 'bitcoin', 'RedditB', 'social']:
        args.model.num_class = 2
    elif args.data.name == 'Reddit5K':
        args.model.num_class = 5
    elif args.data.name == 'Reddit12K':
        args.model.num_class = 11
    elif args.data.name == 'question':
        args.model.num_class = 4
    else:
        raise ValueError(f"Unknown dataset name: {args.data.name}")
    return args
    

def save_all_results_to_csv(all_results, result_file):
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    df = pd.DataFrame(all_results)

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    mean_row = df[numeric_columns].mean()
    std_row = df[numeric_columns].std()

    mean_series = pd.Series({'run_id': 'mean', 'seed': '', 'current_time': '', 'best_checkpoint_path': ''})
    std_series = pd.Series({'run_id': 'std', 'seed': '', 'current_time': '', 'best_checkpoint_path': ''})
    

    for col in numeric_columns:
        mean_series[col] = mean_row[col]
        std_series[col] = std_row[col]

    df = pd.concat([df, pd.DataFrame([mean_series]), pd.DataFrame([std_series])], ignore_index=True)
    df.to_csv(result_file, index=False)
    
def load_config(file_path):
    return OmegaConf.load(file_path)

def create_optimizer(params: Iterator[Parameter], optim_config: Any) -> Optimizer:
    """Creates an optimizer based on the configuration."""
    params = filter(lambda p: p.requires_grad, params)
    lr = float(optim_config.get('lr', 1e-3))
    weight_decay = float(optim_config.get('weight_decay', 1e-5))

    if optim_config.optimizer.lower() == 'adam':
        return Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim_config.optimizer.lower() == 'sgd':
        momentum = optim_config.get('momentum', 0.9)
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optim_config.optimizer.lower() == 'adamw':
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer '{optim_config.optimizer}' is not supported")

def create_scheduler(optimizer: Optimizer, optim_config: Any) -> Any:
    """Creates a learning rate scheduler based on the configuration."""
    if not optim_config.get('lr_scheduler', False):
        return None  # No scheduler if lr_scheduler is False or not specified

    scheduler_type = optim_config.get('scheduler', 'cos').lower()
    max_epochs = optim_config.get('max_epoch', 200)

    if scheduler_type == 'step':
        step_size = optim_config.get('step_size', 100)
        lr_decay = optim_config.get('lr_decay', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=lr_decay)
    elif scheduler_type == 'multistep':
        steps = optim_config.get('steps', [100, 150])
        lr_decay = optim_config.get('lr_decay', 0.1)
        return MultiStepLR(optimizer, milestones=steps, gamma=lr_decay)
    elif scheduler_type == 'cos':
        return CosineAnnealingLR(optimizer, T_max=max_epochs)
    else:
        raise ValueError(f"Scheduler '{scheduler_type}' is not supported")