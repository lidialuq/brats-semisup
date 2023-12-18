import time
import sys

from typing import List, Callable, Union, Dict, Optional
from abc import abstractmethod
from pathlib import Path
from datetime import datetime

# from logger import TensorboardWriter

import wandb
import torch
import py3nvml

import numpy as np

from .logger import get_logger


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
                 loss_function: callable,
                 config: dict,
                 project: str,
                 semi_optimizer: Optional[torch.optim.Optimizer] = None,
                 semi_lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 seed: Optional[int] = None,
                 device: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 ):
        """
        Args:
            model (torch.nn.Module): The model to be trained
            loss_function (MultiLoss): The loss function or loss function class
            optimizer (torch.optim): torch.optim, i.e., the optimizer class
            config (dict): dict of configs
            lr_scheduler (torch.optim.lr_scheduler): pytorch lr_scheduler for manipulating the learning rate
            seed (int): integer seed to enforce non stochasticity,
            device (str): string of the device to be trained on, e.g., "cuda:0"
        """

        # Reproducibility is a good thing
        if isinstance(seed, int):
            torch.manual_seed(seed)

        self.config = config
        self.run = wandb.init(
            config=config,
            entity='crai',
            project=project,
            tags=tags, 
            save_code=True, 
            reinit=True, 
            name=config['name'], 
            mode="online",
            )

        wandb.run.log_code("..")

        self.logger = get_logger(name=__name__)
        self.iterative = config['iterative']

        # setup GPU device if available, move model into configured device
        if device is None:
            self.device, device_ids = self.prepare_device(config['n_gpu'])
        else:
            self.device = torch.device(device)
            device_ids = list()

        self.model = model.to(self.device)
        self.lr_scheduler = lr_scheduler
        self.semi_lr_scheduler = semi_lr_scheduler

        # TODO: Use DistributedDataParallel instead
        if len(device_ids) > 1 and config['n_gpu'] > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.loss_function = loss_function.to(self.device)

        self.optimizer = optimizer
        self.semi_optimizer = semi_optimizer

        self.epochs = int(wandb.config["epochs"])
        self.save_period = int(wandb.config["save_period"])

        self.start_epoch = 1

        self.checkpoint_dir = Path(wandb.config["save_dir"]) / Path(datetime.today().strftime('%Y-%m-%d'))

        self.min_validation_loss = sys.float_info.max  # Minimum validation loss achieved, starting with the larges possible number

    @abstractmethod
    def _train_epoch(self, epoch) -> dict:
        """
        Training logic for an epoch
        Args:
            epoch (int): Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _valid_epoch(self, epoch) -> dict:
        """
        Validation logic after an epoch
        Args:
            epoch (int): Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """

        epochs = self.epochs

        for epoch in range(self.start_epoch, epochs + 1):
            epoch_start_time = time.time()

            loss_dict = self._train_epoch(epoch)
            val_dict = self._valid_epoch(epoch)

            loss_val_dict = {
                **loss_dict, 
                **val_dict, 
                "epoch": epoch,
                }
            wandb.log(loss_val_dict, commit=True)
            
            epoch_end_time = time.time() - epoch_start_time


            if hasattr(self.lr_scheduler, 'get_last_lr'):
                self.logger.info('Current learning rate: {}'.format(self.lr_scheduler.get_last_lr()))
            elif hasattr(self.lr_scheduler, 'get_lr'):
                self.logger.info('Current learning rate: {}'.format(self.lr_scheduler.get_lr()))

            # print logged informations to the screen
            # training loss
            self.logger.info('Mean training loss: {}'.format(loss_dict['loss']))

            for key, valid in val_dict.items():
                self.logger.info('Mean validation {}: {}'.format(str(key), np.mean(valid)))

            if epoch % self.save_period == 0:
                self.save_checkpoint(epoch, best=False)
            if val_dict["val_loss"] < self.min_validation_loss:
                self.min_validation_loss = val_dict["val_loss"]
                self.save_checkpoint(epoch, best=True)

            self.logger.info('-----------------------------------')
        self.save_checkpoint(epoch, best=False)

    def prepare_device(self, n_gpu_use: int):
        """
        setup GPU device if available, move model into configured device
        Args:
            n_gpu_use (int): Number of GPU's to use
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = n_gpu
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu

        free_gpus = py3nvml.get_free_gpus()

        list_ids = [i for i in range(n_gpu) if free_gpus[i]]
        n_gpu_use = min(n_gpu_use, len(list_ids))

        device = torch.device('cuda:{}'.format(list_ids[0]) if n_gpu_use > 0 else 'cpu')
        if device.type == 'cpu':
            self.logger.warning('current selected device is the cpu, you sure about this?')

        self.logger.info('Selected training device is: {}:{}'.format(device.type, device.index))
        self.logger.info('The available gpu devices are: {}'.format(list_ids))

        return device, list_ids

    def save_checkpoint(self, epoch, best: bool = False):
        """
        Saving checkpoints at the given moment
        Args:
            epoch (int), the current epoch of the training
            bool (bool), save as best epoch so far, different naming convention
        """
        arch = type(self.model).__name__

        if not self.semi_optimizer:
            print('saving supervised model')
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.lr_scheduler.state_dict(),
                'config': self.config,
                'loss_func': str(self.loss_function),
                }
        else:
            print('saving semisupervised model')
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'semi_optimizer': self.semi_optimizer.state_dict(),
                'scheduler': self.lr_scheduler.state_dict(),
                'semi_scheduler': self.semi_lr_scheduler.state_dict(),
                'config': self.config,
                'loss_func': str(self.loss_function),
                }

        if best:  # Save best case with different naming convention
            save_path = Path(self.checkpoint_dir) / Path('best_validation')
            filename = str(save_path / 'checkpoint-best.pth')
        else:
            save_path = Path(self.checkpoint_dir) / Path('epoch_' + str(epoch))
            filename = str(save_path / 'checkpoint-epoch{}.pth'.format(epoch))

        save_path.mkdir(parents=True, exist_ok=True)

        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
