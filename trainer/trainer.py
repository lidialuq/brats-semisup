from typing import Callable, Dict, Union, Tuple, List, Optional
from collections import defaultdict
import math

import numpy as np
import torch
import monai
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from .base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
                 loss_function: callable,
                 config: dict,
                 project: str,
                 data_loader: torch.utils.data.dataloader.DataLoader,
                 valid_data_loader: torch.utils.data.dataloader.DataLoader,
                 seed: int = None,
                 device: str = None,
                 tags: Optional[List[str]] = None,
                 mixed_precision: bool = True,
                 semi_optimizer: Optional[torch.optim.Optimizer] = None,
                 semi_lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 semi_supervised_loader: Optional[torch.utils.data.dataloader.DataLoader] = None,
                 phi: Optional[float] = 1.,
                 wait_epoch: int = 0,
                 ):

        super().__init__(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_function=loss_function,
            config=config,
            seed=seed,
            device=device,
            tags=tags,
            project=project,
            )
            

        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.semi_supervised_loader = semi_supervised_loader


        if all((semi_optimizer is not None, semi_lr_scheduler is not None, semi_supervised_loader is not None)):
            self.logger.info('All requirmentes met, semi-supervised learning enabled')
            self.logger.info('Semi-Supervised samples are weighted phi={}'.format(phi))

            self.semi_supervised = True
            self.supervised_cps = False
            self.phi = phi
            self.wait_epoch = wait_epoch
            self.semi_optimizer = semi_optimizer
            self.semi_lr_scheduler = semi_lr_scheduler
            self.semi_supervised_loader = semi_supervised_loader
        elif all((semi_optimizer is not None, semi_lr_scheduler is not None, semi_supervised_loader is None)):
            self.logger.info('Semi_supervised_loader is None, semi-supervised learning disabled but cps regularization enabled')

            self.semi_supervised = False
            self.supervised_cps = True
            self.phi = phi
            self.wait_epoch = wait_epoch
            self.semi_optimizer = semi_optimizer
            self.semi_lr_scheduler = semi_lr_scheduler
            self.semi_supervised_loader = semi_supervised_loader
        else:
            self.semi_supervised = False
            self.supervised_cps = False
        
        self.mixed_precision = mixed_precision

        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)

        self.inputs_pr_iteration = int(config['inputs_pr_iteration'])

        self.batch_size = data_loader.batch_size
        self.len_epoch = len(data_loader) if not self.iterative else self.inputs_pr_iteration
        self.len_valid = len(valid_data_loader)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.dice_metric = monai.metrics.DiceMetric(include_background=False, reduction="mean_batch")

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        losses = list()
        sup_losses = list()

        if self.semi_supervised:
            semi_loader = iter(self.semi_supervised_loader)
            loss_sup_1 = list()
            loss_sup_2 = list()
            cps_loss_list = list()

        for batch_idx, inp in tqdm(enumerate(self.data_loader), total=self.len_epoch):
            data, target = inp['image'].to(self.device), inp['label'].to(self.device)

            if self.semi_supervised:
                semi_data = semi_loader.next()['image']
                semi_data = semi_data.to(self.device)
                self.semi_optimizer.zero_grad()

            self.optimizer.zero_grad()

            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    if self.semi_supervised:
                        loss, sup_loss_1, sup_loss_2, cps_loss = self._semi_supervised(data, semi_data, target, epoch)
                    elif self.supervised_cps:
                        loss, sup_loss_1, sup_loss_2, cps_loss = self._supervised_cps(data, target, epoch)
                    else:
                        out = self.model(data)
                        loss = self._loss(out, target)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                if self.semi_supervised:
                    self.scaler.step(self.semi_optimizer)

                self.scaler.update()
            else:
                if self.semi_supervised:
                    loss, sup_loss_1, sup_loss_2, cps_loss = self._semi_supervised(data, semi_data, target, epoch)
                elif self.supervised_cps:
                    loss, sup_loss_1, sup_loss_2, cps_loss = self._supervised_cps(data, target, epoch)  
                else:
                    out = self.model(data)
                    loss = self._loss(out, target)

                loss.backward()
                self.optimizer.step()
                if self.semi_supervised:
                    self.semi_optimizer.step()

            loss = loss.item()  # Detach loss from comp graph and moves it to the cpu
            losses.append(loss)

            if self.semi_supervised:
                loss_sup_1.append(sup_loss_1.item())
                loss_sup_2.append(sup_loss_2.item())
                cps_loss_list.append(cps_loss.item())

            if batch_idx >= self.inputs_pr_iteration and self.iterative:
                break

        self.optimizer.zero_grad()
        if self.semi_supervised:
            self.semi_optimizer.zero_grad()
        
        self.lr_scheduler.step()
        if self.semi_supervised:
            self.semi_lr_scheduler.step()
        
        if self.semi_supervised:
            return {
                "loss": np.mean(np.array(losses)),
                "loss sup 1": np.mean(np.array(loss_sup_1)),
                "loss sup 2": np.mean(np.array(loss_sup_2)),
                "cps loss": np.mean(np.array(cps_loss_list)),
                }
        return {
            "loss": np.mean(np.array(losses)),
            }

    def _loss(self, out: Union[torch.Tensor, Tuple[torch.Tensor]], target: torch.Tensor):
        if isinstance(out, (list, tuple)):
            output, auxiliary = out
            loss = self.loss_function(output, target)
            auxiliary = auxiliary if isinstance(auxiliary, list) else [auxiliary]
            for aux in auxiliary:
                loss += self.loss_function(aux, target)
            return loss /(len(auxiliary) + 1)
        output = out
        return self.loss_function(output, target)

    def _semi_supervised(self, data, semi_data, target, epoch):
        pred_sup_1 = self.model(data, step=1)
        pred_sup_2 = self.model(data, step=2)

        pred_unsp_1 = self.model(semi_data, step=1)
        pred_unsp_2 = self.model(semi_data, step=2)

        pred_1 = torch.cat([pred_sup_1, pred_unsp_1], dim=0)
        pred_2 = torch.cat([pred_sup_2, pred_unsp_2], dim=0)

        max_1 = torch.argmax(pred_1, dim=1).long()
        max_2 = torch.argmax(pred_2, dim=1).long()
        max_1 = torch.nn.functional.one_hot(max_1, num_classes=4).permute(0, 4, 1, 2, 3).type(torch.float32)
        max_2 = torch.nn.functional.one_hot(max_2, num_classes=4).permute(0, 4, 1, 2, 3).type(torch.float32)
        
        cps_loss = self._loss(pred_1, max_2) + self._loss(pred_2, max_1)

        sup_loss_1 = self._loss(pred_sup_1, target)
        sup_loss_2 = self._loss(pred_sup_2, target)

        if epoch <= self.wait_epoch:

            loss = sup_loss_1 + sup_loss_2
            return loss, sup_loss_1, sup_loss_2, cps_loss

        loss = sup_loss_1 + sup_loss_2 + self.phi*cps_loss
        return loss, sup_loss_1, sup_loss_2, cps_loss

    def _supervised_cps(self, data, target, epoch):
        pred_1 = self.model(data, step=1)
        pred_2 = self.model(data, step=2)

        max_1 = torch.argmax(pred_1, dim=1).long()
        max_2 = torch.argmax(pred_2, dim=1).long()
        max_1 = torch.nn.functional.one_hot(max_1, num_classes=4).permute(0, 4, 1, 2, 3).type(torch.float32)
        max_2 = torch.nn.functional.one_hot(max_2, num_classes=4).permute(0, 4, 1, 2, 3).type(torch.float32)
        
        cps_loss = self._loss(pred_1, max_2) + self._loss(pred_2, max_1)

        sup_loss_1 = self._loss(pred_1, target)
        sup_loss_2 = self._loss(pred_2, target)

        if epoch <= self.wait_epoch:

            loss = sup_loss_1 + sup_loss_2
            return loss, sup_loss_1, sup_loss_2, cps_loss

        loss = sup_loss_1 + sup_loss_2 + self.phi*cps_loss
        return loss, sup_loss_1, sup_loss_2, cps_loss

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """

        self.model.eval()
        losses = list()

        with torch.no_grad():
            for batch_idx, inp in tqdm(enumerate(self.valid_data_loader), total=self.len_valid):
                data, target = inp['image'].to(self.device), inp['label'].to(self.device)

                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        out = self.model(data)
                        loss = self._loss(out, target)
                else:
                    out = self.model(data)
                    loss = self._loss(out, target)

                losses.append(loss.item())

                if isinstance(out, (list, tuple)):
                    out_1, out_2 = out
                    out = (self.logsoftmax(out_1).exp() + self.logsoftmax(out_2).exp())/2
                    out = torch.argmax(out, dim=1)
                    out = torch.nn.functional.one_hot(out, num_classes=4).permute(0, 4, 1, 2, 3).type(torch.float32)
                else:
                    out = torch.argmax(out, dim=1)
                    out = torch.nn.functional.one_hot(out, num_classes=4).permute(0, 4, 1, 2, 3).type(torch.float32)

                self.dice_metric(y_pred=target, y=out)

        dice_metric = self.dice_metric.aggregate()
        self.dice_metric.reset()
    
        dice_edema = dice_metric[0].item()
        dice_enhancing = dice_metric[1].item()
        dice_necrosis = dice_metric[2].item()

        val_dict = {
            'val_loss': np.mean(np.array(losses)),
            'dice_avg': np.mean([dice_edema, dice_enhancing, dice_necrosis]),
            "dice_edema": dice_edema,
            'dice_enhancing': dice_enhancing,
            'dice_necrosis': dice_necrosis,
        }

        return val_dict
