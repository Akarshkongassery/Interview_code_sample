import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required
import collections
import logging
import math
import sys
import copy

import torch
import torch.distributed as dist
import functools
import torch.distributed as dist
from torch.optim.optimizer import Optimizer, required
#import communicate, flatten_tensors, unflatten_tensors
import threading

def scaled_sign(x):
    
    return x.norm(p=1) / x.nelement() * torch.sign(x)


def unscaled_sign(x):
    
    return torch.sign(x)


class StepAHeadErrorFeedbackSGD(Optimizer):
    

    def __init__(self, params, lr=required, momentum=0.9, dampening=0,
                 weight_decay=0, nesterov=False, comp='scaled_sign', memory=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if comp == 'scaled_sign':
            comp = scaled_sign
        elif comp == 'sign':
            comp = unscaled_sign
        elif not callable(comp) and comp is not None:
            raise ValueError("Invalid comp value: {} (must be callable or None)".format(comp))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        comp=comp, memory=memory)

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(StepAHeadErrorFeedbackSGD, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['memory'] = torch.zeros_like(p.data)

                # To compute the gradients norms ratios over time
                param_state['dim'] = p.nelement()
                param_state['gradient'] = None
                param_state['corrected_gradient'] = None

    def __setstate__(self, state):
        super(StepAHeadErrorFeedbackSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            comp = group['comp']
            memory = group['memory']

            for p in group['params']:
                param_state = self.state[p]
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # d_p corresponds to g in alg. 1 from the paper.
                param_state['gradient'] = d_p  # Save the gradient so its norm can be computed later

                d_p = group['lr'] * d_p
                corrected_gradient = param_state['memory'] + d_p

                # Save the corrected gradient to compute the norms
                param_state['corrected_gradient'] = corrected_gradient

                if comp is not None:
                    corrected_gradient = comp(corrected_gradient)

                
                if comp == unscaled_sign:
                    corrected_gradient = group['lr'] * corrected_gradient

                if memory:
                    param_state['memory'] = param_state['memory'] + d_p - corrected_gradient

                p.data.add_(-1, corrected_gradient)

        return loss

    def memory_norm(self):
        
        norm = 0
        for group in self.param_groups:
            for p in group['params']:
                n = p.norm()
                norm += float(n * n)
        return np.sqrt(norm)

    def gradient_norms_ratio(self):
        res = []
        sum_l2_norms = 0
        sum_normalized_l1_norm = 0
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                n1 = param_state['gradient'].norm(p=1)
                n2 = param_state['gradient'].norm(p=2)
                d = param_state['dim']
                sum_l2_norms += n2*n2
                sum_normalized_l1_norm += n1*n1/d
                res.append(n1*n1/n2/n2/d)
        
        res.append(sum_normalized_l1_norm/sum_l2_norms)
        return np.array(res)

    def corrected_gradient_norms_ratio(self):
        res = []
        sum_l2_norms = 0
        sum_normalized_l1_norm = 0
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                n1 = param_state['corrected_gradient'].norm(p=1)
                n2 = param_state['corrected_gradient'].norm(p=2)
                d = param_state['dim']
                sum_l2_norms += n2*n2
                sum_normalized_l1_norm += n1*n1/d
                res.append(n1*n1/n2/n2/d)
        
        res.append(sum_normalized_l1_norm/sum_l2_norms)
        return np.array(res)

    def params_dims(self):
        res = []
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                d = param_state['dim']
                res.append(d)
        return np.array(res)


def flatten_tensors(tensors):
    
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat


def unflatten_tensors(flat, tensors):
    
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)

def communicate(tensors, communication_op, attention=False):
   
    flat_tensor = flatten_tensors(tensors)
    communication_op(tensor=flat_tensor)
    if attention:
        return tensors/flat_tensor
    for f, t in zip(unflatten_tensors(flat_tensor, tensors), tensors):
        t.set_(f)


def SyncAllreduce(model, rank, size):
    
    communication_op = functools.partial(dist.all_reduce)
    params_list = []
    for param in model.parameters():
        param.data.div_(float(size))
        params_list.append(param.data)

    communicate(params_list, communication_op)





class LocalErrorSGD(Optimizer):
    
    def __init__(self, params, gmf, tau, size, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, variance=0):
        
        self.gmf = gmf
        self.size = size
        self.comm_buf = []
        self.itr = 0
        self.cp = tau


        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, variance=variance)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(LocalErrorSGD, self).__init__(params, defaults)


        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]    
                buf = param_state['anchor_model'] = torch.clone(p.data).detach()
                self.comm_buf.append(buf)

    def __setstate__(self, state):
        super(LocalSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]

                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss

    def average(self):
        step_flag = (self.itr != 0 and self.itr % self.cp == 0)
        self.itr += 1
        if step_flag:
            if self.gmf == 0:
                # simple average
                param_list = []
                for group in self.param_groups:
                    for p in group['params']:
                        p.data.div_(self.size)
                        param_list.append(p.data)
                communicate(param_list, dist.all_reduce)
            else:
                # simple average + global momentum
                for group in self.param_groups:
                    lr = group['lr']
                    for p in group['params']:
                        param_state = self.state[p]
                        old_data = param_state['anchor_model']

                        if 'global_momentum_buffer' not in param_state:
                            buf = param_state['global_momentum_buffer'] = torch.clone(p.data).detach()
                            buf.sub_(old_data)
                            buf.div_(-lr)
                        else:
                            buf = param_state['global_momentum_buffer']
                            buf.mul_(self.gmf).sub_(1/lr, p.data).add_(1/lr, old_data)

                        old_data.add_(-lr, buf)
                        old_data.div_(self.size)

                communicate(self.comm_buf, dist.all_reduce)
                for group in self.param_groups:
                    for p in group['params']:
                        param_state = self.state[p]
                        old_data = param_state['anchor_model']
                        p.data.copy_(old_data)