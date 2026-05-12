"""PTQ4ViT quantized conv2d layers.

Copied from references/ptq4vit/code/quant_layers/conv.py with:
- Import path fixes
- .cuda() replaced with device-aware .to() calls
- Only includes classes used by PTQ4ViT config: MinMaxQuantConv2d, PTQSLQuantConv2d, ChannelwiseBatchingQuantConv2d
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product


class MinMaxQuantConv2d(nn.Conv2d):
    def __init__(self,in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',mode='raw',w_bit=8,a_bit=8,bias_bit=None):
        super().__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode)
        self.n_calibration_step=2
        self.mode=mode
        self.w_bit=w_bit
        self.a_bit=a_bit
        self.bias_bit=bias_bit
        assert bias_bit is None,"No support bias bit now"
        self.w_interval=None
        self.a_interval=None
        self.bias_interval=None
        self.raw_input=None
        self.raw_out=None
        self.metric=None
        self.next_nodes=[]
        self.w_qmax=2**(self.w_bit-1)
        self.a_qmax=2**(self.a_bit-1)

    def forward(self, x):
        if self.mode=='raw':
            out=F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        elif self.mode=="quant_forward":
            out=self.quant_forward(x)
        elif self.mode=="calibration_step1":
            out=self.calibration_step1(x)
        elif self.mode=="calibration_step2":
            out=self.calibration_step2(x)
        else:
            raise NotImplementedError
        return out

    def quant_weight_bias(self):
        w=(self.weight/self.w_interval).round_().clamp_(-self.w_qmax,self.w_qmax-1)
        w_sim=w.mul_(self.w_interval)
        if self.bias is not None:
            return w_sim,self.bias
        else:
            return w_sim,None

    def quant_input(self,x):
        x_sim=(x/self.a_interval).round_().clamp_(-self.a_qmax,self.a_qmax-1)
        x_sim.mul_(self.a_interval)
        return x_sim

    def quant_forward(self,x):
        assert self.calibrated is not None,f"You should run calibrate_forward before run quant_forward for {self}"
        w_sim,bias_sim=self.quant_weight_bias()
        x_sim=self.quant_input(x)
        out=F.conv2d(x_sim, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups)
        return out

    def calibration_step1(self,x):
        out=F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self.raw_input=x.cpu().detach()
        self.raw_out=out.cpu().detach()
        return out

    def calibration_step2(self,x):
        self.w_interval=(self.weight.data.abs().max()/(self.w_qmax-0.5)).detach()
        self.a_interval=(x.abs().max()/(self.a_qmax-0.5)).detach()
        self.calibrated=True
        out=self.quant_forward(x)
        return out


class PTQSLQuantConv2d(MinMaxQuantConv2d):
    def __init__(self, in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',mode='raw',w_bit=8,a_bit=8,bias_bit=None,
        metric="L2_norm", search_round=1, eq_alpha=0.1, eq_beta=2, eq_n=100, parallel_eq_n=10,
        n_V=1, n_H=1, init_layerwise=False):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, mode=mode, w_bit=w_bit, a_bit=a_bit, bias_bit=bias_bit)
        self.metric = metric
        self.search_round = search_round
        self.eq_alpha = eq_alpha
        self.eq_beta = eq_beta
        self.eq_n = eq_n
        self.parallel_eq_n = parallel_eq_n
        self.n_H = n_H
        self.n_V = n_V
        self.init_layerwise = init_layerwise
        self.raw_grad = None

    def _get_similarity(self, tensor_raw, tensor_sim, metric=None, dim=-1):
        if metric == "cosine":
            similarity = F.cosine_similarity(tensor_raw, tensor_sim, dim=dim)
        else:
            if metric == "L1_norm":
                similarity = -torch.abs(tensor_raw - tensor_sim)
            elif metric == "L2_norm":
                similarity = -(tensor_raw - tensor_sim) ** 2
            elif metric == "linear_weighted_L2_norm":
                similarity = -tensor_raw.abs() * (tensor_raw - tensor_sim) ** 2
            elif metric == "square_weighted_L2_norm":
                similarity = -(tensor_raw * (tensor_raw - tensor_sim)) ** 2
            elif metric == "hessian":
                raw_grad = self.raw_grad.reshape_as(tensor_raw)
                similarity = -(raw_grad * (tensor_raw - tensor_sim)) ** 2
            else:
                raise NotImplementedError(f"metric {metric} not implemented!")
            similarity = torch.mean(similarity, dim=dim)
        return similarity


class ChannelwiseBatchingQuantConv2d(PTQSLQuantConv2d):
    def __init__(self, in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',mode='raw',w_bit=8,a_bit=8,bias_bit=None,
        metric="L2_norm", search_round=1, eq_alpha=0.1, eq_beta=2, eq_n=100, parallel_eq_n=10,
        n_V=1, n_H=1, init_layerwise=False):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode,
                         mode=mode, w_bit=w_bit, a_bit=a_bit, bias_bit=bias_bit, metric=metric, search_round=search_round, eq_alpha=eq_alpha, eq_beta=eq_beta, eq_n=eq_n, parallel_eq_n=parallel_eq_n,
                         n_V=n_V, n_H=n_H, init_layerwise=init_layerwise)
        self.n_V = self.out_channels
        self.n_H = 1

    @property
    def _device(self):
        return self.weight.device

    def _initialize_calib_parameters(self):
        self.calib_size = int(self.raw_input.shape[0])
        self.calib_batch_size = int(self.raw_input.shape[0])
        while True:
            numel = (2*(self.raw_input.numel()+self.raw_out.numel())/self.calib_size*self.calib_batch_size)
            self.parallel_eq_n = int((15*1024*1024*1024/4)//numel)
            if self.parallel_eq_n <= 1:
                self.calib_need_batching = True
                self.calib_batch_size //= 2
            else:
                break

    def _initialize_intervals(self):
        if self.init_layerwise:
            self.w_interval=((self.weight.abs().max())/(self.w_qmax-0.5)).view(1,1,1,1).repeat(self.out_channels,1,1,1)
        else:
            self.w_interval=((self.weight.abs().amax([1,2,3],keepdim=True))/(self.w_qmax-0.5))
        tmp_a_intervals = []
        for b_st in range(0,self.calib_size,self.calib_batch_size):
            b_ed = min(self.calib_size, b_st+self.calib_batch_size)
            x_ = self.raw_input[b_st:b_ed].to(self._device)
            a_interval_=(x_.abs().max()/(self.a_qmax-0.5)).detach().view(1,1)
            tmp_a_intervals.append(a_interval_)
        self.a_interval = torch.cat(tmp_a_intervals, dim=1).amax(dim=1, keepdim=False)

    def _get_similarity(self, tensor_raw, tensor_sim, metric=None, raw_grad=None):
        if metric == "cosine":
            b, parallel_eq_n, oc = tensor_sim.shape[0], tensor_sim.shape[1], tensor_sim.shape[2]
            similarity = F.cosine_similarity(tensor_raw.view(b,1,oc,-1), tensor_sim.view(b,parallel_eq_n,oc,-1), dim=-1).view(b,parallel_eq_n,oc,1,1)
        else:
            if metric == "L1_norm":
                similarity = -torch.abs(tensor_raw - tensor_sim)
            elif metric == "L2_norm":
                similarity = -(tensor_raw - tensor_sim) ** 2
            elif metric == "linear_weighted_L2_norm":
                similarity = -tensor_raw.abs() * (tensor_raw - tensor_sim) ** 2
            elif metric == "square_weighted_L2_norm":
                similarity = -(tensor_raw * (tensor_raw - tensor_sim)) ** 2
            elif metric == "hessian":
                assert raw_grad is not None, f"raw_grad is None in _get_similarity!"
                raw_grad = raw_grad.reshape_as(tensor_raw)
                similarity = -(raw_grad * (tensor_raw - tensor_sim)) ** 2
            else:
                raise NotImplementedError(f"metric {metric} not implemented!")
        return similarity

    def _search_best_w_interval(self, weight_interval_candidates):
        batch_similarities = []
        for b_st in range(0,self.calib_size,self.calib_batch_size):
            b_ed = min(self.calib_size, b_st+self.calib_batch_size)
            x = self.raw_input[b_st:b_ed].to(self._device)
            raw_out = self.raw_out[b_st:b_ed].to(self._device).unsqueeze(1)
            raw_grad = self.raw_grad[b_st:b_ed].to(self._device)
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                cur_w_interval = weight_interval_candidates[p_st:p_ed]
                oc,ic,kw,kh = self.weight.data.shape
                w_sim = self.weight.unsqueeze(0)
                w_sim = (w_sim/cur_w_interval).round_().clamp_(-self.w_qmax,self.w_qmax-1).mul_(cur_w_interval)
                w_sim = w_sim.reshape(-1,ic,kw,kh)
                bias_sim = self.bias.repeat(p_ed-p_st) if self.bias is not None else None
                x_sim = self.quant_input(x) if self.a_bit < 32 else x
                out_sim = F.conv2d(x_sim, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups)
                out_sim = torch.cat(torch.chunk(out_sim.unsqueeze(1), chunks=p_ed-p_st, dim=2), dim=1)
                similarity = self._get_similarity(raw_out, out_sim, self.metric, raw_grad)
                similarity = torch.mean(similarity, [3,4])
                similarity = torch.sum(similarity, dim=0, keepdim=True)
                similarities.append(similarity)
            similarities = torch.cat(similarities, dim=1)
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=0).sum(dim=0, keepdim=False)
        best_index = batch_similarities.argmax(dim=0).reshape(1,-1,1,1,1)
        self.w_interval = torch.gather(weight_interval_candidates,dim=0,index=best_index).squeeze(dim=0)

    def _search_best_a_interval(self, input_interval_candidates):
        batch_similarities = []
        for b_st in range(0,self.calib_size,self.calib_batch_size):
            b_ed = min(self.calib_size, b_st+self.calib_batch_size)
            x = self.raw_input[b_st:b_ed].to(self._device)
            raw_out = self.raw_out[b_st:b_ed].to(self._device).unsqueeze(1)
            raw_grad = self.raw_grad[b_st:b_ed].to(self._device)
            similarities = []
            for p_st in range(0,self.eq_n,self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                cur_a_interval = input_interval_candidates[p_st:p_ed]
                w_sim, bias_sim = self.quant_weight_bias()
                B,ic,iw,ih = x.shape
                x_sim=x.unsqueeze(0)
                x_sim=(x_sim/(cur_a_interval)).round_().clamp_(-self.a_qmax,self.a_qmax-1)*(cur_a_interval)
                x_sim=x_sim.view(-1,ic,iw,ih)
                out_sim = F.conv2d(x_sim, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups)
                out_sim = torch.cat(torch.chunk(out_sim.unsqueeze(0), chunks=p_ed-p_st, dim=1), dim=0)
                out_sim = out_sim.transpose_(0, 1)
                similarity = self._get_similarity(raw_out, out_sim, self.metric, raw_grad=raw_grad)
                similarity = torch.mean(similarity, dim=[2,3,4])
                similarity = torch.sum(similarity, dim=0, keepdim=True)
                similarities.append(similarity)
            similarities = torch.cat(similarities, dim=1)
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=0).sum(dim=0, keepdim=False)
        a_best_index = batch_similarities.argmax(dim=0).view(1,1,1,1,1)
        self.a_interval = torch.gather(input_interval_candidates,dim=0,index=a_best_index).squeeze()

    def quant_weight_bias(self):
        w_sim = (self.weight/self.w_interval).round_().clamp(-self.w_qmax,self.w_qmax-1).mul_(self.w_interval)
        return w_sim, self.bias

    def quant_forward(self, x):
        assert self.calibrated is not None,f"You should run calibrate_forward before run quant_forward for {self}"
        w_sim,bias_sim=self.quant_weight_bias()
        x_sim=self.quant_input(x) if self.a_bit < 32 else x
        out=F.conv2d(x_sim, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups)
        return out

    def calibration_step2(self):
        self._initialize_calib_parameters()
        self._initialize_intervals()
        weight_interval_candidates = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).to(self._device).view(-1,1,1,1,1) * self.w_interval.unsqueeze(0)
        input_interval_candidates =  torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).to(self._device).view(-1,1,1,1,1) * self.a_interval
        for e in range(self.search_round):
            self._search_best_w_interval(weight_interval_candidates)
            if self.a_bit < 32:
                self._search_best_a_interval(input_interval_candidates)
        self.calibrated = True
        del self.raw_input, self.raw_out, self.raw_grad
