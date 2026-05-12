"""PTQ4ViT calibration orchestrators.

Copied from references/ptq4vit/code/utils/quant_calib.py with:
- Import path fixes
- Added device parameter to constructors, replaced .cuda() with .to(device)
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .quant_linear import MinMaxQuantLinear
from .quant_conv import MinMaxQuantConv2d
from .quant_matmul import MinMaxQuantMatMul


def grad_hook(module, grad_input, grad_output):
    if module.raw_grad is None:
        module.raw_grad = []
    module.raw_grad.append(grad_output[0].cpu().detach())

def linear_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = []
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input.append(input[0].cpu().detach())
    module.raw_out.append(output.cpu().detach())

def conv2d_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = []
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input.append(input[0].cpu().detach())
    module.raw_out.append(output.cpu().detach())

def matmul_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = [[],[]]
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input[0].append(input[0].cpu().detach())
    module.raw_input[1].append(input[1].cpu().detach())
    module.raw_out.append(output.cpu().detach())


class QuantCalibrator():
    def __init__(self, net, wrapped_modules, calib_loader, sequential=True, device=None):
        self.net = net
        self.wrapped_modules = wrapped_modules
        self.calib_loader = calib_loader
        self.sequential = sequential
        self.calibrated = False
        self.device = device or torch.device("cuda")

    def sequential_quant_calib(self):
        n_calibration_steps=2
        for step in range(n_calibration_steps):
            print(f"Start calibration step={step+1}")
            for name,module in self.wrapped_modules.items():
                if hasattr(module, "calibrated"):
                    if step == 1:
                        module.mode = "raw"
                    elif step == 2:
                        module.mode = "quant_forward"
                else:
                    module.mode=f'calibration_step{step+1}'
            with torch.no_grad():
                for inp,target in self.calib_loader:
                    inp=inp.to(self.device)
                    self.net(inp)

        for name,module in self.wrapped_modules.items():
            module.mode='quant_forward'
        torch.cuda.empty_cache()
        print("sequential calibration finished")

    def parallel_quant_calib(self):
        print(f"Start calibration step=1")
        for name,module in self.wrapped_modules.items():
            if hasattr(module, "calibrated"):
                module.mode = "raw"
            else:
                module.mode=f'calibration_step1'
        with torch.no_grad():
            for inp,target in self.calib_loader:
                inp=inp.to(self.device)
                self.net(inp)
        for name,module in self.wrapped_modules.items():
            if hasattr(module, "calibrated"):
                continue
            else:
                module.mode=f"calibration_step2"
                with torch.no_grad():
                    if isinstance(module, MinMaxQuantLinear):
                        module.forward(module.raw_input.to(self.device))
                    elif isinstance(module, MinMaxQuantConv2d):
                        module.forward(module.raw_input.to(self.device))
                    elif isinstance(module, MinMaxQuantMatMul):
                        module.forward(module.raw_input[0].to(self.device), module.raw_input[1].to(self.device))
                    torch.cuda.empty_cache()

        for name,module in self.wrapped_modules.items():
            module.mode='quant_forward'
        torch.cuda.empty_cache()
        print("calibration finished")

    def quant_calib(self):
        calib_layers=[]
        for name,module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f"prepare parallel calibration for {calib_layers}")
        if self.sequential:
            self.sequential_quant_calib()
        else:
            self.parallel_quant_calib()
        self.calibrated = True

    def batching_quant_calib(self):
        calib_layers=[]
        for name,module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f"prepare parallel calibration for {calib_layers}")
        print("start calibration")

        q = tqdm(self.wrapped_modules.items(), desc="Brecq")
        for name, module in q:
            q.set_postfix_str(name)

            hooks = []
            if isinstance(module, MinMaxQuantLinear):
                hooks.append(module.register_forward_hook(linear_forward_hook))
            if isinstance(module, MinMaxQuantConv2d):
                hooks.append(module.register_forward_hook(conv2d_forward_hook))
            if isinstance(module, MinMaxQuantMatMul):
                hooks.append(module.register_forward_hook(matmul_forward_hook))

            for inp, target in self.calib_loader:
                for batch_st in range(0,self.calib_loader.batch_size,self.batch_size):
                    self.net.zero_grad()
                    inp_ = inp[batch_st:batch_st+self.batch_size].to(self.device)
                    self.net(inp_)
                del inp, target
                torch.cuda.empty_cache()

            if isinstance(module, MinMaxQuantLinear):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, MinMaxQuantConv2d):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, MinMaxQuantMatMul):
                module.raw_input = [torch.cat(_, dim=0) for _ in module.raw_input]
                module.raw_out = torch.cat(module.raw_out, dim=0)
            for hook in hooks:
                hook.remove()

            with torch.no_grad():
                if isinstance(module, MinMaxQuantLinear):
                    module.calibration_step2()
                if isinstance(module, MinMaxQuantConv2d):
                    module.calibration_step2()
                if isinstance(module, MinMaxQuantMatMul):
                    module.calibration_step2()
                torch.cuda.empty_cache()

            if self.sequential:
                module.mode = "quant_forward"
            else:
                module.mode = "raw"

        for name, module in self.wrapped_modules.items():
            module.mode = "quant_forward"

        print("calibration finished")


class HessianQuantCalibrator(QuantCalibrator):
    def __init__(self, net, wrapped_modules, calib_loader, sequential=False, batch_size=1, device=None):
        super().__init__(net, wrapped_modules, calib_loader, sequential=sequential, device=device)
        self.batch_size = batch_size

    def batching_quant_calib(self):
        calib_layers=[]
        for name,module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f"prepare parallel calibration for {calib_layers}")
        print("start hessian calibration")

        with torch.no_grad():
            for inp, _ in self.calib_loader:
                raw_pred = self.net(inp.to(self.device))
                raw_pred_softmax = F.softmax(raw_pred, dim=-1).detach()
            torch.cuda.empty_cache()

        q = tqdm(self.wrapped_modules.items(), desc="Hessian")
        for name, module in q:
            q.set_postfix_str(name)

            hooks = []
            if isinstance(module, MinMaxQuantLinear):
                hooks.append(module.register_forward_hook(linear_forward_hook))
            if isinstance(module, MinMaxQuantConv2d):
                hooks.append(module.register_forward_hook(conv2d_forward_hook))
            if isinstance(module, MinMaxQuantMatMul):
                hooks.append(module.register_forward_hook(matmul_forward_hook))
            if hasattr(module, "metric"):
                hooks.append(module.register_backward_hook(grad_hook))

            for inp, target in self.calib_loader:
                for batch_st in range(0,self.calib_loader.batch_size,self.batch_size):
                    self.net.zero_grad()
                    inp_ = inp[batch_st:batch_st+self.batch_size].to(self.device)
                    pred = self.net(inp_)
                    loss = F.kl_div(F.log_softmax(pred, dim=-1), raw_pred_softmax[batch_st:batch_st+self.batch_size], reduction="batchmean")
                    loss.backward()
                del inp, target, pred, loss
                torch.cuda.empty_cache()

            if isinstance(module, MinMaxQuantLinear):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, MinMaxQuantConv2d):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, MinMaxQuantMatMul):
                module.raw_input = [torch.cat(_, dim=0) for _ in module.raw_input]
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if hasattr(module, "metric"):
                module.raw_grad = torch.cat(module.raw_grad, dim=0)
            for hook in hooks:
                hook.remove()

            with torch.no_grad():
                if isinstance(module, MinMaxQuantLinear):
                    module.calibration_step2()
                if isinstance(module, MinMaxQuantConv2d):
                    module.calibration_step2()
                if isinstance(module, MinMaxQuantMatMul):
                    module.calibration_step2()
                torch.cuda.empty_cache()

            if self.sequential:
                module.mode = "quant_forward"
            else:
                module.mode = "raw"

        for name, module in self.wrapped_modules.items():
            module.mode = "quant_forward"

        print("hessian calibration finished")
