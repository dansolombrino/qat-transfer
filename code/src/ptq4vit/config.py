"""PTQ4ViT configuration — OmegaConf-driven replacement for reference configs/PTQ4ViT.py."""

from .quant_conv import ChannelwiseBatchingQuantConv2d
from .quant_linear import PTQSLBatchingQuantLinear, PostGeluPTQSLBatchingQuantLinear
from .quant_matmul import PTQSLBatchingQuantMatMul, SoSPTQSLBatchingQuantMatMul


class PTQ4ViTConfig:
    """Builds PTQSL kwargs from a Hydra config and provides a get_module() factory."""

    def __init__(self, cfg):
        bit = int(cfg.bit)

        conv_fc_name_list = ["qconv", "qlinear_qkv", "qlinear_proj", "qlinear_MLP_1", "qlinear_MLP_2", "qlinear_classifier"]
        matmul_name_list = ["qmatmul_qk", "qmatmul_scorev"]

        self.w_bit = {name: bit for name in conv_fc_name_list}
        self.a_bit = {name: bit for name in conv_fc_name_list}
        self.A_bit = {name: bit for name in matmul_name_list}
        self.B_bit = {name: bit for name in matmul_name_list}

        self.no_softmax = bool(cfg.no_softmax)
        self.no_postgelu = bool(cfg.no_postgelu)

        self.ptqsl_conv2d_kwargs = {
            "metric": str(cfg.metric),
            "eq_alpha": float(cfg.eq_alpha),
            "eq_beta": float(cfg.eq_beta),
            "eq_n": int(cfg.eq_n),
            "search_round": int(cfg.search_round),
            "n_V": 1,
            "n_H": 1,
        }
        self.ptqsl_linear_kwargs = {
            "metric": str(cfg.metric),
            "eq_alpha": float(cfg.eq_alpha),
            "eq_beta": float(cfg.eq_beta),
            "eq_n": int(cfg.eq_n),
            "search_round": int(cfg.search_round),
            "n_V": int(cfg.n_V),
            "n_H": int(cfg.n_H),
            "n_a": int(cfg.n_a),
            "bias_correction": bool(cfg.bias_correction),
        }
        self.ptqsl_matmul_kwargs = {
            "metric": str(cfg.metric),
            "eq_alpha": float(cfg.eq_alpha),
            "eq_beta": float(cfg.eq_beta),
            "eq_n": int(cfg.eq_n),
            "search_round": int(cfg.search_round),
            "n_G_A": int(cfg.n_G_A),
            "n_V_A": int(cfg.n_V_A),
            "n_H_A": int(cfg.n_H_A),
            "n_G_B": int(cfg.n_G_B),
            "n_V_B": int(cfg.n_V_B),
            "n_H_B": int(cfg.n_H_B),
        }

    def get_module(self, module_type, *args, **kwargs):
        if module_type == "qconv":
            kwargs.update(self.ptqsl_conv2d_kwargs)
            module = ChannelwiseBatchingQuantConv2d(
                *args, **kwargs,
                w_bit=self.w_bit["qconv"],
                a_bit=32,  # turn off activation quantization for conv
            )
        elif "qlinear" in module_type:
            kwargs.update(self.ptqsl_linear_kwargs)
            if module_type == "qlinear_qkv":
                kwargs["n_V"] *= 3  # q, k, v
                module = PTQSLBatchingQuantLinear(
                    *args, **kwargs,
                    w_bit=self.w_bit[module_type],
                    a_bit=self.a_bit[module_type],
                )
            elif module_type == "qlinear_MLP_2":
                if self.no_postgelu:
                    module = PTQSLBatchingQuantLinear(
                        *args, **kwargs,
                        w_bit=self.w_bit[module_type],
                        a_bit=self.a_bit[module_type],
                    )
                else:
                    module = PostGeluPTQSLBatchingQuantLinear(
                        *args, **kwargs,
                        w_bit=self.w_bit[module_type],
                        a_bit=self.a_bit[module_type],
                    )
            elif module_type == "qlinear_classifier":
                kwargs["n_V"] = 1
                module = PTQSLBatchingQuantLinear(
                    *args, **kwargs,
                    w_bit=self.w_bit[module_type],
                    a_bit=self.a_bit[module_type],
                )
            else:
                module = PTQSLBatchingQuantLinear(
                    *args, **kwargs,
                    w_bit=self.w_bit[module_type],
                    a_bit=self.a_bit[module_type],
                )
        elif "qmatmul" in module_type:
            kwargs.update(self.ptqsl_matmul_kwargs)
            if module_type == "qmatmul_qk":
                module = PTQSLBatchingQuantMatMul(
                    *args, **kwargs,
                    A_bit=self.A_bit[module_type],
                    B_bit=self.B_bit[module_type],
                )
            elif module_type == "qmatmul_scorev":
                if self.no_softmax:
                    module = PTQSLBatchingQuantMatMul(
                        *args, **kwargs,
                        A_bit=self.A_bit[module_type],
                        B_bit=self.B_bit[module_type],
                    )
                else:
                    module = SoSPTQSLBatchingQuantMatMul(
                        *args, **kwargs,
                        A_bit=self.A_bit[module_type],
                        B_bit=self.B_bit[module_type],
                    )
        else:
            raise ValueError(f"Unknown module_type: {module_type}")
        return module
