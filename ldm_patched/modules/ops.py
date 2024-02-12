# 1st edit by https://github.com/comfyanonymous/ComfyUI
# 2nd edit by Forge Official


import torch
import ldm_patched.modules.model_management
import contextlib


@contextlib.contextmanager
def use_patched_ops(operations):
    op_names = ['Linear', 'Conv2d', 'Conv3d', 'GroupNorm', 'LayerNorm']
    backups = {op_name: getattr(torch.nn, op_name) for op_name in op_names}

    try:
        for op_name in op_names:
            setattr(torch.nn, op_name, getattr(operations, op_name))

        yield

    finally:
        for op_name in op_names:
            setattr(torch.nn, op_name, backups[op_name])
    return


def cast_bias_weight(s, input):
    non_blocking = ldm_patched.modules.model_management.device_supports_non_blocking(input.device)

    weight = s.weight
    bias = s.bias if s.bias is not None else None

    if s.ldm_patched_cast_weights:
        weight = weight.to(device=input.device, dtype=input.dtype, non_blocking=non_blocking)
        bias = bias.to(device=input.device, dtype=input.dtype, non_blocking=non_blocking) if bias is not None else None

    return weight, bias


class disable_weight_init:
    class Linear(torch.nn.Linear):
        ldm_patched_cast_weights = False
        module_key = ''

        def reset_parameters(self):
            return None

        def forward(self, input, *args, **kwargs):
            weight, bias = cast_bias_weight(self, input)

            module_forward_function = torch.nn.functional.linear
            module_forward_overwrite_function = getattr(self, 'module_forward_overwrite_function', None)

            if module_forward_overwrite_function is not None:
                return module_forward_overwrite_function(module_forward_function, input, weight, bias, self,
                                                         getattr(self, 'root_model', None))
            else:
                return module_forward_function(input, weight, bias)

    class Conv2d(torch.nn.Conv2d):
        ldm_patched_cast_weights = False
        module_key = ''

        def reset_parameters(self):
            return None

        def forward(self, input, *args, **kwargs):
            weight, bias = cast_bias_weight(self, input)

            module_forward_function = self._conv_forward
            module_forward_overwrite_function = getattr(self, 'module_forward_overwrite_function', None)

            if module_forward_overwrite_function is not None:
                return module_forward_overwrite_function(module_forward_function, input, weight, bias, self,
                                                         getattr(self, 'root_model', None))
            else:
                return module_forward_function(input, weight, bias)

    class Conv3d(torch.nn.Conv3d):
        ldm_patched_cast_weights = False
        module_key = ''

        def reset_parameters(self):
            return None

        def forward(self, input, *args, **kwargs):
            weight, bias = cast_bias_weight(self, input)

            module_forward_function = self._conv_forward
            module_forward_overwrite_function = getattr(self, 'module_forward_overwrite_function', None)

            if module_forward_overwrite_function is not None:
                return module_forward_overwrite_function(module_forward_function, input, weight, bias, self,
                                                         getattr(self, 'root_model', None))
            else:
                return module_forward_function(input, weight, bias)

    class GroupNorm(torch.nn.GroupNorm):
        ldm_patched_cast_weights = False
        module_key = ''

        def reset_parameters(self):
            return None

        def forward(self, input, *args, **kwargs):
            weight, bias = cast_bias_weight(self, input)

            module_forward_function = lambda i, w, b: torch.nn.functional.group_norm(i, self.num_groups, w, b, self.eps)
            module_forward_overwrite_function = getattr(self, 'module_forward_overwrite_function', None)

            if module_forward_overwrite_function is not None:
                return module_forward_overwrite_function(module_forward_function, input, weight, bias, self,
                                                         getattr(self, 'root_model', None))
            else:
                return module_forward_function(input, weight, bias)

    class LayerNorm(torch.nn.LayerNorm):
        ldm_patched_cast_weights = False
        module_key = ''

        def reset_parameters(self):
            return None

        def forward(self, input, *args, **kwargs):
            weight, bias = cast_bias_weight(self, input)

            module_forward_function = lambda i, w, b: torch.nn.functional.layer_norm(i, self.normalized_shape, w, b, self.eps)
            module_forward_overwrite_function = getattr(self, 'module_forward_overwrite_function', None)

            if module_forward_overwrite_function is not None:
                return module_forward_overwrite_function(module_forward_function, input, weight, bias, self,
                                                         getattr(self, 'root_model', None))
            else:
                return module_forward_function(input, weight, bias)

    @classmethod
    def conv_nd(s, dims, *args, **kwargs):
        if dims == 2:
            return s.Conv2d(*args, **kwargs)
        elif dims == 3:
            return s.Conv3d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")


class manual_cast(disable_weight_init):
    class Linear(disable_weight_init.Linear):
        ldm_patched_cast_weights = True

    class Conv2d(disable_weight_init.Conv2d):
        ldm_patched_cast_weights = True

    class Conv3d(disable_weight_init.Conv3d):
        ldm_patched_cast_weights = True

    class GroupNorm(disable_weight_init.GroupNorm):
        ldm_patched_cast_weights = True

    class LayerNorm(disable_weight_init.LayerNorm):
        ldm_patched_cast_weights = True
