# 1st edit by https://github.com/comfyanonymous/ComfyUI
# 2nd edit by Forge Official


import torch
import ldm_patched.modules.model_management
import contextlib

from modules_forge import stream


# https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/14855/files
gc = {}


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
    context = contextlib.nullcontext
    signal = None

    if stream.using_stream:
        context = stream.stream_context()

    with context(stream.mover_stream):
        bias = None
        non_blocking = ldm_patched.modules.model_management.device_supports_non_blocking(input.device)
        if s.bias is not None:
            bias = s.bias.to(device=input.device, dtype=input.dtype, non_blocking=non_blocking)
        weight = s.weight.to(device=input.device, dtype=input.dtype, non_blocking=non_blocking)

        if stream.using_stream:
            signal = stream.mover_stream.record_event()
    return weight, bias, signal


@contextlib.contextmanager
def main_thread_worker(weight, bias, signal):
    if not stream.using_stream or signal is None:
        yield
        return

    with stream.stream_context()(stream.current_stream):
        stream.current_stream.wait_event(signal)
        yield
        finished_signal = stream.current_stream.record_event()
        size = weight.element_size() * weight.nelement()
        if bias is not None:
            size += bias.element_size() * bias.nelement()
        gc[id(finished_signal)] = (weight, bias, finished_signal, size)

    overhead = sum([l for k, (w, b, s, l) in gc.items()])

    if overhead > 512 * 1024 * 1024:
        stream.mover_stream.synchronize()
        stream.current_stream.synchronize()

    garbage = []
    for k, (w, b, s, l) in gc.items():
        if s.query():
            garbage.append(k)

    for k in garbage:
        del gc[k]
    return


def cleanup_cache():
    global gc

    if stream.current_stream is not None:
        with stream.stream_context()(stream.current_stream):
            for k, (w, b, s, l) in gc.items():
                stream.current_stream.wait_event(s)
        stream.current_stream.synchronize()

    gc.clear()

    if stream.mover_stream is not None:
        stream.mover_stream.synchronize()
    return


class disable_weight_init:
    class Linear(torch.nn.Linear):
        ldm_patched_cast_weights = False
        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias, signal = cast_bias_weight(self, input)
            with main_thread_worker(weight, bias, signal):
                return torch.nn.functional.linear(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv2d(torch.nn.Conv2d):
        ldm_patched_cast_weights = False
        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias, signal = cast_bias_weight(self, input)
            with main_thread_worker(weight, bias, signal):
                return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv3d(torch.nn.Conv3d):
        ldm_patched_cast_weights = False
        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias, signal = cast_bias_weight(self, input)
            with main_thread_worker(weight, bias, signal):
                return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class GroupNorm(torch.nn.GroupNorm):
        ldm_patched_cast_weights = False
        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias, signal = cast_bias_weight(self, input)
            with main_thread_worker(weight, bias, signal):
                return torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)


    class LayerNorm(torch.nn.LayerNorm):
        ldm_patched_cast_weights = False
        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias, signal = cast_bias_weight(self, input)
            with main_thread_worker(weight, bias, signal):
                return torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

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
