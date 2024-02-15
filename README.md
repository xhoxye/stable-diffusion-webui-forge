# Stable Diffusion WebUI Forge

Stable Diffusion WebUI Forge 是一个建立在 [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)（基于 [Gradio](https://www.gradio.app/)）之上的平台，旨在简化开发过程、优化资源管理并加快推理速度。

“Forge”这个名字来源于“Minecraft Forge”。本项目的目标是成为 SD WebUI 的 Forge。

相比于原始的 WebUI（针对 SDXL 在 1024px 分辨率下的推理），你可以期待以下性能提升：

1. 如果你使用的是常见的 8GB 显存 GPU，可以预期推理速度（it/s）提高约 **30~45%**，GPU 内存峰值（在任务管理器中显示）将降低约 700MB 至 1.3GB，不会发生内存溢出（OOM）的最大扩散分辨率将提升约 2 倍至 3 倍，而不会内存溢出（OOM） 的最大扩散批次大小将会增加约 4 倍至 6 倍。

2. 若你使用的是较弱的 6GB 显存 GPU，则可以预期推理速度（it/s）提升约 **60~75%**，GPU 内存峰值下降约 800MB 至 1.5GB，不会内存溢出（OOM） 的最大扩散分辨率将提升约 3 倍，且不会内存溢出（OOM） 的最大扩散批次大小会增加约 4 倍。

3. 对于像 4090 这样拥有 24GB 显存的强大 GPU，你可以预期推理速度（it/s）提升约 **3~6%**，GPU 内存峰值将降低约 1GB 至 1.4GB，不会发生内存溢出（OOM） 的最大扩散分辨率将提升约 1.6 倍，而不会内存溢出（OOM） 的最大扩散批次大小将增加约 2 倍。

4. 若你在 SDXL 中使用 ControlNet，不会发生内存溢出（OOM） 的最大 ControlNet 数量将提升约 2 倍，并且使用 SDXL+ControlNet 时的速度将 **提升约 30~45%**。

Forge 引入的一项非常重要的改变是 **Unet 补丁工具**（Unet Patcher）。通过 Unet Patcher，诸如 Self-Attention Guidance、Kohya High Res Fix、FreeU、StyleAlign、Hypertile 等方法都可以在大约 100 行代码内实现。

得益于 Unet Patcher，现在 Forge 支持许多新的功能，包括 SVD、Z123、带蒙版的 Ip-adapter、带蒙版的 controlnet、photomaker 等等。

**从此无需对 UNet 进行猴子补丁（monkeypatch）操作，也不必担心与其他扩展产生冲突！** 

Forge 还增加了一些采样器，包括但不限于 DDPM、DDPM Karras、DPM++ 2M Turbo、DPM++ 2M SDE Turbo、LCM Karras、Euler A Turbo 等（自 1.7.0 版本起，LCM 已经包含在原始 WebUI 中）。

最后，Forge 承诺我们只做分内之事。Forge 将永远不会对用户界面添加不必要的主观性改动。你仍旧在使用 100% Automatic1111 的 WebUI。 
# 安装 Forge

如果你熟悉 Git 并且希望将 Forge 作为 SD-WebUI 的另一个分支进行安装，请参阅 [此处](https://github.com/continue-revolution/sd-webui-animatediff/blob/forge/master/docs/how-to-use.md#you-have-a1111-and-you-know-git)。通过这种方法，你可以复用之前在原始 OG SD-WebUI 中安装的所有 SD 检查点和扩展，但你需要知道自己正在做什么操作。

如果你清楚自己在做什么，可以采用与 SD-WebUI 相同的方法来安装 Forge（安装 Git、Python，克隆 forge 仓库 `https://github.com/lllyasviel/stable-diffusion-webui-forge.git`，然后运行 webui-user.bat）。

**或者，你也可以直接使用一键安装包（其中已包含 Git 和 Python）。**

[>>> 点击此处下载一键安装包 <<<](https://github.com/lllyasviel/stable-diffusion-webui-forge/releases/download/latest/webui_forge_cu121_torch21.7z)

下载后解压，使用 `update.bat` 进行更新，并使用 `run.bat` 来运行程序。请注意，运行 `update.bat` 是很重要的，否则你可能仍在使用带有未修复潜在问题的旧版本。

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/c49bd60d-82bd-4086-9859-88d472582b94)

# 对比截图

我在多个设备上进行了测试，以下是一个典型的在配备 8GB 显存（3070ti 笔记本电脑）并使用 SDXL 的测试结果。

**这是原版 WebUI：**

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/16893937-9ed9-4f8e-b960-70cd5d1e288f)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/7bbc16fe-64ef-49e2-a595-d91bb658bd94)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/de1747fd-47bc-482d-a5c6-0728dd475943)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/96e5e171-2d74-41ba-9dcc-11bf68be7e16)

（平均占用约 7.4GB/8GB，峰值约为 7.9GB/8GB）

**这是 WebUI Forge 版本：**

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/ca5e05ed-bd86-4ced-8662-f41034648e8c)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/3629ee36-4a99-4d9b-b371-12efb260a283)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/6d13ebb7-c30d-4aa8-9242-c0b5a1af8c95)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/c4f723c3-6ea7-4539-980b-0708ed2a69aa)

（平均值和峰值均为 6.3GB/8GB）

可以看到，Forge 不会改变 WebUI 的输出结果。安装 Forge 并不是一个会导致效果破坏的变化。Forge 即使对于最复杂的提示如 `fantasy landscape with a [mountain:lake:0.25] and [an oak:a christmas tree:0.75][ in foreground::0.6][ in background:0.25] [shoddy:masterful:0.5]` 也能完美地保持 WebUI 原有不变的结果。

所有你之前的成果在 Forge 中依然能够正常工作！

# Forge 后端

Forge 后端移除了 WebUI 中所有与资源管理相关的代码，并对所有内容进行了重构。之前所有的命令行标志，如 `medvram, lowvram, medvram-sdxl, precision full, no half, no half vae, attention_xxx, upcast unet` 等，现在均已 **移除**。添加这些标志不会导致错误，但它们现在将不再起任何作用。**我们强烈建议 Forge 用户移除所有命令行标志，并让 Forge 自动决定如何加载模型。**

在没有任何命令行标志的情况下，Forge 可以在 4GB 显存下运行 SDXL，在 2GB 显存下运行 SD1.5。

**唯一可能仍需使用的标志**是 `--always-offload-from-vram`（此选项将会使程序运行速度变 **慢**）。这个选项会让 Forge 始终从显存中卸载模型。当你需要与其他软件同时使用并希望 Forge 使用较少的显存、为其他软件腾出一些显存空间时，或者当你正在使用某些会与 Forge 竞争显存的老版扩展时，以及（非常罕见地）遇到内存溢出（OOM）情况时，这个选项可能会有用。

如果你确实想尝试调整命令行标志，还可以通过以下方式额外控制 GPU：

（极端显存占用情况）

    --always-gpu
    --always-cpu

（少见的注意力机制案例）

    --attention-split
    --attention-quad
    --attention-pytorch
    --disable-xformers
    --disable-attention-upcast

（浮点类型）

    --all-in-fp32
    --all-in-fp16
    --unet-in-bf16
    --unet-in-fp16
    --unet-in-fp8-e4m3fn
    --unet-in-fp8-e5m2
    --vae-in-fp16
    --vae-in-fp32
    --vae-in-bf16
    --clip-in-fp8-e4m3fn
    --clip-in-fp8-e5m2
    --clip-in-fp16
    --clip-in-fp32

（稀有平台支持）

    --directml
    --disable-ipex-hijack
    --pytorch-deterministic

再次强调，除非你非常确定自己确实需要这些选项，否则 Forge 不推荐用户使用任何命令行标志。

# UNet 补丁工具

请注意，Forge 并未使用任何其他软件作为后端支持。其后端的全称是 `Stable Diffusion WebUI with Forge 后端`，或简称为 `Forge 后端`。为了降低开发者的学习成本，API 和 Python 符号设计与先前的软件相似。

现在开发扩展变得超级简单。我们终于有了一个可补丁化的 UNet 结构。

以下是一个仅包含 80 行代码的单个文件示例，用于支持 FreeU 功能：

`extensions-builtin/sd_forge_freeu/scripts/forge_freeu.py`

```python
import torch
import gradio as gr
from modules import scripts


def Fourier_filter(x, threshold, scale):
    x_freq = torch.fft.fftn(x.float(), dim=(-2, -1))
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))
    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)
    crow, ccol = H // 2, W //2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask
    x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real
    return x_filtered.to(x.dtype)


def set_freeu_v2_patch(model, b1, b2, s1, s2):
    model_channels = model.model.model_config.unet_config["model_channels"]
    scale_dict = {model_channels * 4: (b1, s1), model_channels * 2: (b2, s2)}

    def output_block_patch(h, hsp, *args, **kwargs):
        scale = scale_dict.get(h.shape[1], None)
        if scale is not None:
            hidden_mean = h.mean(1).unsqueeze(1)
            B = hidden_mean.shape[0]
            hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
            hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
            hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / \
                          (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)
            h[:, :h.shape[1] // 2] = h[:, :h.shape[1] // 2] * ((scale[0] - 1) * hidden_mean + 1)
            hsp = Fourier_filter(hsp, threshold=1, scale=scale[1])
        return h, hsp

    m = model.clone()
    m.set_model_output_block_patch(output_block_patch)
    return m


class FreeUForForge(scripts.Script):
    def title(self):
        return "FreeU Integrated"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            freeu_enabled = gr.Checkbox(label='Enabled', value=False)
            freeu_b1 = gr.Slider(label='B1', minimum=0, maximum=2, step=0.01, value=1.01)
            freeu_b2 = gr.Slider(label='B2', minimum=0, maximum=2, step=0.01, value=1.02)
            freeu_s1 = gr.Slider(label='S1', minimum=0, maximum=4, step=0.01, value=0.99)
            freeu_s2 = gr.Slider(label='S2', minimum=0, maximum=4, step=0.01, value=0.95)

        return freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        # This will be called before every sampling.
        # If you use highres fix, this will be called twice.
        
        freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2 = script_args

        if not freeu_enabled:
            return

        unet = p.sd_model.forge_objects.unet

        unet = set_freeu_v2_patch(unet, freeu_b1, freeu_b2, freeu_s1, freeu_s2)

        p.sd_model.forge_objects.unet = unet

        # Below codes will add some logs to the texts below the image outputs on UI.
        # The extra_generation_params does not influence results.
        p.extra_generation_params.update(dict(
            freeu_enabled=freeu_enabled,
            freeu_b1=freeu_b1,
            freeu_b2=freeu_b2,
            freeu_s1=freeu_s1,
            freeu_s2=freeu_s2,
        ))

        return
```

其外观如下所示：

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/277bac6e-5ea7-4bff-b71a-e55a60cfc03c)

类似 HyperTile、KohyaHighResFix、SAG（Self-Attention Guidance）这样的组件，都可以在 100 行代码内实现（可查看相关代码）。

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/06472b03-b833-4816-ab47-70712ac024d3)

ControlNets 终于可以被不同的扩展程序调用。

现在实现 Stable Video Diffusion 和 Zero123 也非常简单（请参阅相关代码）。

*Stable Video Diffusion:*

`extensions-builtin/sd_forge_svd/scripts/forge_svd.py`

```python
import torch
import gradio as gr
import os
import pathlib

from modules import script_callbacks
from modules.paths import models_path
from modules.ui_common import ToolButton, refresh_symbol
from modules import shared

from modules_forge.forge_util import numpy_to_pytorch, pytorch_to_numpy
from ldm_patched.modules.sd import load_checkpoint_guess_config
from ldm_patched.contrib.external_video_model import VideoLinearCFGGuidance, SVD_img2vid_Conditioning
from ldm_patched.contrib.external import KSampler, VAEDecode


opVideoLinearCFGGuidance = VideoLinearCFGGuidance()
opSVD_img2vid_Conditioning = SVD_img2vid_Conditioning()
opKSampler = KSampler()
opVAEDecode = VAEDecode()

svd_root = os.path.join(models_path, 'svd')
os.makedirs(svd_root, exist_ok=True)
svd_filenames = []


def update_svd_filenames():
    global svd_filenames
    svd_filenames = [
        pathlib.Path(x).name for x in
        shared.walk_files(svd_root, allowed_extensions=[".pt", ".ckpt", ".safetensors"])
    ]
    return svd_filenames


@torch.inference_mode()
@torch.no_grad()
def predict(filename, width, height, video_frames, motion_bucket_id, fps, augmentation_level,
            sampling_seed, sampling_steps, sampling_cfg, sampling_sampler_name, sampling_scheduler,
            sampling_denoise, guidance_min_cfg, input_image):
    filename = os.path.join(svd_root, filename)
    model_raw, _, vae, clip_vision = \
        load_checkpoint_guess_config(filename, output_vae=True, output_clip=False, output_clipvision=True)
    model = opVideoLinearCFGGuidance.patch(model_raw, guidance_min_cfg)[0]
    init_image = numpy_to_pytorch(input_image)
    positive, negative, latent_image = opSVD_img2vid_Conditioning.encode(
        clip_vision, init_image, vae, width, height, video_frames, motion_bucket_id, fps, augmentation_level)
    output_latent = opKSampler.sample(model, sampling_seed, sampling_steps, sampling_cfg,
                                      sampling_sampler_name, sampling_scheduler, positive,
                                      negative, latent_image, sampling_denoise)[0]
    output_pixels = opVAEDecode.decode(vae, output_latent)[0]
    outputs = pytorch_to_numpy(output_pixels)
    return outputs


def on_ui_tabs():
    with gr.Blocks() as svd_block:
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label='Input Image', source='upload', type='numpy', height=400)

                with gr.Row():
                    filename = gr.Dropdown(label="SVD Checkpoint Filename",
                                           choices=svd_filenames,
                                           value=svd_filenames[0] if len(svd_filenames) > 0 else None)
                    refresh_button = ToolButton(value=refresh_symbol, tooltip="Refresh")
                    refresh_button.click(
                        fn=lambda: gr.update(choices=update_svd_filenames),
                        inputs=[], outputs=filename)

                width = gr.Slider(label='Width', minimum=16, maximum=8192, step=8, value=1024)
                height = gr.Slider(label='Height', minimum=16, maximum=8192, step=8, value=576)
                video_frames = gr.Slider(label='Video Frames', minimum=1, maximum=4096, step=1, value=14)
                motion_bucket_id = gr.Slider(label='Motion Bucket Id', minimum=1, maximum=1023, step=1, value=127)
                fps = gr.Slider(label='Fps', minimum=1, maximum=1024, step=1, value=6)
                augmentation_level = gr.Slider(label='Augmentation Level', minimum=0.0, maximum=10.0, step=0.01,
                                               value=0.0)
                sampling_steps = gr.Slider(label='Sampling Steps', minimum=1, maximum=200, step=1, value=20)
                sampling_cfg = gr.Slider(label='CFG Scale', minimum=0.0, maximum=50.0, step=0.1, value=2.5)
                sampling_denoise = gr.Slider(label='Sampling Denoise', minimum=0.0, maximum=1.0, step=0.01, value=1.0)
                guidance_min_cfg = gr.Slider(label='Guidance Min Cfg', minimum=0.0, maximum=100.0, step=0.5, value=1.0)
                sampling_sampler_name = gr.Radio(label='Sampler Name',
                                                 choices=['euler', 'euler_ancestral', 'heun', 'heunpp2', 'dpm_2',
                                                          'dpm_2_ancestral', 'lms', 'dpm_fast', 'dpm_adaptive',
                                                          'dpmpp_2s_ancestral', 'dpmpp_sde', 'dpmpp_sde_gpu',
                                                          'dpmpp_2m', 'dpmpp_2m_sde', 'dpmpp_2m_sde_gpu',
                                                          'dpmpp_3m_sde', 'dpmpp_3m_sde_gpu', 'ddpm', 'lcm', 'ddim',
                                                          'uni_pc', 'uni_pc_bh2'], value='euler')
                sampling_scheduler = gr.Radio(label='Scheduler',
                                              choices=['normal', 'karras', 'exponential', 'sgm_uniform', 'simple',
                                                       'ddim_uniform'], value='karras')
                sampling_seed = gr.Number(label='Seed', value=12345, precision=0)

                generate_button = gr.Button(value="Generate")

                ctrls = [filename, width, height, video_frames, motion_bucket_id, fps, augmentation_level,
                         sampling_seed, sampling_steps, sampling_cfg, sampling_sampler_name, sampling_scheduler,
                         sampling_denoise, guidance_min_cfg, input_image]

            with gr.Column():
                output_gallery = gr.Gallery(label='Gallery', show_label=False, object_fit='contain',
                                            visible=True, height=1024, columns=4)

        generate_button.click(predict, inputs=ctrls, outputs=[output_gallery])
    return [(svd_block, "SVD", "svd")]


update_svd_filenames()
script_callbacks.on_ui_tabs(on_ui_tabs)
```

请注意，尽管上述代码看起来像是独立的代码片段，但实际上它们会自动卸载/释放任何其他模型。例如，下面是我打开 WebUI、加载 SDXL、生成一张图像，然后转到 SVD（Stable Video Diffusion），并生成图像帧的过程。可以看到 GPU 内存得到了完美的管理，SDXL 被移动至内存中，随后 SVD 被移动至 GPU 中运行。

注意，这一资源管理过程完全自动化。这使得编写扩展程序变得非常简单。

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/de1a2d05-344a-44d7-bab8-9ecc0a58a8d3)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/14bcefcf-599f-42c3-bce9-3fd5e428dd91)

同样地，对于 Zero123：

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/7685019c-7239-47fb-9cb5-2b7b33943285)

### 编写一个简单的 ControlNet 示例：

以下是一个简单的扩展示例，展示了一个完全独立的 ControlNet 通道，它不会与其他任何扩展冲突：

`extensions-builtin/sd_forge_controlnet_example/scripts/sd_forge_controlnet_example.py`

请注意，这个扩展是隐藏的，因为它仅供开发者使用。若要在用户界面中查看，请使用命令行选项 `--show-controlnet-example`。

此示例中的内存优化完全自动进行。你无需关心内存和推理速度的问题，但如果你希望提升性能，可以考虑缓存对象。

```python
# Use --show-controlnet-example to see this extension.

import cv2
import gradio as gr
import torch

from modules import scripts
from modules.shared_cmd_options import cmd_opts
from modules_forge.shared import supported_preprocessors
from modules.modelloader import load_file_from_url
from ldm_patched.modules.controlnet import load_controlnet
from modules_forge.controlnet import apply_controlnet_advanced
from modules_forge.forge_util import numpy_to_pytorch
from modules_forge.shared import controlnet_dir


class ControlNetExampleForge(scripts.Script):
    model = None

    def title(self):
        return "ControlNet Example for Developers"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.HTML('This is an example controlnet extension for developers.')
            gr.HTML('You see this extension because you used --show-controlnet-example')
            input_image = gr.Image(source='upload', type='numpy')
            funny_slider = gr.Slider(label='This slider does nothing. It just shows you how to transfer parameters.',
                                     minimum=0.0, maximum=1.0, value=0.5)

        return input_image, funny_slider

    def process(self, p, *script_args, **kwargs):
        input_image, funny_slider = script_args

        # This slider does nothing. It just shows you how to transfer parameters.
        del funny_slider

        if input_image is None:
            return

        # controlnet_canny_path = load_file_from_url(
        #     url='https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/sai_xl_canny_256lora.safetensors',
        #     model_dir=model_dir,
        #     file_name='sai_xl_canny_256lora.safetensors'
        # )
        controlnet_canny_path = load_file_from_url(
            url='https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/control_v11p_sd15_canny_fp16.safetensors',
            model_dir=controlnet_dir,
            file_name='control_v11p_sd15_canny_fp16.safetensors'
        )
        print('The model [control_v11p_sd15_canny_fp16.safetensors] download finished.')

        self.model = load_controlnet(controlnet_canny_path)
        print('Controlnet loaded.')

        return

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        # This will be called before every sampling.
        # If you use highres fix, this will be called twice.

        input_image, funny_slider = script_args

        if input_image is None or self.model is None:
            return

        B, C, H, W = kwargs['noise'].shape  # latent_shape
        height = H * 8
        width = W * 8
        batch_size = p.batch_size

        preprocessor = supported_preprocessors['canny']

        # detect control at certain resolution
        control_image = preprocessor(
            input_image, resolution=512, slider_1=100, slider_2=200, slider_3=None)

        # here we just use nearest neighbour to align input shape.
        # You may want crop and resize, or crop and fill, or others.
        control_image = cv2.resize(
            control_image, (width, height), interpolation=cv2.INTER_NEAREST)

        # Output preprocessor result. Now called every sampling. Cache in your own way.
        p.extra_result_images.append(control_image)

        print('Preprocessor Canny finished.')

        control_image_bchw = numpy_to_pytorch(control_image).movedim(-1, 1)

        unet = p.sd_model.forge_objects.unet

        # Unet has input, middle, output blocks, and we can give different weights
        # to each layers in all blocks.
        # Below is an example for stronger control in middle block.
        # This is helpful for some high-res fix passes. (p.is_hr_pass)
        positive_advanced_weighting = {
            'input': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            'middle': [1.0],
            'output': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        }
        negative_advanced_weighting = {
            'input': [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25],
            'middle': [1.05],
            'output': [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25]
        }

        # The advanced_frame_weighting is a weight applied to each image in a batch.
        # The length of this list must be same with batch size
        # For example, if batch size is 5, the below list is [0.2, 0.4, 0.6, 0.8, 1.0]
        # If you view the 5 images as 5 frames in a video, this will lead to
        # progressively stronger control over time.
        advanced_frame_weighting = [float(i + 1) / float(batch_size) for i in range(batch_size)]

        # The advanced_sigma_weighting allows you to dynamically compute control
        # weights given diffusion timestep (sigma).
        # For example below code can softly make beginning steps stronger than ending steps.
        sigma_max = unet.model.model_sampling.sigma_max
        sigma_min = unet.model.model_sampling.sigma_min
        advanced_sigma_weighting = lambda s: (s - sigma_min) / (sigma_max - sigma_min)

        # You can even input a tensor to mask all control injections
        # The mask will be automatically resized during inference in UNet.
        # The size should be B 1 H W and the H and W are not important
        # because they will be resized automatically
        advanced_mask_weighting = torch.ones(size=(1, 1, 512, 512))

        # But in this simple example we do not use them
        positive_advanced_weighting = None
        negative_advanced_weighting = None
        advanced_frame_weighting = None
        advanced_sigma_weighting = None
        advanced_mask_weighting = None

        unet = apply_controlnet_advanced(unet=unet, controlnet=self.model, image_bchw=control_image_bchw,
                                         strength=0.6, start_percent=0.0, end_percent=0.8,
                                         positive_advanced_weighting=positive_advanced_weighting,
                                         negative_advanced_weighting=negative_advanced_weighting,
                                         advanced_frame_weighting=advanced_frame_weighting,
                                         advanced_sigma_weighting=advanced_sigma_weighting,
                                         advanced_mask_weighting=advanced_mask_weighting)

        p.sd_model.forge_objects.unet = unet

        # Below codes will add some logs to the texts below the image outputs on UI.
        # The extra_generation_params does not influence results.
        p.extra_generation_params.update(dict(
            controlnet_info='You should see these texts below output images!',
        ))

        return


# Use --show-controlnet-example to see this extension.
if not cmd_opts.show_controlnet_example:
    del ControlNetExampleForge

```

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/822fa2fc-c9f4-4f58-8669-4b6680b91063)


### 添加预处理器

以下代码展示了如何添加一个具有完美内存管理功能的 normalbae 预处理器。

你可以使用任意独立扩展来添加预处理器，且你的预处理器将通过 `modules_forge.shared.preprocessors` 被所有其他扩展读取和使用。

以下是位于 `extensions-builtin\forge_preprocessor_normalbae\scripts\preprocessor_normalbae.py` 中的相关代码：
```python
from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import preprocessor_dir, add_supported_preprocessor
from modules_forge.forge_util import resize_image_with_pad
from modules.modelloader import load_file_from_url

import types
import torch
import numpy as np

from einops import rearrange
from annotator.normalbae.models.NNET import NNET
from annotator.normalbae import load_checkpoint
from torchvision import transforms


class PreprocessorNormalBae(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name = 'normalbae'
        self.tags = ['NormalMap']
        self.model_filename_filters = ['normal']
        self.slider_resolution = PreprocessorParameter(
            label='Resolution', minimum=128, maximum=2048, value=512, step=8, visible=True)
        self.slider_1 = PreprocessorParameter(visible=False)
        self.slider_2 = PreprocessorParameter(visible=False)
        self.slider_3 = PreprocessorParameter(visible=False)
        self.show_control_mode = True
        self.do_not_need_model = False
        self.sorting_priority = 100  # higher goes to top in the list

    def load_model(self):
        if self.model_patcher is not None:
            return

        model_path = load_file_from_url(
            "https://huggingface.co/lllyasviel/Annotators/resolve/main/scannet.pt",
            model_dir=preprocessor_dir)

        args = types.SimpleNamespace()
        args.mode = 'client'
        args.architecture = 'BN'
        args.pretrained = 'scannet'
        args.sampling_ratio = 0.4
        args.importance_ratio = 0.7
        model = NNET(args)
        model = load_checkpoint(model_path, model)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.model_patcher = self.setup_model_patcher(model)

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        input_image, remove_pad = resize_image_with_pad(input_image, resolution)

        self.load_model()

        self.move_all_model_patchers_to_gpu()

        assert input_image.ndim == 3
        image_normal = input_image

        with torch.no_grad():
            image_normal = self.send_tensor_to_model_device(torch.from_numpy(image_normal))
            image_normal = image_normal / 255.0
            image_normal = rearrange(image_normal, 'h w c -> 1 c h w')
            image_normal = self.norm(image_normal)

            normal = self.model_patcher.model(image_normal)
            normal = normal[0][-1][:, :3]
            normal = ((normal + 1) * 0.5).clip(0, 1)

            normal = rearrange(normal[0], 'c h w -> h w c').cpu().numpy()
            normal_image = (normal * 255.0).clip(0, 255).astype(np.uint8)

        return remove_pad(normal_image)


add_supported_preprocessor(PreprocessorNormalBae())

```

# 新特性（原始 WebUI 不具备的功能）

得益于 Unet 补丁工具，Forge 现在支持许多新的功能，包括 SVD、Z123、带蒙版的 Ip-Adapter、带蒙版的控制网络（ControlNet）、照片生成器（PhotoMaker）等。

带蒙版的 Ip-Adapter

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/d26630f9-922d-4483-8bf9-f364dca5fd50)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/03580ef7-235c-4b03-9ca6-a27677a5a175)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/d9ed4a01-70d4-45b4-a6a7-2f765f158fae)

带蒙版的 ControlNet

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/872d4785-60e4-4431-85c7-665c781dddaa)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/335a3b33-1ef8-46ff-a462-9f1b4f2c49fc)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/b3684a15-8895-414e-8188-487269dfcada)

照片生成器（PhotoMaker）

（请注意，PhotoMaker 是一个特殊控制器，需要你在提示词中添加触发词 "photomaker"。你的提示应类似于 "一张 photomaker 的照片")

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/07b0b626-05b5-473b-9d69-3657624d59be)

Marigold 深度效果

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/bdf54148-892d-410d-8ed9-70b4b121b6e7)

# 新增采样器（原版未包含的）

    DDPM
    DDPM Karras
    DPM++ 2M Turbo
    DPM++ 2M SDE Turbo
    LCM Karras
    Euler A Turbo

# 关于扩展插件

ControlNet 和 TiledVAE 已被集成，你应该卸载这两个扩展：

    sd-webui-controlnet
    multidiffusion-upscaler-for-automatic1111

注意，**AnimateDiff** 正在由 [continue-revolution](https://github.com/continue-revolution) 在 [sd-webui-animatediff forge/master 分支](https://github.com/continue-revolution/sd-webui-animatediff/tree/forge/master) 和 [sd-forge-animatediff](https://github.com/continue-revolution/sd-forge-animatediff) （它们保持同步更新）进行构建。（continue-revolution 原话：“prompt travel, inf t2v, controlnet v2v 已经证明工作良好；motion lora, i2i batch 功能仍在建设中，可能在一个星期内完成”）

其他扩展如以下所列应当可以正常运行：

    canvas-zoom
    translations/localizations
    Dynamic Prompts
    Adetailer
    Ultimate SD Upscale
    Reactor

然而，如果新扩展使用 Forge 构建，其代码量会大大减少。通常情况下，如果旧扩展采用 Forge 的 Unet 补丁工具重构，大约可以移除 80% 的代码，特别是当它们需要调用 ControlNet 时。

# 贡献指南

Forge 使用机器人每天下午从 https://github.com/AUTOMATIC1111/stable-diffusion-webui/tree/dev 自动获取提交和代码（若合并通过自动化的 Git 机器人或我的编译器，或者通过我的 ChatGPT 机器人成功完成），否则会在半夜（如果我的编译器和 ChatGPT 机器人都未能自动合并，则我将手动审查并处理）。

所有能在 https://github.com/AUTOMATIC1111/stable-diffusion-webui/tree/dev 实现的 PR 应该提交到那里。

欢迎在此处提交与 Forge 功能相关的 PR。