from typing import *
import os, sys
import re
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import shared, devices, sd_models, errors
from lyco_logger import logger


metadata_tags_order = {"ss_sd_model_name": 1, "ss_resolution": 2, "ss_clip_skip": 3, "ss_num_train_images": 10, "ss_tag_frequency": 20}


re_digits = re.compile(r"\d+")
re_x_proj = re.compile(r"(.*)_([qkv]_proj)$")

re_unet_conv_in = re.compile(r"lora_unet_conv_in(.+)")
re_unet_conv_out = re.compile(r"lora_unet_conv_out(.+)")
re_unet_time_embed = re.compile(r"lora_unet_time_embedding_linear_(\d+)(.+)")

re_unet_down_blocks = re.compile(r"lora_unet_down_blocks_(\d+)_attentions_(\d+)_(.+)")
re_unet_mid_blocks = re.compile(r"lora_unet_mid_block_attentions_(\d+)_(.+)")
re_unet_up_blocks = re.compile(r"lora_unet_up_blocks_(\d+)_attentions_(\d+)_(.+)")

re_unet_down_blocks_res = re.compile(r"lora_unet_down_blocks_(\d+)_resnets_(\d+)_(.+)")
re_unet_mid_blocks_res = re.compile(r"lora_unet_mid_block_resnets_(\d+)_(.+)")
re_unet_up_blocks_res = re.compile(r"lora_unet_up_blocks_(\d+)_resnets_(\d+)_(.+)")

re_unet_downsample = re.compile(r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv(.+)")
re_unet_upsample = re.compile(r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv(.+)")

re_text_block = re.compile(r"lora_te_text_model_encoder_layers_(\d+)_(.+)")


def convert_diffusers_name_to_compvis(key, is_sd2):
    # I don't know why but some state dict has this kind of thing
    key = key.replace('text_model_text_model', 'text_model')
    def match(match_list, regex):
        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []
    
    if match(m, re_unet_conv_in):
        return f'diffusion_model_input_blocks_0_0{m[0]}'
    
    if match(m, re_unet_conv_out):
        return f'diffusion_model_out_2{m[0]}'
    
    if match(m, re_unet_time_embed):
        return f"diffusion_model_time_embed_{m[0]*2-2}{m[1]}"
    
    if match(m, re_unet_down_blocks):
        return f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[1]}_1_{m[2]}"

    if match(m, re_unet_mid_blocks):
        return f"diffusion_model_middle_block_1_{m[1]}"

    if match(m, re_unet_up_blocks):
        return f"diffusion_model_output_blocks_{m[0] * 3 + m[1]}_1_{m[2]}"

    if match(m, re_unet_down_blocks_res):
        block = f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[1]}_0_"
        if m[2].startswith('conv1'):
            return f"{block}in_layers_2{m[2][len('conv1'):]}"
        elif m[2].startswith('conv2'):
            return f"{block}out_layers_3{m[2][len('conv2'):]}"
        elif m[2].startswith('time_emb_proj'):
            return f"{block}emb_layers_1{m[2][len('time_emb_proj'):]}"
        elif m[2].startswith('conv_shortcut'):
            return f"{block}skip_connection{m[2][len('conv_shortcut'):]}"

    if match(m, re_unet_mid_blocks_res):
        block = f"diffusion_model_middle_block_{m[0]*2}_"
        if m[1].startswith('conv1'):
            return f"{block}in_layers_2{m[1][len('conv1'):]}"
        elif m[1].startswith('conv2'):
            return f"{block}out_layers_3{m[1][len('conv2'):]}"
        elif m[1].startswith('time_emb_proj'):
            return f"{block}emb_layers_1{m[1][len('time_emb_proj'):]}"
        elif m[1].startswith('conv_shortcut'):
            return f"{block}skip_connection{m[1][len('conv_shortcut'):]}"

    if match(m, re_unet_up_blocks_res):
        block = f"diffusion_model_output_blocks_{m[0] * 3 + m[1]}_0_"
        if m[2].startswith('conv1'):
            return f"{block}in_layers_2{m[2][len('conv1'):]}"
        elif m[2].startswith('conv2'):
            return f"{block}out_layers_3{m[2][len('conv2'):]}"
        elif m[2].startswith('time_emb_proj'):
            return f"{block}emb_layers_1{m[2][len('time_emb_proj'):]}"
        elif m[2].startswith('conv_shortcut'):
            return f"{block}skip_connection{m[2][len('conv_shortcut'):]}"

    if match(m, re_unet_downsample):
        return f"diffusion_model_input_blocks_{m[0]*3+3}_0_op{m[1]}"

    if match(m, re_unet_upsample):
        return f"diffusion_model_output_blocks_{m[0]*3 + 2}_{1+(m[0]!=0)}_conv{m[1]}"

    if match(m, r"lora_te_text_model_encoder_layers_(\d+)_(.+)"):
        if is_sd2:
            if 'mlp_fc1' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
            elif 'mlp_fc2' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
            else:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    return key


def assign_lyco_names_to_compvis_modules(sd_model):
    lyco_layer_mapping = {}

    for name, module in shared.sd_model.cond_stage_model.wrapped.named_modules():
        lyco_name = name.replace(".", "_")
        lyco_layer_mapping[lyco_name] = module
        module.lyco_layer_name = lyco_name

    for name, module in shared.sd_model.model.named_modules():
        lyco_name = name.replace(".", "_")
        lyco_layer_mapping[lyco_name] = module
        module.lyco_layer_name = lyco_name

    sd_model.lyco_layer_mapping = lyco_layer_mapping


class LycoOnDisk:
    def __init__(self, name, filename):
        self.name = name
        self.filename = filename
        self.metadata = {}

        _, ext = os.path.splitext(filename)
        if ext.lower() == ".safetensors":
            try:
                self.metadata = sd_models.read_metadata_from_safetensors(filename)
            except Exception as e:
                errors.display(e, f"reading lora {filename}")

        if self.metadata:
            m = {}
            for k, v in sorted(self.metadata.items(), key=lambda x: metadata_tags_order.get(x[0], 999)):
                m[k] = v

            self.metadata = m

        self.ssmd_cover_images = self.metadata.pop('ssmd_cover_images', None)  # those are cover images and they are too big to display in UI as text


class LycoModule:
    def __init__(self, name):
        self.name = name
        self.te_multiplier = 1.0
        self.unet_multiplier = 1.0
        self.dyn_dim = None
        self.modules = {}
        self.mtime = None


class FullModule:
    def __init__(self):
        self.weight = None
        self.alpha = None
        self.scale = None
        self.dim = None
        self.shape = None


class LycoUpDownModule:
    def __init__(self):
        self.up_model = None
        self.mid_model = None
        self.down_model = None
        self.alpha = None
        self.scale = None
        self.dim = None
        self.shape = None
        self.bias = None


def make_weight_cp(t, wa, wb):
    temp = torch.einsum('i j k l, j r -> i r k l', t, wb)
    return torch.einsum('i j k l, i r -> r j k l', temp, wa)


class LycoHadaModule:
    def __init__(self):
        self.t1 = None
        self.w1a = None
        self.w1b = None
        self.t2 = None
        self.w2a = None
        self.w2b = None
        self.alpha = None
        self.scale = None
        self.dim = None
        self.shape = None
        self.bias = None


class IA3Module:
    def __init__(self):
        self.w = None
        self.alpha = None
        self.scale = None
        self.dim = None
        self.on_input = None


def make_kron(orig_shape, w1, w2):
    if len(w2.shape) == 4:
        w1 = w1.unsqueeze(2).unsqueeze(2)
    w2 = w2.contiguous()
    return torch.kron(w1, w2).reshape(orig_shape)


class LycoKronModule:
    def __init__(self):
        self.w1 = None
        self.w1a = None
        self.w1b = None
        self.w2 = None
        self.t2 = None
        self.w2a = None
        self.w2b = None
        self._alpha = None
        self.scale = None
        self.dim = None
        self.shape = None
        self.bias = None
    
    @property
    def alpha(self):
        if self.w1a is None and self.w2a is None:
            return None
        else:
            return self._alpha
    
    @alpha.setter
    def alpha(self, x):
        self._alpha = x


CON_KEY = {
    "lora_up.weight", "dyn_up",
    "lora_down.weight", "dyn_down",
    "lora_mid.weight"
}
HADA_KEY = {
    "hada_t1",
    "hada_w1_a",
    "hada_w1_b",
    "hada_t2",
    "hada_w2_a",
    "hada_w2_b",
}
IA3_KEY = {
    "weight",
    "on_input"
}
KRON_KEY = {
    "lokr_w1",
    "lokr_w1_a",
    "lokr_w1_b",
    "lokr_t2",
    "lokr_w2",
    "lokr_w2_a",
    "lokr_w2_b",
}

def load_lyco(name, filename):
    lyco = LycoModule(name)
    lyco.mtime = os.path.getmtime(filename)

    sd = sd_models.read_state_dict(filename)
    is_sd2 = 'model_transformer_resblocks' in shared.sd_model.lyco_layer_mapping

    keys_failed_to_match = []

    for key_diffusers, weight in sd.items():
        fullkey = convert_diffusers_name_to_compvis(key_diffusers, is_sd2)
        key, lyco_key = fullkey.split(".", 1)
        
        sd_module = shared.sd_model.lyco_layer_mapping.get(key, None)
        
        if sd_module is None:
            m = re_x_proj.match(key)
            if m:
                sd_module = shared.sd_model.lyco_layer_mapping.get(m.group(1), None)
        
        if sd_module is None:
            logger.warn(f'key failed to match: {key_diffusers}')
            keys_failed_to_match.append(key_diffusers)
            continue

        lyco_module = lyco.modules.get(key, None)
        if lyco_module is None:
            lyco_module = LycoUpDownModule()
            lyco.modules[key] = lyco_module

        if lyco_key == "alpha":
            lyco_module.alpha = weight.item()
            continue
        
        if lyco_key == "scale":
            lyco_module.scale = weight.item()
            continue
        
        if lyco_key == "diff":
            weight = weight.to(device=devices.cpu, dtype=devices.dtype)
            weight.requires_grad_(False)
            lyco_module = FullModule()
            lyco.modules[key] = lyco_module
            lyco_module.weight = weight
            continue
        
        if 'bias_' in lyco_key:
            if lyco_module.bias is None:
                lyco_module.bias = [None, None, None]
            if 'bias_indices' == lyco_key:
                lyco_module.bias[0] = weight
            elif 'bias_values' == lyco_key:
                lyco_module.bias[1] = weight
            elif 'bias_size' == lyco_key:
                lyco_module.bias[2] = weight
            
            if all((i is not None) for i in lyco_module.bias):
                print('build bias')
                lyco_module.bias = torch.sparse_coo_tensor(
                    lyco_module.bias[0],
                    lyco_module.bias[1],
                    tuple(lyco_module.bias[2]),
                ).to(device=devices.cpu, dtype=devices.dtype)
                lyco_module.bias.requires_grad_(False)
            continue
        
        if lyco_key in CON_KEY:
            if (type(sd_module) == torch.nn.Linear
                or type(sd_module) == torch.nn.modules.linear.NonDynamicallyQuantizableLinear
                or type(sd_module) == torch.nn.MultiheadAttention):
                weight = weight.reshape(weight.shape[0], -1)
                module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
            elif type(sd_module) == torch.nn.Conv2d:
                if lyco_key == "lora_down.weight" or lyco_key == "dyn_up":
                    if len(weight.shape) == 2:
                        weight = weight.reshape(weight.shape[0], -1, 1, 1)
                    if weight.shape[2] != 1 or weight.shape[3] != 1:
                        module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], sd_module.kernel_size, sd_module.stride, sd_module.padding, bias=False)
                    else:
                        module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
                elif lyco_key == "lora_mid.weight":
                    module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], sd_module.kernel_size, sd_module.stride, sd_module.padding, bias=False)
                elif lyco_key == "lora_up.weight" or lyco_key == "dyn_down":
                    module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
            else:
                assert False, f'Lyco layer {key_diffusers} matched a layer with unsupported type: {type(sd_module).__name__}'
            
            if hasattr(sd_module, 'weight'):
                lyco_module.shape = sd_module.weight.shape
            with torch.no_grad():
                if weight.shape != module.weight.shape:
                    weight = weight.reshape(module.weight.shape)
                module.weight.copy_(weight)

            module.to(device=devices.cpu, dtype=devices.dtype)
            module.requires_grad_(False)

            if lyco_key == "lora_up.weight" or lyco_key == "dyn_up":
                lyco_module.up_model = module
            elif lyco_key == "lora_mid.weight":
                lyco_module.mid_model = module
            elif lyco_key == "lora_down.weight" or lyco_key == "dyn_down":
                lyco_module.down_model = module
                lyco_module.dim = weight.shape[0]
            else:
                print(lyco_key)
        elif lyco_key in HADA_KEY:
            if type(lyco_module) != LycoHadaModule:
                alpha = lyco_module.alpha
                bias = lyco_module.bias
                lyco_module = LycoHadaModule()
                lyco_module.alpha = alpha
                lyco_module.bias = bias
                lyco.modules[key] = lyco_module
            if hasattr(sd_module, 'weight'):
                lyco_module.shape = sd_module.weight.shape
            
            weight = weight.to(device=devices.cpu, dtype=devices.dtype)
            weight.requires_grad_(False)
            
            if lyco_key == 'hada_w1_a':
                lyco_module.w1a = weight
            elif lyco_key == 'hada_w1_b':
                lyco_module.w1b = weight
                lyco_module.dim = weight.shape[0]
            elif lyco_key == 'hada_w2_a':
                lyco_module.w2a = weight
            elif lyco_key == 'hada_w2_b':
                lyco_module.w2b = weight
                lyco_module.dim = weight.shape[0]
            elif lyco_key == 'hada_t1':
                lyco_module.t1 = weight
            elif lyco_key == 'hada_t2':
                lyco_module.t2 = weight
            
        elif lyco_key in IA3_KEY:
            if type(lyco_module) != IA3Module:
                lyco_module = IA3Module()
                lyco.modules[key] = lyco_module
            
            if lyco_key == "weight":
                lyco_module.w = weight.to(devices.device, dtype=devices.dtype)
            elif lyco_key == "on_input":
                lyco_module.on_input = weight
        elif lyco_key in KRON_KEY:
            if not isinstance(lyco_module, LycoKronModule):
                alpha = lyco_module.alpha
                bias = lyco_module.bias
                lyco_module = LycoKronModule()
                lyco_module.alpha = alpha
                lyco_module.bias = bias
                lyco.modules[key] = lyco_module
            if hasattr(sd_module, 'weight'):
                lyco_module.shape = sd_module.weight.shape
            
            weight = weight.to(device=devices.cpu, dtype=devices.dtype)
            weight.requires_grad_(False)
            
            if lyco_key == 'lokr_w1':
                lyco_module.w1 = weight
            elif lyco_key == 'lokr_w1_a':
                lyco_module.w1a = weight
            elif lyco_key == 'lokr_w1_b':
                lyco_module.w1b = weight
                lyco_module.dim = weight.shape[0]
            elif lyco_key == 'lokr_w2':
                lyco_module.w2 = weight
            elif lyco_key == 'lokr_w2_a':
                lyco_module.w2a = weight
            elif lyco_key == 'lokr_w2_b':
                lyco_module.w2b = weight
                lyco_module.dim = weight.shape[0]
            elif lyco_key == 'lokr_t2':
                lyco_module.t2 = weight
        else:
            assert False, f'Bad Lyco layer name: {key_diffusers} - must end in lyco_up.weight, lyco_down.weight or alpha'

    if len(keys_failed_to_match) > 0:
        print(shared.sd_model.lyco_layer_mapping)
        print(f"Failed to match keys when loading Lyco {filename}: {keys_failed_to_match}")

    return lyco


def load_lycos(names, te_multipliers=None, unet_multipliers=None, dyn_dims=None):
    already_loaded = {}

    for lyco in loaded_lycos:
        if lyco.name in names:
            already_loaded[lyco.name] = lyco

    loaded_lycos.clear()

    lycos_on_disk = [available_lycos.get(name, None) for name in names]
    if any([x is None for x in lycos_on_disk]):
        list_available_lycos()

        lycos_on_disk = [available_lycos.get(name, None) for name in names]

    for i, name in enumerate(names):
        lyco = already_loaded.get(name, None)

        lyco_on_disk = lycos_on_disk[i]
        if lyco_on_disk is not None:
            if lyco is None or os.path.getmtime(lyco_on_disk.filename) > lyco.mtime:
                lyco = load_lyco(name, lyco_on_disk.filename)

        if lyco is None:
            print(f"Couldn't find Lora with name {name}")
            continue

        lyco.te_multiplier = te_multipliers[i] if te_multipliers else 1.0
        lyco.unet_multiplier = unet_multipliers[i] if unet_multipliers else lyco.te_multiplier
        lyco.dyn_dim = dyn_dims[i] if dyn_dims else None
        loaded_lycos.append(lyco)


def _rebuild_conventional(up, down, shape, dyn_dim=None):
    up = up.reshape(up.size(0), -1)
    down = down.reshape(down.size(0), -1)
    if dyn_dim is not None:
        up = up[:, :dyn_dim]
        down = down[:dyn_dim, :]
    return (up @ down).reshape(shape)


def _rebuild_cp_decomposition(up, down, mid):
    up = up.reshape(up.size(0), -1)
    down = down.reshape(down.size(0), -1)
    return torch.einsum('n m k l, i n, m j -> i j k l', mid, up, down)


def rebuild_weight(module, orig_weight: torch.Tensor, dyn_dim: int=None) -> torch.Tensor:
    output_shape: Sized
    if module.__class__.__name__ == 'LycoUpDownModule':
        up = module.up_model.weight.to(orig_weight.device, dtype=orig_weight.dtype)
        down = module.down_model.weight.to(orig_weight.device, dtype=orig_weight.dtype)
        
        output_shape = [up.size(0), down.size(1)]
        if (mid:=module.mid_model) is not None:
            # cp-decomposition
            mid = mid.weight.to(orig_weight.device, dtype=orig_weight.dtype)
            updown = _rebuild_cp_decomposition(up, down, mid)
            output_shape += mid.shape[2:]
        else:
            if len(down.shape) == 4:
                output_shape += down.shape[2:]
            updown = _rebuild_conventional(up, down, output_shape, dyn_dim)
        
    elif module.__class__.__name__ == 'LycoHadaModule':
        w1a = module.w1a.to(orig_weight.device, dtype=orig_weight.dtype)
        w1b = module.w1b.to(orig_weight.device, dtype=orig_weight.dtype)
        w2a = module.w2a.to(orig_weight.device, dtype=orig_weight.dtype)
        w2b = module.w2b.to(orig_weight.device, dtype=orig_weight.dtype)
        
        output_shape = [w1a.size(0), w1b.size(1)]
        
        if module.t1 is not None:
            output_shape = [w1a.size(1), w1b.size(1)]
            t1 = module.t1.to(orig_weight.device, dtype=orig_weight.dtype)
            updown1 = make_weight_cp(t1, w1a, w1b)
            output_shape += t1.shape[2:]
        else:
            if len(w1b.shape) == 4:
                output_shape += w1b.shape[2:]
            updown1 = _rebuild_conventional(w1a, w1b, output_shape)
        
        if module.t2 is not None:
            t2 = module.t2.to(orig_weight.device, dtype=orig_weight.dtype)
            updown2 = make_weight_cp(t2, w2a, w2b)
        else:
            updown2 = _rebuild_conventional(w2a, w2b, output_shape)
        
        updown = updown1 * updown2
    
    elif module.__class__.__name__ == 'FullModule':
        output_shape = module.weight.shape
        updown = module.weight.to(orig_weight.device, dtype=orig_weight.dtype)
    
    elif module.__class__.__name__ == 'IA3Module':
        output_shape = [module.w.size(0), orig_weight.size(1)]
        if module.on_input:
            output_shape.reverse()
        else:
            module.w = module.w.reshape(-1, 1)
        updown = orig_weight * module.w
        
    elif module.__class__.__name__ == 'LycoKronModule':
        if module.w1 is not None:
            w1 = module.w1.to(orig_weight.device, dtype=orig_weight.dtype)
        else:
            w1a = module.w1a.to(orig_weight.device, dtype=orig_weight.dtype)
            w1b = module.w1b.to(orig_weight.device, dtype=orig_weight.dtype)
            w1 = w1a @ w1b
        
        if module.w2 is not None:
            w2 = module.w2.to(orig_weight.device, dtype=orig_weight.dtype)
        elif module.t2 is None:
            w2a = module.w2a.to(orig_weight.device, dtype=orig_weight.dtype)
            w2b = module.w2b.to(orig_weight.device, dtype=orig_weight.dtype)
            w2 = w2a @ w2b
        else:
            t2 = module.t2.to(orig_weight.device, dtype=orig_weight.dtype)
            w2a = module.w2a.to(orig_weight.device, dtype=orig_weight.dtype)
            w2b = module.w2b.to(orig_weight.device, dtype=orig_weight.dtype)
            w2 = make_weight_cp(t2, w2a, w2b)
        
        output_shape = [w1.size(0)*w2.size(0), w1.size(1)*w2.size(1)]
        if len(orig_weight.shape) == 4:
            output_shape = orig_weight.shape
        
        updown = make_kron(
            output_shape, w1, w2
        )
    
    else:
        raise NotImplementedError(
            f"Unknown module type: {module.__class__.__name__}\n"
            "If the type is one of "
            "'LycoUpDownModule', 'LycoHadaModule', 'FullModule', 'IA3Module', 'LycoKronModule'"
            "You may have other lyco extension that conflict with locon extension."
        )
    
    if hasattr(module, 'bias') and module.bias != None:
        updown = updown.reshape(module.bias.shape)
        updown += module.bias.to(orig_weight.device, dtype=orig_weight.dtype)
        updown = updown.reshape(output_shape)
    
    if len(output_shape) == 4:
        updown = updown.reshape(output_shape)
    
    if orig_weight.size().numel() == updown.size().numel():
        updown = updown.reshape(orig_weight.shape)
    # print(torch.sum(updown))
    return updown


def lyco_calc_updown(lyco, module, target, multiplier):
    with torch.no_grad():
        updown = rebuild_weight(module, target, lyco.dyn_dim)
        if lyco.dyn_dim and module.dim:
            dim = min(lyco.dyn_dim, module.dim)
        elif lyco.dyn_dim:
            dim = lyco.dyn_dim
        elif module.dim:
            dim = module.dim
        else:
            dim = None
        
        scale = (
            module.scale if module.scale is not None
            else module.alpha / dim if dim is not None and module.alpha is not None
            else 1.0
        )
        # print(scale, module.alpha, module.dim, lyco.dyn_dim)
        updown = updown * multiplier * scale
        return updown


def lyco_apply_weights(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.MultiheadAttention]):
    """
    Applies the currently selected set of Lycos to the weights of torch layer self.
    If weights already have this particular set of lycos applied, does nothing.
    If not, restores orginal weights from backup and alters weights according to lycos.
    """

    lyco_layer_name = getattr(self, 'lyco_layer_name', None)
    if lyco_layer_name is None:
        return

    current_names = getattr(self, "lyco_current_names", ())
    lora_prev_names = getattr(self, "lora_prev_names", ())
    lora_names = getattr(self, "lora_current_names", ())
    wanted_names = tuple((x.name, x.te_multiplier, x.unet_multiplier, x.dyn_dim) for x in loaded_lycos)

    # We take lora_changed as base_weight changed
    # but functional lora will not affect the weight so take it as unchanged
    lora_changed = lora_prev_names != lora_names
    lora_functional = getattr(shared.opts, 'lora_functional', False)
    lora_changed = lora_changed and not lora_functional

    lyco_changed = current_names != wanted_names

    weights_backup = getattr(self, "lyco_weights_backup", None)

    if ((len(loaded_lycos) and weights_backup is None)
        or (weights_backup is not None and lora_changed)):
        # backup when:
        #  * apply lycos but haven't backed up any weights
        #  * have outdated backed up weights
        if isinstance(self, torch.nn.MultiheadAttention):
            weights_backup = (
                self.in_proj_weight.to(devices.cpu, copy=True), 
                self.out_proj.weight.to(devices.cpu, copy=True)
            )
        else:
            weights_backup = self.weight.to(devices.cpu, copy=True)
        self.lyco_weights_backup = weights_backup
    elif len(loaded_lycos) == 0:
        # when we unload all the lycos and have no weights to backup
        # clean backup weights to save ram
        self.lyco_weights_backup = None

    if lyco_changed or lora_changed:
        if weights_backup is not None:
            if isinstance(self, torch.nn.MultiheadAttention):
                self.in_proj_weight.copy_(weights_backup[0])
                self.out_proj.weight.copy_(weights_backup[1])
            else:
                self.weight.copy_(weights_backup)

        for lyco in loaded_lycos:
            module = lyco.modules.get(lyco_layer_name, None)
            multiplier = (
                lyco.te_multiplier if 'transformer' in lyco_layer_name[:20] 
                else lyco.unet_multiplier
            )
            if module is not None and hasattr(self, 'weight'):
                # print(lyco_layer_name, multiplier)
                updown = lyco_calc_updown(lyco, module, self.weight, multiplier)
                if len(self.weight.shape) == 4 and self.weight.shape[1] == 9:
                    # inpainting model. zero pad updown to make channel[1]  4 to 9
                    updown = F.pad(updown, (0, 0, 0, 0, 0, 5))
                self.weight += updown
                continue

            module_q = lyco.modules.get(lyco_layer_name + "_q_proj", None)
            module_k = lyco.modules.get(lyco_layer_name + "_k_proj", None)
            module_v = lyco.modules.get(lyco_layer_name + "_v_proj", None)
            module_out = lyco.modules.get(lyco_layer_name + "_out_proj", None)

            if isinstance(self, torch.nn.MultiheadAttention) and module_q and module_k and module_v and module_out:
                updown_q = lyco_calc_updown(lyco, module_q, self.in_proj_weight, multiplier)
                updown_k = lyco_calc_updown(lyco, module_k, self.in_proj_weight, multiplier)
                updown_v = lyco_calc_updown(lyco, module_v, self.in_proj_weight, multiplier)
                updown_qkv = torch.vstack([updown_q, updown_k, updown_v])

                self.in_proj_weight += updown_qkv
                self.out_proj.weight += lyco_calc_updown(lyco, module_out, self.out_proj.weight, multiplier)
                continue

            if module is None:
                continue

            logger.error(3, f'failed to calculate lyco weights for layer {lyco_layer_name}')

        setattr(self, "lora_prev_names", lora_names)
        setattr(self, "lyco_current_names", wanted_names)


def lyco_reset_cached_weight(self: Union[torch.nn.Conv2d, torch.nn.Linear]):
    setattr(self, "lyco_current_names", ())
    setattr(self, "lyco_weights_backup", None)


def lyco_Linear_forward(self, input):
    lyco_apply_weights(self)

    return torch.nn.Linear_forward_before_lyco(self, input)


def lyco_Linear_load_state_dict(self, *args, **kwargs):
    lyco_reset_cached_weight(self)

    return torch.nn.Linear_load_state_dict_before_lyco(self, *args, **kwargs)


def lyco_Conv2d_forward(self, input):
    lyco_apply_weights(self)

    return torch.nn.Conv2d_forward_before_lyco(self, input)


def lyco_Conv2d_load_state_dict(self, *args, **kwargs):
    lyco_reset_cached_weight(self)

    return torch.nn.Conv2d_load_state_dict_before_lyco(self, *args, **kwargs)


def lyco_MultiheadAttention_forward(self, *args, **kwargs):
    lyco_apply_weights(self)

    return torch.nn.MultiheadAttention_forward_before_lyco(self, *args, **kwargs)


def lyco_MultiheadAttention_load_state_dict(self, *args, **kwargs):
    lyco_reset_cached_weight(self)

    return torch.nn.MultiheadAttention_load_state_dict_before_lyco(self, *args, **kwargs)


def list_available_lycos(model_dir=shared.cmd_opts.lyco_dir):
    available_lycos.clear()

    os.makedirs(model_dir, exist_ok=True)

    candidates = \
        glob.glob(os.path.join(model_dir, '**/*.pt'), recursive=True) + \
        glob.glob(os.path.join(model_dir, '**/*.safetensors'), recursive=True) + \
        glob.glob(os.path.join(model_dir, '**/*.ckpt'), recursive=True)

    for filename in sorted(candidates, key=str.lower):
        if os.path.isdir(filename):
            continue

        name = os.path.splitext(os.path.basename(filename))[0]

        available_lycos[name] = LycoOnDisk(name, filename)


available_lycos: Dict[str, LycoOnDisk] = {}
loaded_lycos: List[LycoModule] = []

list_available_lycos()
