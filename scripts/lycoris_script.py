import torch
import gradio as gr

import lycoris
import extra_networks_lyco
import ui_extra_networks_lyco
from modules import script_callbacks, ui_extra_networks, extra_networks, shared


def unload():
    torch.nn.Linear.forward = torch.nn.Linear_forward_before_lyco
    torch.nn.Linear._load_from_state_dict = torch.nn.Linear_load_state_dict_before_lyco
    torch.nn.Conv2d.forward = torch.nn.Conv2d_forward_before_lyco
    torch.nn.Conv2d._load_from_state_dict = torch.nn.Conv2d_load_state_dict_before_lyco
    torch.nn.MultiheadAttention.forward = torch.nn.MultiheadAttention_forward_before_lyco
    torch.nn.MultiheadAttention._load_from_state_dict = torch.nn.MultiheadAttention_load_state_dict_before_lyco


def before_ui():
    if shared.cmd_opts.lyco_patch_lora:
        print(
            '=================================\n'
            'Triggered lyco-patch-lora, will take lora_dir and <lora> format.\n'
            'lyco_dir and <lyco> format is disabled\n'
            'This patch may affect other lora extension\n'
            '(if they don\'t support the lycoris extension or just use lora/lyco to determine which extension is working).\n'
            '================================='
        )
        extra_networks.register_extra_network(extra_networks_lyco.ExtraNetworkLyCORIS(
            'lora'
        ))
    else:
        ui_extra_networks.register_page(ui_extra_networks_lyco.ExtraNetworksPageLyCORIS())
        extra_networks.register_extra_network(extra_networks_lyco.ExtraNetworkLyCORIS())


if not hasattr(torch.nn, 'Linear_forward_before_lyco'):
    torch.nn.Linear_forward_before_lyco = torch.nn.Linear.forward

if not hasattr(torch.nn, 'Linear_load_state_dict_before_lyco'):
    torch.nn.Linear_load_state_dict_before_lyco = torch.nn.Linear._load_from_state_dict

if not hasattr(torch.nn, 'Conv2d_forward_before_lyco'):
    torch.nn.Conv2d_forward_before_lyco = torch.nn.Conv2d.forward

if not hasattr(torch.nn, 'Conv2d_load_state_dict_before_lyco'):
    torch.nn.Conv2d_load_state_dict_before_lyco = torch.nn.Conv2d._load_from_state_dict

if not hasattr(torch.nn, 'MultiheadAttention_forward_before_lyco'):
    torch.nn.MultiheadAttention_forward_before_lyco = torch.nn.MultiheadAttention.forward

if not hasattr(torch.nn, 'MultiheadAttention_load_state_dict_before_lyco'):
    torch.nn.MultiheadAttention_load_state_dict_before_lyco = torch.nn.MultiheadAttention._load_from_state_dict

torch.nn.Linear.forward = lycoris.lyco_Linear_forward
torch.nn.Linear._load_from_state_dict = lycoris.lyco_Linear_load_state_dict
torch.nn.Conv2d.forward = lycoris.lyco_Conv2d_forward
torch.nn.Conv2d._load_from_state_dict = lycoris.lyco_Conv2d_load_state_dict
torch.nn.MultiheadAttention.forward = lycoris.lyco_MultiheadAttention_forward
torch.nn.MultiheadAttention._load_from_state_dict = lycoris.lyco_MultiheadAttention_load_state_dict

script_callbacks.on_model_loaded(lycoris.assign_lyco_names_to_compvis_modules)
script_callbacks.on_script_unloaded(unload)
script_callbacks.on_before_ui(before_ui)


shared.options_templates.update(shared.options_section(('extra_networks', "Extra Networks"), {
    "sd_lyco": shared.OptionInfo("None", "Add LyCORIS to prompt", gr.Dropdown, lambda: {"choices": ["None"] + [x for x in lycoris.available_lycos]}, refresh=lycoris.list_available_lycos),
}))
