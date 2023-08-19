import re


def disable_lycoris():
    try:
        from launch import git_tag
    except:
        return False
    match = re.search(r'(\d+(?:\.\d)+)', git_tag())
    if match:
        for v1, v2 in zip([int(v) for v in match.group().split(".")], [1, 5, 0]):
            if v1 < v2:
                return False
        return True
    return False


if disable_lycoris():
    print("""
========================= a1111-sd-webui-lycoris =========================
Starting from stable-diffusion-webui version 1.5.0
a1111-sd-webui-lycoris extension is no longer needed

All its features have been integrated into the native LoRA extension
LyCORIS models can now be used as if there are regular LoRA models

This extension has been automatically deactivated
Please remove this extension
==========================================================================
""")
else:
    import torch
    import gradio as gr

    import lycoris
    import extra_networks_lyco
    import ui_extra_networks_lyco
    from fastapi import FastAPI

    import logging
    from lyco_logger import logger

    from modules import script_callbacks, ui_extra_networks, extra_networks, shared

    def api_locyoris(_: gr.Blocks, app: FastAPI):
        @app.get("/sdapi/v1/lycos")
        async def get_lycos():
            return [create_lyco_json(obj) for obj in lycoris.available_lycos.values()]

        @app.post("/sdapi/v1/refresh-lycos")
        async def refresh_lycos():
            return lycoris.list_available_lycos()

    def create_lyco_json(obj: lycoris.LycoOnDisk):
        return {
            "name": obj.name,
            "path": obj.filename,
            "metadata": obj.metadata,
        }

    script_callbacks.on_app_started(api_locyoris)

    def unload():
        torch.nn.Linear.forward = torch.nn.Linear_forward_before_lyco
        torch.nn.Linear._load_from_state_dict = torch.nn.Linear_load_state_dict_before_lyco
        torch.nn.Conv2d.forward = torch.nn.Conv2d_forward_before_lyco
        torch.nn.Conv2d._load_from_state_dict = torch.nn.Conv2d_load_state_dict_before_lyco
        torch.nn.MultiheadAttention.forward = torch.nn.MultiheadAttention_forward_before_lyco
        torch.nn.MultiheadAttention._load_from_state_dict = torch.nn.MultiheadAttention_load_state_dict_before_lyco


    def before_ui():
        if shared.cmd_opts.lyco_debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Set lyco logger level to DEBUG")

        if shared.cmd_opts.lyco_patch_lora:
            logger.warning('Triggered lyco-patch-lora, will take lora_dir and <lora> format.')
            for idx, x in enumerate(ui_extra_networks.extra_pages):
                if x.name=='lora':
                    break
            else:
                idx = -1

            if idx != -1:
                ui_extra_networks.extra_pages.pop(idx)

            ui_extra_networks.register_page(ui_extra_networks_lyco.ExtraNetworksPageLyCORIS(
                'lora'
            ))
            extra_networks.register_extra_network(extra_networks_lyco.ExtraNetworkLyCORIS(
                'lora'
            ))
            lycoris.list_available_lycos(shared.cmd_opts.lora_dir)
        else:
            ui_extra_networks.register_page(ui_extra_networks_lyco.ExtraNetworksPageLyCORIS())
            extra_networks.register_extra_network(extra_networks_lyco.ExtraNetworkLyCORIS())
            lycoris.list_available_lycos(shared.cmd_opts.lyco_dir)


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
