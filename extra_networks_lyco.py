from modules import extra_networks, shared
import lycoris


def parse_args(params:list):
    if 'dyn' in params[-1]:
        dyn_dim = int(params.pop()[3:])
    else:
        dyn_dim = None
    
    te_multipliers = float(params[0]) if len(params) else 1.0
    unet_multipliers = float(params[1]) if len(params) > 1 else te_multipliers
    return te_multipliers, unet_multipliers, dyn_dim


class ExtraNetworkLyCORIS(extra_networks.ExtraNetwork):
    def __init__(self):
        super().__init__('lyco')

    def activate(self, p, params_list):
        additional = shared.opts.sd_lyco

        if additional != "" and additional in lycoris.available_lycos and len([x for x in params_list if x.items[0] == additional]) == 0:
            p.all_prompts = [x + f"<lyco:{additional}:{shared.opts.extra_networks_default_multiplier}>" for x in p.all_prompts]
            params_list.append(extra_networks.ExtraNetworkParams(items=[additional, shared.opts.extra_networks_default_multiplier]))

        names = []
        te_multipliers = []
        unet_multipliers = []
        dyn_dims = []
        for params in params_list:
            assert len(params.items) > 0

            names.append(params.items[0])
            te, unet, dyn_dim = parse_args(params.items[1:])
            te_multipliers.append(te)
            unet_multipliers.append(unet)
            dyn_dims.append(dyn_dim)

        lycoris.load_lycos(names, te_multipliers, unet_multipliers, dyn_dims)

    def deactivate(self, p):
        pass
