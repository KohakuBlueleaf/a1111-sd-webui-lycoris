from modules import extra_networks, shared
import lycoris

class ExtraNetworkLyCORIS(extra_networks.ExtraNetwork):
    def __init__(self):
        super().__init__('lyco')

    def activate(self, p, params_list):
        additional = shared.opts.sd_lyco

        if additional != "" and additional in lycoris.available_lycos and len([x for x in params_list if x.items[0] == additional]) == 0:
            p.all_prompts = [x + f"<lyco:{additional}:{shared.opts.extra_networks_default_multiplier}>" for x in p.all_prompts]
            params_list.append(extra_networks.ExtraNetworkParams(items=[additional, shared.opts.extra_networks_default_multiplier]))

        names = []
        multipliers = []
        for params in params_list:
            assert len(params.items) > 0

            names.append(params.items[0])
            multipliers.append(float(params.items[1]) if len(params.items) > 1 else 1.0)

        lycoris.load_lycos(names, multipliers)

    def deactivate(self, p):
        pass
