from modules import extra_networks, shared
import lycoris
from lyco_logger import logger


default_args = [
    ('te', 1.0, float),
    ('unet', None, float),
    ('dyn', None, int)
]


def parse_args(params:list):
    arg_list = []
    kwarg_list = {}
    
    for i in params:
        if '=' in str(i):
            k, v = i.split('=', 1)
            kwarg_list[k] = v
        else:
            arg_list.append(i)
    
    args = []
    for name, default, type in default_args:
        if name in kwarg_list:
            x = kwarg_list[name]
        elif arg_list:
            x = arg_list.pop(0)
        else:
            x = default
        
        if x == 'default':
            x = default
        elif x is not None:
            x = type(x)
        
        args.append(x)
    
    return args


class ExtraNetworkLyCORIS(extra_networks.ExtraNetwork):
    def __init__(self, base_name = 'lyco'):
        super().__init__(base_name)
        self.base_name = base_name
        self.cache = ()

    def activate(self, p, params_list):
        if self.base_name == 'lora':
            additional = shared.opts.sd_lora
        elif self.base_name == 'lyco':
            additional = shared.opts.sd_lyco

        if additional != "" and additional in lycoris.available_lycos and len([x for x in params_list if x.items[0] == additional]) == 0:
            p.all_prompts = [
                x + 
                f"<{self.base_name}:{additional}:{shared.opts.extra_networks_default_multiplier}>" 
                for x in p.all_prompts
            ]
            params_list.append(
                extra_networks.ExtraNetworkParams(
                    items=[additional, shared.opts.extra_networks_default_multiplier])
                )

        names = []
        te_multipliers = []
        unet_multipliers = []
        dyn_dims = []
        for params in params_list:
            assert len(params.items) > 0

            names.append(params.items[0])
            te, unet, dyn_dim = parse_args(params.items[1:])
            if unet is None:
                unet = te
            te_multipliers.append(te)
            unet_multipliers.append(unet)
            dyn_dims.append(dyn_dim)
        
        all_lycos = tuple(
            (name, te, unet, dyn)
            for name, te, unet, dyn in zip(names, te_multipliers, unet_multipliers, dyn_dims)
        )
        
        if all_lycos != self.cache:
            for name, te, unet, dyn in all_lycos:
                logger.debug(f"\nApply LyCORIS model: {name}: te={te}, unet={unet}, dyn={dyn}")
            self.cache = all_lycos
        lycoris.load_lycos(names, te_multipliers, unet_multipliers, dyn_dims)

    def deactivate(self, p):
        pass
