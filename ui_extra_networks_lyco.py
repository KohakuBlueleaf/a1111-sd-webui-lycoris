import json
import os
import lycoris

from modules import shared, ui_extra_networks


class ExtraNetworksPageLyCORIS(ui_extra_networks.ExtraNetworksPage):
    def __init__(self, base_name='lyco', model_dir=shared.cmd_opts.lyco_dir):
        super().__init__('LyCORIS')
        self.base_name = base_name
        self.model_dir = model_dir

    def refresh(self):
        lycoris.list_available_lycos(self.model_dir)

    def list_items(self):
        for index, (name, lyco_on_disk) in enumerate(lycoris.available_lycos.items()):
            path, ext = os.path.splitext(lyco_on_disk.filename)
            sort_keys = {} if not 'get_sort_keys' in dir(self) else self.get_sort_keys(lyco_on_disk.filename)
            yield {
                "name": name,
                "filename": path,
                "preview": self.find_preview(path),
                "description": self.find_description(path),
                "search_term": self.search_terms_from_path(lyco_on_disk.filename),
                "prompt": (
                    json.dumps(f"<{self.base_name}:{name}")
                    + " + " + json.dumps(f':{shared.opts.extra_networks_default_multiplier}')
                    + " + " + json.dumps(">")
                ),
                "local_preview": f"{path}.{shared.opts.samples_format}",
                "metadata": json.dumps(lyco_on_disk.metadata, indent=4) if lyco_on_disk.metadata else None,
                "sort_keys": {'default': index, **sort_keys},
            }

    def allowed_directories_for_previews(self):
        return [self.model_dir]

