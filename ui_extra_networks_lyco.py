import json
import os
import lycoris

from modules import shared, ui_extra_networks


class ExtraNetworksPageLyCORIS(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('LyCORIS')

    def refresh(self):
        lycoris.list_available_lycos()

    def list_items(self):
        for name, lyco_on_disk in lycoris.available_lycos.items():
            path, ext = os.path.splitext(lyco_on_disk.filename)
            yield {
                "name": name,
                "filename": path,
                "preview": self.find_preview(path),
                "description": self.find_description(path),
                "search_term": self.search_terms_from_path(lyco_on_disk.filename),
                "prompt": json.dumps(f"<lyco:{name}:") + " + opts.extra_networks_default_multiplier + " + json.dumps(">"),
                "local_preview": f"{path}.{shared.opts.samples_format}",
                "metadata": json.dumps(lyco_on_disk.metadata, indent=4) if lyco_on_disk.metadata else None,
            }

    def allowed_directories_for_previews(self):
        return [shared.cmd_opts.lyco_dir]

