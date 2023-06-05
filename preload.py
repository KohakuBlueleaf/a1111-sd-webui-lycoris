import os
from modules import paths


def preload(parser):
    parser.add_argument("--lyco-dir", type=str, help="Path to directory with LyCORIS networks.", default=os.path.join(paths.models_path, 'LyCORIS'))
    parser.add_argument("--lyco-patch-lora", action="store_true", help="Patch the built-in lora. Will use the lora_dir and <lora> format, but disable the <lyco> format.", default=False)
    parser.add_argument("--lyco-debug", action="store_true", help="Print extra info when using lycoris model", default=False)