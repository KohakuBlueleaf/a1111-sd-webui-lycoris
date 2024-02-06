import os
from modules import paths


def preload(parser):
    parser.add_argument("--lyco-dir", type=str, help="Path to directory with Lora networks.", default=os.path.join(paths.models_path, 'LyCORIS'))
