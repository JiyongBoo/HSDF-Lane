import importlib
import os
import sys


def load_config_module(config_file):
    config_dir = os.path.dirname(os.path.abspath(config_file))
    project_root = os.path.abspath(os.path.join(config_dir, os.pardir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    spec = importlib.util.spec_from_file_location("tuly.config", config_file)
    configs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(configs)
    return configs