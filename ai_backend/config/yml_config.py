import yaml


def load_config(file_path):
    with open('config/config.yaml', 'r') as file:
        return yaml.safe_load(file)
