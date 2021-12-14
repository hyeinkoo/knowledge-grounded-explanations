import yaml


def process_parameters_yaml() -> dict:
    with open(f'parameters.yml') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    return params


def get_yaml_parameter(name_parameter: str) -> dict:
    with open(f'parameters.yml') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    return params[name_parameter]
