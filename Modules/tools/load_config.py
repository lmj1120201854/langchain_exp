import os
import yaml

def load_env_config(file_path):
    # print(os.getcwd())
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
        for key1, value1 in config.items():
            for key2, value2 in value1.items():
                # print(f"{key1}_{key2} = {value2}")
                os.environ[f"{key1}_{key2}"] = str(value2)

def load_module_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
        return config

if __name__ == "__main__":
    env_config_path = 'langchain_exp\configs\env.yaml'
    module_config_path = 'langchain_exp\configs\module.yaml'
    # load_env_config(env_config_path)
    config = load_module_config(module_config_path)
    print(config)