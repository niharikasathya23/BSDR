import yaml  

def load_config(file_path):
    """
    Loads a YAML configuration file and returns the contents as a dictionary.

    Args:
        file_path (str): Path to the YAML configuration file to be loaded.

    Returns:
        dict: Contents of the YAML file as a dictionary.
    """
    # Open and read the YAML file, then convert its content into a dictionary
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
