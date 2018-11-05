# -*- coding: utf-8 -*-
def parse_model_config(path):
    """
    Parses yolo configuration file.
    Returns a list of blocks. 
    Each blocks describes a block in the neural network to be built. 
    Block is represented as a dictionary in the list.
    """
    module_defs = []
    with open(path) as file:
        for line in file:
            line = line.strip()
            if line:
                if line.startswith('#'):
                    # This marks a comment, so ignore
                    continue
                elif line.startswith('['):
                    # This marks the start of a new block
                    module_defs.append({})
                    module_defs[-1]['type'] = line[1:-1].strip()
                else:
                    # Contents of a block, in the form of key = value
                    key, value = line.split("=")
                    module_defs[-1][key.strip()] = value.strip()
    return module_defs
