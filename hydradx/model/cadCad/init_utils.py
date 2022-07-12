def get_configuration(timesteps: int):
    config_dict = {
        'N': 1,  # number of monte carlo runs
        'T': range(timesteps),  # number of timesteps - 147439 is the length of uniswap_events
        'M': {'timesteps': [timesteps]},  # simulation parameters
    }
    return config_dict
