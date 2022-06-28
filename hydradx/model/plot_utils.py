import matplotlib.pyplot as plt


def plot(events: list = None, pool: str = '', agent: str = '', asset: str = '', prop: str = ''):
    datastream = [event['state'] for event in events]
    keyword = ''
    target = ''
    if pool:
        keyword = pool
        target = "pool"
        datastream = [event['state'].pools for event in events]
    elif agent:
        keyword = agent
        target = "agent"
        datastream = [event['state'].agents for event in events]
    elif asset:
        keyword = asset
        target = "asset"
        datastream = [event['state'].external_market for event in events]

    if not keyword:
        raise TypeError('plot() requires at least one of the following parameters: pool, agent, or asset.')

    if keyword == 'all':
        for key in datastream[0]:
            plt.figure(figsize=(20, 5))
            plot_stream(datastream, key, prop, label=f"{target} '{key}'")
    elif keyword in datastream[0]:
        plt.figure(figsize=(20, 5))
        plot_stream(datastream, keyword, prop, label=f"{target} '{keyword}'")
    else:
        raise KeyError(f'{target} not found.')


def plot_stream(datastream: list, key: str = '', prop: str = '', label: str = '', subplot: plt.Subplot = None):
    """
    'datastream' should be in the form [{'agent1': <Agent>, 'agent2': <Agent>}, {'agent1: <Agent>'...]
    'key' in this case would be 'agent1' or 'agent2'.
    'prop' is the property of the object that we're graphing, such as 'holdings'.
      - if the object is a number, this can be skipped
    'label' is the prefix of the title to put on the graph: "agent 'agent1'" for instance.
    """
    if key:
        # now we do a couple more list comprehensions to get it down to just a list of floats
        first_step = datastream[0][key]
        # first_step here should be dict of [str: object or float]
        if prop:
            if hasattr(first_step, prop):
                datastream = [getattr(step[key], prop) for step in datastream]
            else:
                raise AttributeError(f'"{type(datastream[0])}" object has no attribute {prop}.')
        else:
            datastream = [step[key] for step in datastream]

        first_step = datastream[0]
        # first_step here should be either dict or float

        if isinstance(first_step, dict):
            # if it's a dict, we'll break it down into a list of floats and plot each separately

            for i, k in enumerate(first_step):
                bounds = (1, len(first_step.keys()), i + 1)
                subplot = plt.subplot(*bounds, title=label + f' {prop}-{k}')
                plot_stream(
                    datastream=[step[k] for step in datastream],
                    subplot=subplot
                )
            return

    ax: plt.Subplot = subplot or plt.subplot(1, 3, 1, title=f'{label} {prop}')
    ax.plot(range(len(datastream)), datastream)