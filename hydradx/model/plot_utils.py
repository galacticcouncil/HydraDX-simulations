import matplotlib.pyplot as plt


def plot(
        events: list = None,
        pool: str = '',
        agent: str = '',
        asset: str = '',
        prop: str = '',
        key: str = 'all',
        subplot: plt.Subplot = None
):
    """
    Given several specifiers, automatically create a graph or a series of graphs as appropriate.
    Examples:
        plot(events, pool='omnipool', prop='LRNA', key='R1'):
            * plots LRNA in the R1/LRNA pool over time
        plot(events, pool='all', prop='liquidity', key=['R1', 'R2'])
            * plots R1 and R2 liquidity in all pools which have R1 or R2
        plot(events, pool='R1/R2', prop='impermanent_loss', key='all')
    """
    if pool:
        group = "pools"
        section = [pool]
        label = "pool"
    elif agent:
        group = "agents"
        section = [agent]
        label = "agent"
    elif asset:
        group = "external_market"
        section = [asset]
        label = "asset price"
    else:
        raise TypeError('plot() requires at least one of the following parameters: pool, agent, or asset.')

    if 'all' in [pool, agent, asset]:
        section = [key for key in getattr(events[0]['state'], group)]

    if key == 'all':
        key = None
        if prop:
            if isinstance(prop, str):
                test_prop = getattr(getattr(events[0]['state'], group)[section[0]], prop)
                if isinstance(test_prop, dict):
                    # e.g. prop == liquidity, which is a dict. In this case we will graph all keys in the dict.
                    keys = 'all'
                else:
                    # e.g. prop == market_cap, which is a float
                    keys = ['']
            else:
                keys = ['']
        else:
            # e.g. asset is specified, meaning we don't need prop or key
            keys = ['']
    elif isinstance(key, list):
        keys = key
    else:
        keys = [key]

    if len(keys) == 1:
        plt.figure(figsize=(20, 5))

    for instance in section:
        if isinstance(prop, list):
            use_props = prop
        else:
            use_props = [prop]

        for p, prop in enumerate(use_props):
            if keys == 'all':
                use_keys = getattr(getattr(events[0]['state'], group)[instance], prop).keys()
            else:
                use_keys = keys

            if len(use_keys) > 1 or keys == 'all':
                # start a new line if we are doing multiple graphs at a time
                plt.figure(figsize=(20, 5))

            for k, key in enumerate(use_keys):
                ax: plt.Subplot = subplot or plt.subplot(
                    1, max(len(use_keys), len(use_props)), max(p, k)+1, title=f'{label}: {instance} {prop} {key}'
                )
                y = get_datastream(events=events, group=group, instance=instance, prop=prop, key=key)
                x = range(len(y))
                ax.plot(x, y)


def get_datastream(
        events: list,
        group: str = '',
        instance: str = '',
        pool: str = '',
        agent: str = '',
        asset: str = '',
        prop: str = '',
        key: str = ''
) -> list[float]:
    """
    Takes entire events array and some specifiers as arguments.
    Outputs one list of floats. Basically a list comprehension helper function.
    """

    if pool:
        group = 'pools'
        instance = pool
    if agent:
        group = 'agents'
        instance = agent
    if asset:
        group = 'external_market'
        key = asset

    if not prop:
        return [getattr(event['state'], group)[instance or key] for event in events]
    if not key:
        return [getattr(getattr(event['state'], group)[instance], prop) for event in events]

    try:
        return [getattr(getattr(event['state'], group)[instance], prop)[key] for event in events]
    except KeyError:
        # this may occur, for example, if a certain pool doesn't contain the assets specified by *key*
        return []
