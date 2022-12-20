import matplotlib.pyplot as plt
from typing import Callable
from .processing import pool_val, market_prices, value_assets, cash_out, impermanent_loss
from .amm.global_state import GlobalState
from numbers import Number


class Datastream:
    def __init__(
            self,
            group: str = '',
            instance: str = '',
            pool: str = '',
            oracle: str = '',
            agent: str = '',
            asset: str or list = '',
            prop: str or list = '',
            key: str or list = ''
    ):
        self.group = group
        self.instance = instance
        self.pool = pool
        self.oracle = oracle
        self.agent = agent
        self.asset = asset
        self.prop = prop
        self.key = key

    def assemble(
            self,
            state: GlobalState,
            group: str = '',
            instance: str = '',
            pool: str = '',
            oracle: str = '',
            agent: str = '',
            asset: str or list = '',
            prop: str or list = '',
            key: str or list = ''
    ):
        group = group or self.group
        instance = instance or self.instance
        pool = pool or self.pool
        oracle = oracle or self.oracle
        agent = agent or self.agent
        asset = asset or self.asset
        prop = prop or self.prop
        key = key or self.key

        """
        Recursively generate a dict from the state, using the provided keys to select the desired values.
        """

        if not (group and instance):
            if pool:
                group = "pools"
                instance = pool
            elif agent:
                group = "agents"
                instance = agent
            elif asset:
                group = "external_market"
                instance = asset

        streams = {}

        if instance == 'all':
            instance = list(getattr(state, group).keys())
        elif oracle == 'all':
            oracle = list(state.pools[pool or instance].oracles.keys())
        elif prop == 'all':
            if oracle:
                prop = list(vars(state.pools[pool or instance].oracles[oracle]).keys())
            else:
                prop = list(vars(getattr(state, group)[instance]).keys())
        elif key == 'all':
            if oracle:
                key = getattr(getattr(state, group)[instance].oracles[oracle], prop)
                if isinstance(key, dict):
                    key = list(key.keys())
                else:
                    key = ''
            else:
                key = list(getattr(getattr(state, group)[instance], prop).keys())

        if isinstance(instance, list):
            for instance in instance:
                streams[instance] = self.assemble(state, group=group, instance=instance)

        elif isinstance(oracle, list):
            for oracle in oracle:
                streams[oracle] = self.assemble(
                    state,
                    group=group, instance=instance,
                    oracle=oracle
                )

        elif isinstance(prop, list):
            for prop in prop:
                streams[prop] = self.assemble(
                    state,
                    group=group, instance=instance, oracle=oracle,
                    prop=prop
                )

        elif isinstance(key, list):
            for key in key:
                streams[key] = self.assemble(
                    state,
                    group=group, instance=instance, oracle=oracle, prop=prop,
                    key=key
                )

        else:
            return self.get_stream(
                group=group,
                instance=instance,
                oracle=oracle,
                prop=prop,
                key=key
            )

        def assembly(state: GlobalState):
            return {
                key: stream(state)
                for key, stream in streams.items()
            }

        return assembly

    @staticmethod
    def get_stream(
            path: str or list = '',
            group: str = '',
            instance: str = '',
            pool: str = '',
            oracle: str = '',
            agent: str = '',
            asset: str = '',
            prop: str = '',
            key: str = ''
    ) -> Callable:
        """
        Takes a set of parameters and returns a function which will return the appropriate data stream.
        """

        if path:
            if isinstance(path, str):
                path = [path]

            def next_key(state, k):
                return state[k]

            def assembly(state):
                return_val = state
                for k in path:
                    return_val = next_key(return_val, k)
                return return_val

            return assembly

        if pool:
            group = 'pools'
            instance = pool

        if agent:
            group = 'agents'
            instance = agent

        if asset:
            group = 'external_market'

        if group == 'external_market':
            key = asset or instance or key

        if not prop:
            return lambda state: getattr(state, group)[instance or key]
        elif not key and not oracle:
            if prop == 'pool_val':
                if group == 'pools':
                    return lambda state: state.pool_val(getattr(state, group)[instance])
                else:
                    raise ValueError('Cannot get pool_val for non-pool')
            elif prop == 'deposit_val':
                if group == 'agents':
                    return lambda state: value_assets(
                        state.market_prices(state.agents[instance].initial_holdings),
                        state.agents[instance].initial_holdings
                    )
                else:
                    raise ValueError('Cannot calculate deposit value for non-agent')
            elif prop == 'withdraw_val':
                if group == 'agents':
                    return lambda state: state.cash_out(state.agents[instance])
                else:
                    raise ValueError('Cannot calculate withdraw value for non-agent')
            elif prop == 'impermanent_loss':
                if group == 'agents':
                    return lambda state: state.impermanent_loss(instance)
                else:
                    raise ValueError('Cannot calculate impermanent loss for non-agent')
            # elif prop == 'holdings_val':
            # elif prop == 'token_count':
            # elif prop == 'trade_volume':
            else:
                return lambda state: getattr(getattr(state, group)[instance], prop)
        elif not oracle:
            # prop may be either a dict or a function
            def get_prop(state):
                if isinstance(getattr(getattr(state, group)[instance], prop), Callable):
                    return getattr(getattr(state, group)[instance], prop)(key)
                else:
                    return getattr(getattr(state, group)[instance], prop)[key]

            return get_prop
        else:
            # oracle
            if key:
                return lambda state: getattr(getattr(state, group)[instance].oracles[oracle], prop)[key]
            else:
                return lambda state: getattr(getattr(state, group)[instance].oracles[oracle], prop)


def plot(
        events: list = None,
        pool: str = '',
        agent: str = '',
        asset: str = '',
        oracle: str = '',
        prop: str or list = '',
        key: str or list = '',
        subplot: plt.Subplot = None,
        time_range: tuple = (),
        label: str = '',
        title: str = '',
        x: list or str = None,
        y: list or str = None,
):
    """
    Given several specifiers, automatically create a graph or a series of graphs as appropriate.
    Examples:
        plot(events, pool='omnipool', prop='LRNA', key='R1'):
            * plots LRNA in the R1/LRNA pool over time
        plot(events, pool='all', prop='liquidity', key=['R1', 'R2'])
            * plots R1 and R2 liquidity in all pools which have R1 or R2
        plot(events, pool='R1/R2', prop='impermanent_loss', key='all')
        plot(x='time', y='long oracle HDX price')
            * assuming a Datastream titled 'long oracle HDX price' was specified in initial_state.save_data,
             plots the associated output
        plot(y='long oracle HDX price')
            * same as above, because 'time' is default for x
    """

    if not subplot:
        plt.figure(figsize=(20, 5))
        plt.title(title)

    if time_range:
        events = events[time_range[0]: time_range[1]]
        use_range = f'(time steps {time_range[0]} - {time_range[1]})'
    else:
        use_range = ''

    if pool:
        title = f'{pool} {" " + oracle + " " or " "}{prop}{" " + key + " " if isinstance(key, str) else " "}{use_range}'
    elif agent:
        title = f'{agent} {prop}{" " + key + " " if isinstance(key, str) else " "}{use_range}'
    elif asset:
        title = f'asset price: {asset if isinstance(asset, str) else ""} {use_range}'

    if events and isinstance(events[0], Number):
        y = events
    elif not y:
        datastream = Datastream(pool=pool, agent=agent, asset=asset, oracle=oracle, prop=prop, key=key).assemble(
            events[0]['state']
        )
        y = [datastream(event['state']) for event in events]
    elif isinstance(y, str):
        title = title or y
        y = [event[y] for event in events]

    if isinstance(y[0], dict):
        for i, k in enumerate(y[0].keys()):
            if isinstance(y[1][k], Number):
                subplot = plt.subplot(1, len(y[0]), i + 1, title=f'{title} {k}')
            plot(x=x, y=[y[k] for y in y], title=f'{title} {k}', subplot=subplot)
        return
    if not x or x == 'time':
        x = range(len(y))

    ax = subplot or plt.subplot(1, 1, 1, title=title)
    ax.plot(x, y, label=label)
    return ax

    # if pool:
    #     group = "pools"
    #     section = [pool]
    #     title = "pool"
    # elif agent:
    #     group = "agents"
    #     section = [agent]
    #     title = "agent"
    # elif asset:
    #     group = "external_market"
    #     section = [asset]
    #     title = "asset price"
    # else:
    #     group = None
    #     raise TypeError('plot() requires at least one of the following parameters: pool, agent, or asset.')
    #     # group = None
    #
    # if 'all' in [pool, agent, asset]:
    #     section = [key for key in getattr(events[0]['state'], group)]
    #
    # if 'state' in events[0]:
    #     for i, instance in enumerate(section):
    #         if isinstance(prop, list):
    #             use_props = prop
    #         else:
    #             use_props = [prop]
    #
    #         if (len(use_props) > 1 or i == 0) and not subplot:
    #             plt.figure(figsize=(20, 5))
    #
    #         for p, use_prop in enumerate(use_props):
    #
    #             if key == 'all':
    #                 if use_prop:
    #                     if isinstance(use_prop, str):
    #                         test_prop = getattr(getattr(events[0]['state'], group)[section[0]], use_prop)
    #                         if isinstance(test_prop, dict):
    #                             # e.g. prop == liquidity, which is a dict.
    #                             # In this case we will graph all keys in the dict.
    #                             keys = 'all'
    #                         else:
    #                             # e.g. prop == market_cap, which is a float
    #                             keys = ['']
    #                     else:
    #                         keys = ['']
    #                 else:
    #                     # e.g. asset is specified, meaning we don't need prop or key
    #                     keys = ['']
    #             elif isinstance(key, list):
    #                 keys = key
    #             else:
    #                 keys = [key]
    #
    #             if keys == 'all':
    #                 use_keys = getattr(getattr(events[0]['state'], group)[instance], use_prop).keys()
    #             else:
    #                 use_keys = keys
    #
    #             if len(use_keys) > 1 and p > 0:
    #                 # start a new line if we are doing multiple graphs at a time
    #                 plt.figure(figsize=(20, 5))
    #
    #             for k, use_key in enumerate(use_keys):
    #                 ax = subplot or plt.subplot(
    #                     1, max(len(use_keys), len(use_props), len(section)), max(p, k, i) + 1,
    #                     title=f'{title}: {instance}{" " + oracle + " " or " "}{use_prop} {use_key} {use_range}'
    #                 )
    #                 y = get_datastream(
    #                     events=events,
    #                     group=group,
    #                     instance=instance,
    #                     oracle=oracle,
    #                     prop=use_prop,
    #                     key=use_key
    #                 )
    #                 x = range(len(y))
    #                 ax.plot(x, y, label=label)
    # else:
    #     # if events is a dict of datastreams, not a list of events
    #     ax = subplot or plt.subplot(1, 1, 1)
    #     ax.plot(range(len(events)), [event[prop] for event in events], label=label)
    #
    # return ax


def get_datastream(
        events: list,
        group: str = '',
        instance: str = '',
        pool: str = '',
        oracle: str = '',
        agent: str = '',
        asset: str = '',
        prop: str = '',
        key: str = ''
) -> list[float]:
    """
    Takes entire events array and some specifiers as arguments.
    Outputs one list of floats. Basically a list comprehension helper function.
    """

    datastream = Datastream.get_stream(
        pool=pool,
        agent=agent,
        asset=asset,
        group=group,
        instance=instance,
        prop=prop,
        key=key,
        oracle=oracle
    )
    return [datastream(event['state']) for event in events]


def best_fit_line(data_array: list[float]):
    """
    Calculate the best fit line for the given array. The x coordinates are assumed to be the indices of the array.
    Usage: pyplot.plot(*best_fit_line(data_array))
    """
    avg_x = len(data_array) / 2
    avg_y = sum(data_array) / len(data_array)

    slope = (sum([(i - avg_x) * (data_array[i] - avg_y) for i in range(len(data_array))]) /
             sum([(i - avg_x) ** 2 for i in range(len(data_array))]))
    intercept = avg_y - slope * avg_x

    return [range(len(data_array)), [x * slope + intercept for x in range(len(data_array))]]


def color_gradient(length: int, color1: tuple = (255, 0, 0), color2: tuple = (0, 0, 255)) -> list[str]:
    gradient = []
    for i in range(length):
        gradient.append('#' + (
                hex(int(color1[0] * (1 - i / length) + color2[0] * i / length))[2:].zfill(2) +
                hex(int(color1[1] * (1 - i / length) + color2[1] * i / length))[2:].zfill(2) +
                hex(int(color1[2] * (1 - i / length) + color2[2] * i / length))[2:].zfill(2)
        ))
    return gradient
