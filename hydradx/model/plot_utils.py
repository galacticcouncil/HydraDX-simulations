import matplotlib.pyplot as plt
from typing import Callable
from .processing import pool_val, market_prices, value_assets, cash_out
from .amm.global_state import GlobalState


class Datastream:
    def __init__(
        self,
        group: str = '',
        instance: str = '',
        pool: str = '',
        oracle: str = '',
        agent: str = '',
        asset: str = '',
        prop: str = '',
        key: str = ''
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
        state: GlobalState
    ):
        group = self.group
        instance = self.instance
        pool = self.pool
        oracle = self.oracle
        agent = self.agent
        asset = self.asset
        prop = self.prop
        key = self.key

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
        elif agent:
            group = "agents"
            section = [agent]
        elif asset:
            group = "external_market"
            section = [asset]
        # else:
        #     group = None
        #     raise TypeError('plot() requires at least one of the following parameters: pool, agent, or asset.')
        #     # group = None

        if 'all' in [pool, agent, asset]:
            section = [key for key in getattr(state, group)]
            streams = {
                instance: self.get_stream(
                    group=group,
                    instance=instance,
                    prop=prop,
                    key=key
                )
                for instance in section
            }

            def assembly(state: GlobalState):
                return {
                    key: stream(state)
                    for key, stream in streams.items()
                }

            return assembly

        elif group == 'pools' and oracle == 'all':
            streams = {
                oracle: self.get_stream(
                    group=group,
                    instance=pool,
                    oracle=oracle,
                    prop=prop,
                    key=key
                )
                for oracle in state.pools[pool or instance].oracles
            }

            def assembly(state: GlobalState):
                return {
                    key: stream(state)
                    for key, stream in streams.items()
                }

            return assembly

        else:
            return self.get_stream(
                group=group,
                instance=instance,
                pool=pool,
                agent=agent,
                asset=asset,
                oracle=oracle,
                prop=prop,
                key=key
            )

    @staticmethod
    def get_stream(
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
        elif not key:
            if prop == 'pool_val':
                if group == 'pools':
                    return lambda state: pool_val(state, getattr(state, group)[instance])
                else:
                    raise ValueError('Cannot get pool_val for non-pool')
            elif prop == 'deposit_val':
                if group == 'agents':
                    return lambda state: value_assets(
                        market_prices(state, state.agents[instance].initial_holdings),
                        getattr(state, group)[instance]
                    )
                else:
                    raise ValueError('Cannot calculate deposit value for non-agent')
            elif prop == 'withdraw_val':
                if group == 'agents':
                    return lambda state: cash_out(state, state.agents[instance])
                else:
                    raise ValueError('Cannot calculate withdraw value for non-agent')
            elif prop == 'impermanent_loss':
                if group == 'agents':
                    return lambda state: cash_out(  # withdraw_val
                        state, state.agents[instance]
                    ) / value_assets(  # deposit_val
                        market_prices(state, state.agents[instance].initial_holdings),
                        getattr(state, group)[instance]
                    ) - 1
                else:
                    raise ValueError('Cannot calculate impermanent loss for non-agent')
            # elif prop == 'holdings_val':
            # elif prop == 'token_count':
            # elif prop == 'trade_volume':
            else:
                return lambda state: getattr(getattr(state, group)[instance], prop)
        elif not oracle:
            def get_prop(state):
                if isinstance(getattr(getattr(state, group)[instance], prop), Callable):
                    return getattr(getattr(state, group)[instance], prop)(key)
                else:
                    return getattr(getattr(state, group)[instance], prop)[key]
            return get_prop
        else:
            # oracle
            return lambda state: getattr(getattr(state, group)[instance].oracles[oracle], prop)[key]


def plot(
        events: list = None,
        pool: str = '',
        agent: str = '',
        asset: str = '',
        oracle: str = '',
        prop: str or list = '',
        key: str or list = 'all',
        subplot: plt.Subplot = None,
        label: str = '',
        time_range: tuple = ()
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

    ax: plt.Subplot = None

    if time_range:
        events = events[time_range[0]: time_range[1]]
        use_range = f'(time steps {time_range[0]} - {time_range[1]})'
    else:
        use_range = ''

    if pool:
        group = "pools"
        section = [pool]
        title = "pool"
    elif agent:
        group = "agents"
        section = [agent]
        title = "agent"
    elif asset:
        group = "external_market"
        section = [asset]
        title = "asset price"
    else:
        group = None
        raise TypeError('plot() requires at least one of the following parameters: pool, agent, or asset.')
        # group = None

    if 'all' in [pool, agent, asset]:
        section = [key for key in getattr(events[0]['state'], group)]

    if 'state' in events[0]:
        for i, instance in enumerate(section):
            if isinstance(prop, list):
                use_props = prop
            else:
                use_props = [prop]

            if (len(use_props) > 1 or i == 0) and not subplot:
                plt.figure(figsize=(20, 5))

            for p, use_prop in enumerate(use_props):

                if key == 'all':
                    if use_prop:
                        if isinstance(use_prop, str):
                            test_prop = getattr(getattr(events[0]['state'], group)[section[0]], use_prop)
                            if isinstance(test_prop, dict):
                                # e.g. prop == liquidity, which is a dict.
                                # In this case we will graph all keys in the dict.
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

                if keys == 'all':
                    use_keys = getattr(getattr(events[0]['state'], group)[instance], use_prop).keys()
                else:
                    use_keys = keys

                if len(use_keys) > 1 and p > 0:
                    # start a new line if we are doing multiple graphs at a time
                    plt.figure(figsize=(20, 5))

                for k, use_key in enumerate(use_keys):
                    ax = subplot or plt.subplot(
                        1, max(len(use_keys), len(use_props), len(section)), max(p, k, i) + 1,
                        title=f'{title}: {instance}{" " + oracle + " " or " "}{use_prop} {use_key} {use_range}'
                    )
                    y = get_datastream(
                        events=events,
                        group=group,
                        instance=instance,
                        oracle=oracle,
                        prop=use_prop,
                        key=use_key
                    )
                    x = range(len(y))
                    ax.plot(x, y, label=label)
    else:
        # if events is a dict of datastreams, not a list of events
        ax = subplot or plt.subplot(1, 1, 1)
        ax.plot(range(len(events)), [event[prop] for event in events], label=label)

    return ax


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

