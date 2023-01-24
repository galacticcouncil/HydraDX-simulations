import matplotlib.pyplot as plt
from typing import Callable
from .amm.global_state import GlobalState
from numbers import Number


def get_datastream(
        events: list,
        group: str = '',
        instance: str = '',
        pool: str = '',
        oracle: str = '',
        agent: str = '',
        asset: str or list = '',
        prop: str or list = '',
        key: str or list = ''
):
    initial_state = events[0]

    """
    generate a lit of values from the state, using the given parameters
    ONE of these may be a list or 'all', in which case the function will return a dict of lists
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

    if instance == 'all':
        instance = list(getattr(initial_state, group).keys())
    elif oracle == 'all':
        oracle = list(initial_state.pools[pool or instance].oracles.keys())
    elif prop == 'all':
        if oracle:
            prop = list(filter(
                lambda x: isinstance(x, dict),
                list(vars(initial_state.pools[pool or instance].oracles[oracle]).keys())
            ))
        else:
            prop = list(vars(getattr(initial_state, group)[instance]).keys())
    elif key == 'all':
        if oracle:
            key = getattr(getattr(initial_state, group)[instance].oracles[oracle], prop)
            if isinstance(key, dict):
                key = list(key.keys())
            else:
                key = ''
        else:
            key = list(getattr(getattr(initial_state, group)[instance], prop).keys())
    elif prop and key == '':
        if hasattr(getattr(initial_state, group)[instance], prop):
            if isinstance(getattr(getattr(initial_state, group)[instance], prop), dict):
                key = list(getattr(getattr(initial_state, group)[instance], prop).keys())

    if isinstance(instance, list):
        return {
            i: get_single_stream(events, group=group, instance=i, oracle=oracle, prop=prop, key=key)
            for i in instance
        }

    elif isinstance(oracle, list):
        return {
            i: get_single_stream(events, group=group, instance=instance, oracle=i, prop=prop, key=key)
            for i in oracle
        }

    elif isinstance(prop, list):
        return {
            i: get_single_stream(events, group=group, instance=instance, oracle=oracle, prop=i, key=key)
            for i in prop
        }

    elif isinstance(key, list):
        return {
            i: get_single_stream(events, group=group, instance=instance, oracle=oracle, prop=prop, key=i)
            for i in key
        }

    else:
        return get_single_stream(
            events,
            group=group,
            instance=instance,
            oracle=oracle,
            prop=prop,
            key=key
        )


def get_single_stream(
        events,
        group: str = '',
        instance: str = '',
        oracle: str = '',
        prop: str = '',
        key: str = ''
) -> list[Number]:
    """
    Takes a set of parameters and returns a list of values from the state
    """

    if group == 'external_market':
        key = instance or key

    initial_state = events[0]
    if hasattr(initial_state, prop):
        return [getattr(event, prop)(getattr(event, group)[instance]) for event in events]
    elif not prop:
        return [getattr(event, group)[instance or key] for event in events]
    elif not oracle:
        # prop may be either a dict, a function or a number
        if isinstance(getattr(getattr(initial_state, group)[instance], prop), Callable):
            return [getattr(getattr(event, group)[instance], prop)
                    (getattr(event, group)[instance], key) for event in events]
        elif isinstance(getattr(getattr(initial_state, group)[instance], prop), dict):
            return [getattr(getattr(event, group)[instance], prop)[key] for event in events]
        else:
            return [getattr(getattr(event, group)[instance], prop) for event in events]
    else:
        # oracle
        if key:
            return [getattr(getattr(event, group)[instance].oracles[oracle], prop)[key] for event in events]
        else:
            return [getattr(getattr(event, group)[instance].oracles[oracle], prop) for event in events]


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
        title = f'{pool}{" " + oracle + " " or " "}{prop}'
    elif agent:
        title = f'{agent} {prop}'
    elif asset:
        title = f'asset price:'
        key = asset

    if events and isinstance(events[0], Number):
        y = events
    elif not y:
        y = get_datastream(events, pool=pool, agent=agent, asset=asset, oracle=oracle, prop=prop, key=key)
    elif isinstance(y, str):
        title = title or y
        y = [event[y] for event in events]

    if isinstance(y, dict):
        for i, k in enumerate(y.keys()):
            if isinstance(y[k][0], Number):
                subplot = plt.subplot(1, len(y.keys()), i + 1, title=f'{title} {k}')
            plot(x=x, y=y[k], title=f'{title} {k}', subplot=subplot)
        return
    if not x or x == 'time':
        x = range(len(y))

    ax = subplot or plt.subplot(1, 1, 1, title=f'{title} {key} {use_range}')
    ax.plot(x, y, label=label)
    return ax


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

