import copy

import pandas as pd

# from amm.amm import WorldState
from .amm import omnipool_amm
from .amm.amm import Agent, Market


def postprocessing(events, params_to_include: list[str] = ()):
    """
    Definition:
    Refine and extract metrics from the simulation

    optional parameters:
     * 'hold_val': adds a ['hold_val'] key to the agents dataframe, equal to the total value of all
      that agent's liquidity provisions at the current spot price.

     * 'withdraw_val': adds a ['withdraw_val'] key to the agents dataframe, equal to the total value
      of all that agent's liquidity provisions if sold and then denominated in LRNA.

     * 'pool_val': adds a ['pool_val'] key to each step in the dataframe, indicating the total protocol-owned
      value held by all pools in the omnipool, denominated in LRNA
    """
    token_names = [pool.name for pool in events[0]['WorldState'].exchange.pool_list]
    # n = len(events[0]['AMM']['R'])
    agent_d = {'simulation': [], 'subset': [], 'run': [], 'substep': [], 'timestep': [], 'q': [], 'agent_label': []}

    exchange_d = {
        "timestep": [],
        "L": [],
        'simulation': [],
        'substep': [],
        'subset': [],
        'run': []
    }
    exchange_d.update({f'R-{i}': [] for i in token_names})
    exchange_d.update({f'Q-{i}': [] for i in token_names})
    exchange_d.update({f'B-{i}': [] for i in token_names})
    exchange_d.update({f'S-{i}': [] for i in token_names})
    exchange_d.update({f'P-{i}': [] for i in token_names})

    # agent_d.update({f'p-{i}': [] for i in range(token_names)})
    agent_d.update({f'r-{i}': [] for i in token_names})
    agent_d.update({f's-{i}': [] for i in token_names})

    params_to_include = set(params_to_include)
    agent_params = {
        'hold_val',
        'withdraw_val',
    }
    exchange_params = {
        'pool_val'
    }
    unrecognized_params = params_to_include.difference(agent_params | exchange_params)
    if unrecognized_params:
        raise ValueError(f'Unrecognized parameter {unrecognized_params}')

    # add optional params to the dictionaries
    for key in params_to_include & agent_params:
        agent_d[key] = []
    for key in params_to_include & exchange_params:
        exchange_d[key] = []

    pool_tokens = [pool.assetName for pool in events[0]['WorldState'].exchange.pool_list]

    # build the DFs
    for (n, step) in enumerate(events):
        omnipool: omnipool_amm.OmniPool = step['WorldState'].exchange
        agents: dict[str, omnipool_amm.OmnipoolAgent] = step['WorldState'].agents

        # add items to exchange dictionary
        for token in pool_tokens:
            exchange_d[f'R-{token}'].append(omnipool.pool(token).assetQuantity)
            exchange_d[f'Q-{token}'].append(omnipool.pool(token).lrnaQuantity)
            exchange_d[f'B-{token}'].append(omnipool.pool(token).sharesOwnedByProtocol)
            exchange_d[f'S-{token}'].append(omnipool.pool(token).shares)
            exchange_d[f'P-{token}'].append(omnipool.pool(token).ratio)
        exchange_d['L'].append(omnipool.L)
        for key in ['simulation', 'subset', 'run', 'substep', 'timestep']:
            exchange_d[key].append(step[key])

        # optional exchange parameters
        if 'pool_val' in exchange_d:
            exchange_d['pool_val'].append(pool_val(omnipool))

        # add items to agents dictionary
        for (a, agent_name) in enumerate(agents):
            for token in pool_tokens:
                # agent_d[f'p-{i}'].append(
                #     agents[agent_name].position(omnipool.pool(token).shareToken).price
                #     if agents[agent_name].position(omnipool.pool(token).shareToken) else 0
                # )
                agent_d[f's-{token}'].append(agents[agent_name].holdings(omnipool.pool(token).shareToken) or 0)
                agent_d[f'r-{token}'].append(agents[agent_name].holdings(token) or 0)
            agent_d['agent_label'].append(agent_name)
            agent_d['q'].append(agents[agent_name].q)
            for key in ['simulation', 'subset', 'run', 'substep', 'timestep']:
                agent_d[key].append(step[key])

            # optional agent parameters
            if 'withdraw_val' in agent_d:
                agent_d['withdraw_val'].append(val_pool(agents[agent_name], omnipool))
            if 'hold_val' in agent_d:
                agent_d['hold_val'].append(val_hold(agents[agent_name], omnipool))

    df = pd.DataFrame(exchange_d)
    agent_df = pd.DataFrame(agent_d)

    # subset to last substep
    df = df[df['substep'] == df.substep.max()]
    agent_df = agent_df[agent_df['substep'] == agent_df.substep.max()]

    return df, agent_df


def val_pool(agent: Agent, market: Market) -> float:
    """ How much are all this agent's exchange shares worth if sold off, denominated in (pre-sale) LRNA? """
    market_copy = copy.deepcopy(market)
    agent_copy = copy.deepcopy(agent)
    return (
        agent_copy.erase_external_holdings()
        .remove_all_liquidity(market_copy)
        .value_holdings(market)
    )


def val_hold(agent: Agent, market: Market):
    """ How much are this agent's holdings in the exchange worth at the current spot price? """
    agent_copy = copy.deepcopy(agent)
    return (
        agent_copy.erase_external_holdings()
        .value_holdings(market)
    )

    # state = get_state_from_row(row)
    # agent = orig_agent_d[row['agent_label']]
    # value = amm.value_assets(state, agent)
    # return value


def pool_val(market: omnipool_amm.OmniPool):
    """ The total value of all pool shares owned by the protocol, denominated in LRNA. """
    return sum([
        pool.sharesOwnedByProtocol / pool.shares * pool.ratio  # * (1 - market.assetFee)
        for pool in market.pool_list
    ])
