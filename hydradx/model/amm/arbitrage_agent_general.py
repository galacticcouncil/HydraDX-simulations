from hydradx.model.amm.agents import Agent
from hydradx.model.amm.amm import AMM
from hydradx.model.amm.centralized_market import CentralizedMarket
from hydradx.model.amm.omnipool_amm import OmnipoolState


# note that this function mutates exchanges, agents, and max_liquidity
def get_arb_opps(
        exchanges: dict[str: AMM],
        config: list[dict],
        max_liquidity: dict[str: dict[str: float]]
) -> list[tuple[float, int]]:
    """
    Look through provided dictionary of exchanges and find the best arbitrage opportunities.
    """
    arb_opps = []

    for i, arb_cfg in enumerate(config):
        exchange_names = list(arb_cfg['exchanges'].keys())
        ex_1: AMM = exchanges[exchange_names[0]]
        ex_2: AMM = exchanges[exchange_names[1]]

        tkn_pair_1 = arb_cfg['exchanges'][exchange_names[0]]
        tkn_pair_2 = arb_cfg['exchanges'][exchange_names[1]]

        if max_liquidity[exchange_names[0]][tkn_pair_1[0]] > 0 and max_liquidity[exchange_names[1]][tkn_pair_2[1]] > 0:
            ex_1_sell_price = ex_1.sell_spot(tkn_pair_1[1], tkn_pair_1[0]) * (1 - arb_cfg['buffer'])
            ex_1_buy_price = ex_1.buy_spot(tkn_pair_1[1], tkn_pair_1[0])

            if ex_1_sell_price:
                ex_2_buy_price = ex_2.buy_spot(tkn_pair_2[1], tkn_pair_2[0])
                if ex_2_buy_price and ex_2_buy_price < ex_1_sell_price:  # buy from ex2, sell to ex1
                    arb_opps.append(((ex_1_sell_price - ex_2_buy_price) / ex_2_buy_price, i))

            if ex_1_buy_price:
                ex_2_sell_price = ex_2.sell_spot(tkn_pair_2[1], tkn_pair_2[0])
                if ex_2_sell_price:
                    ex_2_sell_price *= (1 - arb_cfg['buffer'])
                    if ex_2_sell_price > ex_1_buy_price: # buy from ex1, sell to ex2
                        arb_opps.append(((ex_2_sell_price - ex_1_buy_price) / ex_1_buy_price, i))

    arb_opps.sort(key=lambda x: x[0], reverse=True)
    return arb_opps


def get_arb_swaps(
        exchanges: dict[str: AMM],
        config: list[dict],
        max_liquidity: dict[str: dict[str: float]] = None,
        max_iters: int = 10,
        precision: float = 1e-10
):
    """
    return a list of swaps that can be executed to take advantage of the best arbitrage opportunities
    """
    if not max_liquidity:
        max_liquidity = {ex_name: {tkn: 1000000000 for tkn in ex.asset_list} for ex_name, ex in exchanges.items()}

    arb_opps = get_arb_opps(exchanges, config, max_liquidity)
    all_swaps = []
    test_exchanges = {ex_name: ex.copy() for ex_name, ex in exchanges.items()}
    test_agents = {ex_name: Agent(holdings=max_liquidity[ex_name].copy()) for ex_name in exchanges}
    arbs_index = 0
    while arb_opps:
        arb_cfg = config[arb_opps[arbs_index][1]]
        swap = process_next_swap(
            exchanges=test_exchanges,
            swap_config=arb_cfg,
            agents=test_agents,
            max_liquidity=max_liquidity,
            max_iters=max_iters,
            precision=precision
        )
        for ex_name in arb_cfg['exchanges'].keys():
            if exchanges[ex_name].fail:
                raise AssertionError('max liquidity exceeded?')
        if swap:
            all_swaps.append(swap)
        elif len(arb_opps) > arbs_index + 1:
            arbs_index += 1
            continue
        arbs_index = 0
        new_arb_opps = get_arb_opps(test_exchanges, config, max_liquidity)
        if arb_opps and new_arb_opps and arb_opps[0][0] == new_arb_opps[0][0]:
            break
        arb_opps = new_arb_opps

    return all_swaps


def process_next_swap(
        exchanges: dict[str: AMM],
        swap_config: dict,
        agents: dict[str: Agent],
        max_liquidity: dict[str: [dict[str: float]]],
        precision: float = 1e-10,
        max_iters: int = 10
):
    """
    Calculate what the appropriate size of the swap should be, and execute it on the virtual exchanges
    and virtual agents to keep track of the results.
    """
    buffer = swap_config['buffer']
    exchange_names = list(swap_config['exchanges'].keys())
    tkn_pairs = swap_config['exchanges']
    slippage_tolerance = {ex_name: buffer / 2 for ex_name in exchange_names}

    # make sure the assets are specified in the correct order
    for ex_name in swap_config['exchanges']:
        if isinstance(exchanges[ex_name], CentralizedMarket):
            if tkn_pairs[ex_name] not in exchanges[ex_name].order_book:
                raise ValueError(f'Asset pair {tkn_pairs[ex_name]} not found in order book. Are they ordered wrong?')

    # all denominated in tkn_pair[1]
    buy_price = {
        **{ex_name: exchanges[ex_name].buy_spot(
            tkn_buy=tkn_pairs[ex_name][0], tkn_sell=tkn_pairs[ex_name][1]
        ) for ex_name in exchange_names},
        'feeless': {ex_name: exchanges[ex_name].buy_spot(
            tkn_buy=tkn_pairs[ex_name][0], tkn_sell=tkn_pairs[ex_name][1], fee=0
        ) for ex_name in exchange_names}
    }
    sell_price = {
        **{ex_name: exchanges[ex_name].sell_spot(
            tkn_buy=tkn_pairs[ex_name][1], tkn_sell=tkn_pairs[ex_name][0]
        ) for ex_name in exchange_names},
        'feeless': {ex_name: exchanges[ex_name].sell_spot(
            tkn_buy=tkn_pairs[ex_name][1], tkn_sell=tkn_pairs[ex_name][0], fee=0
        ) for ex_name in exchange_names}
    }

    buy_ex, sell_ex = None, None
    for ex_name, other_ex in (exchange_names, reversed(exchange_names)):
        if buy_price[ex_name] and sell_price[other_ex] and buy_price[ex_name] < sell_price[other_ex]:
            buy_ex = ex_name
            sell_ex = other_ex
            break

    if not (buy_ex and sell_ex):
        return {}

    amt = calculate_arb_amount(
        buy_ex=exchanges[buy_ex], sell_ex=exchanges[sell_ex],
        buy_ex_tkn_pair=tkn_pairs[buy_ex], sell_ex_tkn_pair=tkn_pairs[sell_ex],
        buffer=buffer, min_amt=1e-6,
        buy_ex_max_sell=max_liquidity[buy_ex][tkn_pairs[buy_ex][0]],
        sell_ex_max_sell=max_liquidity[sell_ex][tkn_pairs[sell_ex][0]],
        precision=precision,
        max_iters=max_iters
    )

    if amt == 0:
        return ()

    init_amt = {
        buy_ex: agents[buy_ex].holdings[tkn_pairs[buy_ex][1]],
        sell_ex: agents[sell_ex].holdings[tkn_pairs[sell_ex][1]]
    }
    exchanges[buy_ex].swap(
        agent=agents[buy_ex],
        tkn_buy=tkn_pairs[buy_ex][0],
        tkn_sell=tkn_pairs[buy_ex][1],
        buy_quantity=amt
    )
    exchanges[sell_ex].swap(
        agent=agents[sell_ex],
        tkn_buy=tkn_pairs[sell_ex][1],
        tkn_sell=tkn_pairs[sell_ex][0],
        sell_quantity=amt
    )
    amt_in = {
        buy_ex: init_amt[buy_ex] - agents[buy_ex].holdings[tkn_pairs[buy_ex][1]],
        sell_ex: amt
    }
    amt_out = {
        buy_ex: amt,
        sell_ex: agents[sell_ex].holdings[tkn_pairs[sell_ex][1]] - init_amt[sell_ex]
    }
    if tkn_pairs[buy_ex][1] in max_liquidity[buy_ex]:
        max_liquidity[buy_ex][tkn_pairs[buy_ex][1]] -= amt_in[buy_ex]
    if tkn_pairs[buy_ex][0] in max_liquidity[buy_ex]:
        max_liquidity[buy_ex][tkn_pairs[buy_ex][0]] += amt_out[buy_ex]
    if tkn_pairs[sell_ex][0] in max_liquidity[sell_ex]:
        max_liquidity[sell_ex][tkn_pairs[sell_ex][0]] -= amt_in[sell_ex]
    if tkn_pairs[sell_ex][1] in max_liquidity[sell_ex]:
        max_liquidity[sell_ex][tkn_pairs[sell_ex][1]] += amt_out[sell_ex]
    swap = {
        buy_ex: {
            'trade': 'buy',
            'buy_asset': tkn_pairs[buy_ex][0],
            'sell_asset': tkn_pairs[buy_ex][1],
            'amount': amt,
        },
        sell_ex: {
            'trade': 'sell',
            'buy_asset': tkn_pairs[sell_ex][1],
            'sell_asset': tkn_pairs[sell_ex][0],
            'amount': amt,
        }
    }
    if isinstance(exchanges[buy_ex], CentralizedMarket):
        swap[buy_ex].update({'price': buy_price['feeless'][buy_ex] * (1 + slippage_tolerance[buy_ex])})
    else:
        swap[buy_ex].update({'max_sell': amt_in[buy_ex] * (1 + slippage_tolerance[buy_ex])})
    if isinstance(exchanges[sell_ex], CentralizedMarket):
        swap[sell_ex].update({'price': sell_price['feeless'][sell_ex] * (1 - slippage_tolerance[sell_ex])})
    else:
        swap[sell_ex].update({'min_buy': amt_out[sell_ex] * (1 - slippage_tolerance[sell_ex])})
    return swap


def calculate_arb_amount(
        buy_ex: AMM,
        sell_ex: AMM,
        sell_ex_tkn_pair: tuple[str, str],
        buy_ex_tkn_pair: tuple[str, str],
        buffer: float = 0.0,
        min_amt: float = 1e-18,
        sell_ex_max_sell: float = None,
        buy_ex_max_sell: float = None,
        precision: float = 0,
        max_iters: int = 50
) -> float:
    if min_amt < 1e-18:
        min_amt = 1e-18
    """
    For a given token pair, calculate the optimum allowable trade size.
    """
    # we will buy buy_ex_tkn_pair[0] on buy_ex and sell sell_ex_tkn_pair[0] on sell_ex
    test_agent = Agent(holdings={tkn: float('inf') for tkn in buy_ex_tkn_pair + sell_ex_tkn_pair})
    test_ex_buy = buy_ex.copy()
    test_ex_sell = sell_ex.copy()
    test_ex_buy.swap(
        test_agent, tkn_buy=buy_ex_tkn_pair[0], tkn_sell=buy_ex_tkn_pair[1], buy_quantity=min_amt
    )
    test_ex_sell.swap(
        test_agent, tkn_sell=sell_ex_tkn_pair[0], tkn_buy=sell_ex_tkn_pair[1], sell_quantity=min_amt
    )
    buy_price = test_ex_buy.buy_spot(tkn_buy=buy_ex_tkn_pair[0], tkn_sell=buy_ex_tkn_pair[1])
    sell_price = test_ex_sell.sell_spot(tkn_sell=sell_ex_tkn_pair[0], tkn_buy=sell_ex_tkn_pair[1]) * (1 - buffer)

    if buy_price > sell_price or test_ex_buy.fail or test_ex_sell.fail:
        return 0

    # we use binary search to find the amount that can be swapped
    buy_ex_max_sell = buy_ex_max_sell if buy_ex_max_sell is not None else float('inf')
    max_buy = buy_ex.calculate_buy_from_sell(
        tkn_buy=buy_ex_tkn_pair[0], tkn_sell=buy_ex_tkn_pair[1], sell_quantity=buy_ex_max_sell
    )
    amt_low = min_amt
    amt_high = min(
        sell_ex_max_sell if sell_ex_max_sell is not None else float('inf'),
        buy_ex.buy_limit(tkn_buy=buy_ex_tkn_pair[0], tkn_sell=buy_ex_tkn_pair[1]),
        sell_ex.sell_limit(tkn_sell=sell_ex_tkn_pair[0], tkn_buy=sell_ex_tkn_pair[1])
    )
    amt = amt_high
    for i in range(max_iters):
        test_ex_buy = buy_ex.copy()
        test_ex_sell = sell_ex.copy()
        if not isinstance(test_ex_buy, CentralizedMarket):
            test_ex_buy.swap(
                test_agent, tkn_buy=buy_ex_tkn_pair[0], tkn_sell=buy_ex_tkn_pair[1], buy_quantity=amt
            )
            buy_price = test_ex_buy.buy_spot(buy_ex_tkn_pair[0], buy_ex_tkn_pair[1])
        if not isinstance(test_ex_sell, CentralizedMarket):
            test_ex_sell.swap(
                test_agent, tkn_sell=sell_ex_tkn_pair[0], tkn_buy=sell_ex_tkn_pair[1], sell_quantity=amt
            )
            sell_price = test_ex_sell.sell_spot(sell_ex_tkn_pair[0], sell_ex_tkn_pair[1]) * (1 - buffer)

        if test_ex_buy.fail or test_ex_sell.fail or buy_price > sell_price or amt > max_buy:
            amt_high = amt
        else:
            amt_low = amt
            if amt_low == amt_high:  # full amount can be traded
                break
            elif abs(1 - buy_price / sell_price) <= precision:
                break  # close enough!

        amt = amt_low + (amt_high - amt_low) / 2

    if amt_low == min_amt:
        return 0
    else:
        return amt_low


def flatten_swaps(swaps):
    """
    Take a list of swap pairs and flatten it into a list of individual swaps.
    """
    return [{'exchange': exchange, **trade[exchange]} for trade in swaps for exchange in trade]


def execute_arb(exchanges: dict[str: AMM], agent: Agent, swaps: list[dict]):
    if len(swaps) == 0:
        return
    for i, swap in enumerate((flatten_swaps(swaps) if len(swaps[0].keys()) == 2 else swaps)):
        tkn_buy = swap['buy_asset']
        tkn_sell = swap['sell_asset']
        ex = exchanges[swap['exchange']]
        ex.fail = ''
        if swap['trade'] == 'buy':
            ex.swap(agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=swap['amount'])
        elif swap['trade'] == 'sell':
            ex.swap(agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=swap['amount'])
        else:
            raise ValueError('Incorrect trade type.')


def calculate_profit(init_agent, agent, asset_map=None):
    asset_map = {} if asset_map is None else asset_map
    profit_asset = {tkn: agent.holdings[tkn] - init_agent.holdings[tkn] for tkn in agent.holdings}
    profit = {}

    for tkn in profit_asset:
        mapped_tkn = tkn if tkn not in asset_map else asset_map[tkn]
        if mapped_tkn not in profit:
            profit[mapped_tkn] = 0
        profit[mapped_tkn] += profit_asset[tkn]

    # if sum([profit_asset[k] for k in profit_asset]) != sum([profit[k] for k in profit]):
    #     raise
    return profit


def combine_swaps(
        exchanges: dict[str, AMM],
        agent: Agent,
        swaps: list[dict],
        equivalency_map: dict[str, str],
        max_liquidity: dict[str, dict[str, float]] = None,
        # ref_exchange: AMM = None
):
    """
    take the list of swaps and try to get the same result more efficiently
    in particular, make sure to buy *at least* as much of each asset as the net from the original list
    """
    net_swaps = {}
    return_swaps = []

    if len(swaps) == 0:
        return swaps
    if len(swaps[0].keys()) == 2:
        swaps = flatten_swaps(swaps)

    for ex_name, ex in exchanges.items():

        test_agent = agent.copy()
        test_ex = ex.copy()
        default_swaps = list(filter(lambda s: s['exchange'] == ex_name, swaps))

        for swap in default_swaps:
            tkn_sell = swap['sell_asset']
            tkn_buy = swap['buy_asset']
            if swap['trade'] == 'buy':
                test_ex.swap(test_agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=swap['amount'])
            else:
                test_ex.swap(test_agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=swap['amount'])
        net_swaps[ex_name] = {tkn: test_agent.holdings[tkn] - agent.holdings[tkn] for tkn in ex.asset_list}
        max_liquidity_ex = (
            max_liquidity[ex_name] if ex_name in max_liquidity
            else max_liquidity['cex'][ex_name]
        ) if max_liquidity else {tkn: float('inf') for tkn in ex.asset_list}

        # default_profit = calculate_profit(agent, test_agent, asset_map=asset_map)
        # default_profit_usd = ref_exchange.value_assets(assets=default_profit, equivalency_map=asset_map)

        test_ex = ex.copy()
        test_agent = agent.copy()
        optimized_swaps = []

        buy_tkns = {tkn: 0 for tkn in ex.asset_list}
        buy_tkns.update({
            tkn: quantity for tkn, quantity in
            sorted(
                filter(lambda x: x[1] > 0, net_swaps[ex_name].items()),
                key=lambda x: ex.value_assets({x[0]: x[1]}, equivalency_map=equivalency_map)
            )
        })

        sell_tkns = {
            tkn: -quantity for tkn, quantity in
            filter(lambda x: x[1] < 0, net_swaps[ex_name].items())
        }
        i = 0

        while sum(buy_tkns.values()) > 0 and i < 3:
            i += 1
            for tkn_buy, buy_quantity in buy_tkns.items():
                if buy_quantity <= 0:
                    continue
                # order possible sell tokens according to availability and price
                best_sell_tkns = {
                    tkn: (sell_tkns[tkn], price)
                    for tkn, price in filter(
                        lambda x: x[1] > 0 and x[0] != tkn_buy,  # all tokens we want to sell for which there is a price
                        {
                            x: test_ex.sell_spot(x, tkn_buy)
                            or (
                                   test_ex.buy_spot(tkn_buy, x)
                                   if test_ex.buy_spot(tkn_buy, x) else 0
                               )
                            for x in sell_tkns
                        }.items()
                    )
                }
                for tkn_sell in best_sell_tkns:
                    sell_quantity, price = best_sell_tkns[tkn_sell]
                    previous_tkn_sell = test_agent.holdings[tkn_sell]
                    previous_tkn_buy = test_agent.holdings[tkn_buy]
                    max_buy = test_ex.calculate_buy_from_sell(
                        tkn_sell=tkn_sell,
                        tkn_buy=tkn_buy,
                        sell_quantity=min(sell_quantity, max_liquidity_ex[tkn_sell])
                    )
                    if max_buy <= 0:
                        continue
                    if max_buy <= buy_tkns[tkn_buy]:
                        # buy as much as we can without going over sell_quantity
                        test_ex.swap(test_agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=max_buy)
                        optimized_swaps.append({
                            'exchange': ex_name,
                            'trade': 'buy',
                            'buy_asset': tkn_buy,
                            'sell_asset': tkn_sell,
                            'amount': max_buy
                        })
                        buy_tkns[tkn_buy] -= test_agent.holdings[tkn_buy] - previous_tkn_buy
                    else:
                        # buy enough to satisfy buy_quantity
                        test_ex.swap(test_agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=buy_tkns[tkn_buy])
                        optimized_swaps.append({
                            'exchange': ex_name,
                            'trade': 'buy',
                            'buy_asset': tkn_buy,
                            'sell_asset': tkn_sell,
                            'amount': buy_tkns[tkn_buy]
                        })
                        buy_tkns[tkn_buy] = 0

                    sell_tkns[tkn_sell] -= previous_tkn_sell - test_agent.holdings[tkn_sell]
                    max_liquidity_ex[tkn_sell] -= previous_tkn_sell - test_agent.holdings[tkn_sell]
                    max_liquidity_ex[tkn_buy] += test_agent.holdings[tkn_buy] - previous_tkn_buy
                    if buy_tkns[tkn_buy] <= 0:
                        break
        if sum(buy_tkns.values()) > 0:
            return_swaps += default_swaps
        else:
            return_swaps += optimized_swaps

    return return_swaps
    #
    #     stablecoin = ex.stablecoin if hasattr(ex, 'stablecoin') else 'USD'
    #     for tkn_sell in sell_tkns:
    #         if sell_tkns[tkn_sell] > 0:
    #             test_ex.swap(
    #                 agent=test_agent,
    #                 tkn_buy=stablecoin,
    #                 tkn_sell=tkn_sell,
    #                 sell_quantity=sell_tkns[tkn_sell]
    #             )
    #             optimized_swaps.append({
    #                 'exchange': ex_name,
    #                 'trade': 'sell',
    #                 'buy_asset': stablecoin,
    #                 'sell_asset': tkn_sell,
    #                 'amount': sell_tkns[tkn_sell]
    #             })
    #     for tkn_buy in buy_tkns:
    #         if tkn_buy == stablecoin:
    #             continue
    #         if buy_tkns[tkn_buy] > 0:
    #             test_ex.fail = ''
    #             test_ex.swap(
    #                 agent=test_agent,
    #                 tkn_buy=tkn_buy,
    #                 tkn_sell=stablecoin,
    #                 buy_quantity=buy_tkns[tkn_buy]
    #             )
    #             if not test_ex.fail:
    #                 optimized_swaps.append({
    #                     'exchange': ex_name,
    #                     'trade': 'buy',
    #                     'buy_asset': tkn_buy,
    #                     'sell_asset': stablecoin,
    #                     'amount': buy_tkns[tkn_buy]
    #                 })
    #     return_swaps += optimized_swaps
    # return return_swaps
        #             else:
        #                 for intermediate_tkn in ex.asset_list:
        #                     if ex.buy_spot(tkn_buy, intermediate_tkn) and ex.buy_spot(intermediate_tkn, stablecoin):
        #                         buy_quantity = test_ex.calculate_sell_from_buy(
        #                             tkn_buy=tkn_buy,
        #                             tkn_sell=intermediate_tkn,
        #                             buy_quantity=buy_tkns[tkn_buy]
        #                         )
        #                         test_ex.swap(
        #                             agent=test_agent,
        #                             tkn_buy=intermediate_tkn,
        #                             tkn_sell=stablecoin,
        #                             buy_quantity=ex.calculate_sell_from_buy(
        #                                 tkn_buy=tkn_buy,
        #                                 tkn_sell=intermediate_tkn,
        #                                 buy_quantity=buy_quantity
        #                             )
        #                         )
        #                         optimized_swaps.append({
        #                             'exchange': ex_name,
        #                             'trade': 'buy',
        #                             'buy_asset': intermediate_tkn,
        #                             'sell_asset': stablecoin,
        #                             'amount': ex.calculate_sell_from_buy(
        #                                 tkn_buy=tkn_buy,
        #                                 tkn_sell=intermediate_tkn,
        #                                 buy_quantity=buy_quantity
        #                             )
        #                         })
        #                         test_ex.swap(
        #                             agent=test_agent,
        #                             tkn_buy=tkn_buy,
        #                             tkn_sell=intermediate_tkn,
        #                             buy_quantity=buy_tkns[tkn_buy]
        #                         )
        #                         optimized_swaps.append({
        #                             'exchange': ex_name,
        #                             'trade': 'buy',
        #                             'buy_asset': tkn_buy,
        #                             'sell_asset': intermediate_tkn,
        #                             'amount': buy_tkns[tkn_buy]
        #                         })
        #                         break
        #
        # optimized_profit = calculate_profit(agent, test_agent, asset_map=asset_map)
        # optimized_profit_usd = ref_exchange.value_assets(assets=optimized_profit, equivalency_map=asset_map)
        # if optimized_profit_usd < default_profit_usd:
        #     return_swaps += default_swaps
        # else:
        #     return_swaps += optimized_swaps

    # make absolutely sure nothing went below 0
    # test_agent = agent.copy()
    # test_exchanges = {ex_name: ex.copy() for ex_name, ex in exchanges.items()}
    # execute_arb(exchanges, test_agent, return_swaps)
    # test_profit = calculate_profit(agent, test_agent, asset_map=asset_map)
    # if min(test_profit.values()) < 0:
    #     return all_swaps
    # return return_swaps
