from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.centralized_market import CentralizedMarket


# note that this function mutates state, test_agent, test_cex, and max_liquidity
def process_next_swap(state, test_agent, test_cex, tkn_pair, ob_tkn_pair, buffer, max_liquidity, iters):
    bids, asks = test_cex.order_book[ob_tkn_pair].bids, test_cex.order_book[ob_tkn_pair].asks
    tkn = tkn_pair[0]
    numeraire = tkn_pair[1]
    cex_fee = test_cex.trade_fee
    dex_slippage_tolerance = buffer/2
    cex_slippage_tolerance = buffer/2

    tkn_lrna_fee = state.lrna_fee[tkn].compute(tkn=tkn)
    numeraire_lrna_fee = state.lrna_fee[numeraire].compute(tkn=numeraire)
    tkn_asset_fee = state.asset_fee[tkn].compute(tkn=tkn)
    numeraire_asset_fee = state.asset_fee[numeraire].compute(tkn=numeraire)

    op_spot = OmnipoolState.price(state, tkn, numeraire)
    buy_spot = op_spot / ((1 - numeraire_lrna_fee) * (1 - tkn_asset_fee))
    sell_spot = op_spot * (1 - tkn_lrna_fee) * (1 - numeraire_asset_fee)
    swap = {}

    if bids and buy_spot < bids[0][0] * (1 - cex_fee - buffer):  # tkn is coming out of pool, numeraire going into pool
        bid = bids[0]
        max_liq_tkn = max_liquidity['cex'][tkn] if tkn in max_liquidity['cex'] else float('inf')
        max_liq_num = max_liquidity['dex'][numeraire] if numeraire in max_liquidity['dex'] else float('inf')
        amt = calculate_arb_amount_bid(state, tkn, numeraire, bid, cex_fee, buffer, min_amt=1e-6,
                                       max_liq_tkn=max_liq_tkn, max_liq_num=max_liq_num, precision=1e-10,
                                       max_iters=iters)

        if amt != 0:
            init_amt = test_agent.holdings[numeraire]
            state.swap(test_agent, tkn_buy=tkn, tkn_sell=numeraire, buy_quantity=amt)
            amt_in = init_amt - test_agent.holdings[numeraire]
            test_cex.swap(test_agent, tkn_sell=ob_tkn_pair[0], tkn_buy=ob_tkn_pair[1], sell_quantity=amt)
            swap = {
                'dex': {
                    'trade': 'buy',
                    'buy_asset': tkn,
                    'sell_asset': numeraire,
                    'price': bid[0],
                    'amount': amt,
                    'max_sell': amt_in * (1 + dex_slippage_tolerance)
                },
                'cex': {
                    'trade': 'sell',
                    'buy_asset': ob_tkn_pair[1],
                    'sell_asset': ob_tkn_pair[0],
                    'price': bid[0] * (1 - cex_slippage_tolerance),
                    'amount': amt
                }
            }
            if tkn in max_liquidity['cex']:
                max_liquidity['cex'][tkn] -= amt
            if numeraire in max_liquidity['cex']:
                max_liquidity['cex'][numeraire] += amt_in
            if tkn in max_liquidity['dex']:
                max_liquidity['dex'][tkn] += amt
            if numeraire in max_liquidity['dex']:
                max_liquidity['dex'][numeraire] -= amt_in

    elif asks and sell_spot > asks[0][0] * (1 + cex_fee + buffer):
        ask = asks[0]
        max_liq_tkn = max_liquidity['dex'][tkn] if tkn in max_liquidity['dex'] else float('inf')
        max_liq_num = max_liquidity['cex'][numeraire] if numeraire in max_liquidity['cex'] else float('inf')
        amt = calculate_arb_amount_ask(state, tkn, numeraire, ask, cex_fee, buffer, min_amt=1e-6,
                                       max_liq_tkn=max_liq_tkn, max_liq_num=max_liq_num, precision=1e-10,
                                       max_iters=iters)
        if amt != 0:
            init_amt = test_agent.holdings[numeraire]
            state.swap(test_agent, tkn_buy=numeraire, tkn_sell=tkn, sell_quantity=amt)
            amt_out = test_agent.holdings[numeraire] - init_amt
            test_cex.swap(test_agent, tkn_sell=ob_tkn_pair[1], tkn_buy=ob_tkn_pair[0], buy_quantity=amt)
            swap = {
                'dex': {
                    'trade': 'sell',
                    'buy_asset': numeraire,
                    'sell_asset': tkn,
                    'price': ask[0],
                    'amount': amt,
                    'min_buy': amt_out * (1 - dex_slippage_tolerance)},
                'cex': {
                    'trade': 'buy',
                    'buy_asset': ob_tkn_pair[0],
                    'sell_asset': ob_tkn_pair[1],
                    'price': ask[0] * (1 + cex_slippage_tolerance),
                    'amount': amt
                }
            }
            if tkn in max_liquidity['cex']:
                max_liquidity['cex'][tkn] += amt
            if numeraire in max_liquidity['cex']:
                max_liquidity['cex'][numeraire] -= amt_out
            if tkn in max_liquidity['dex']:
                max_liquidity['dex'][tkn] -= amt
            if numeraire in max_liquidity['dex']:
                max_liquidity['dex'][numeraire] += amt_out

    return swap


def get_arb_opps(op_state, cex_dict, config):
    arb_opps = []

    # for arb_pair in buffer:
    for i, arb_cfg in enumerate(config):
        tkn_pair = arb_cfg['tkn_pair']
        ob_tkn_pair = arb_cfg['order_book']
        exchange = arb_cfg['exchange']
        pair_order_book = cex_dict[exchange].order_book[ob_tkn_pair]
        cex_fee = cex_dict[exchange].trade_fee
        buffer = arb_cfg['buffer']

        dex_spot_price = OmnipoolState.price(op_state, tkn_pair[0], tkn_pair[1])

        if len(pair_order_book.bids) > 0:
            bid_price = pair_order_book.bids[0][0]
            cex_sell_price = bid_price * (1 - cex_fee - buffer)

            numeraire_lrna_fee = op_state.lrna_fee[tkn_pair[1]].compute(tkn=tkn_pair[1])
            tkn_asset_fee = op_state.asset_fee[tkn_pair[0]].compute(tkn=tkn_pair[0])
            dex_buy_price = dex_spot_price / ((1 - tkn_asset_fee) * (1 - numeraire_lrna_fee))

            if dex_buy_price < cex_sell_price:  # buy from DEX, sell to CEX
                arb_opps.append(((cex_sell_price - dex_buy_price) / dex_buy_price, i))

        if len(pair_order_book.asks) > 0:
            ask_price = pair_order_book.asks[0][0]
            cex_buy_price = ask_price * (1 + cex_fee + buffer)

            numeraire_asset_fee = op_state.asset_fee[tkn_pair[1]].compute(tkn=tkn_pair[1])
            tkn_lrna_fee = op_state.lrna_fee[tkn_pair[0]].compute(tkn=tkn_pair[0])
            dex_sell_price = dex_spot_price * (1 - numeraire_asset_fee) * (1 - tkn_lrna_fee)

            if dex_sell_price > cex_buy_price:  # buy from CEX, sell to DEX
                arb_opps.append(((dex_sell_price - cex_buy_price) / cex_buy_price, i))

    arb_opps.sort(key=lambda x: x[0], reverse=True)
    return arb_opps


def flatten_swaps(swaps):
    return [
        {'exchange': 'omnipool', **trade['dex']}
        if key == 'dex' else
        {'exchange': trade['exchange'], **trade['cex']}
        for trade in swaps for key in trade if key in ['dex', 'cex']
    ]


def get_arb_swaps(op_state, cex_dict, config, max_liquidity=None, iters=20):
    arb_opps = get_arb_opps(op_state, cex_dict, config)

    if max_liquidity is None:
        max_liquidity = {k: {'cex': {}, 'dex': {}} for k in cex_dict}

    init_amt = 1000000000
    all_swaps = []
    state = op_state.copy()
    test_cex_dict = {exchange: cex_dict[exchange].copy() for exchange in cex_dict}
    holdings = {asset: init_amt for asset in state.asset_list}
    for ex in test_cex_dict:
        for asset in test_cex_dict[ex].asset_list:
            if asset not in holdings:
                holdings[asset] = init_amt
    test_agent = Agent(holdings=holdings, unique_id='bot')
    while arb_opps:
        arb_cfg = config[arb_opps[0][1]]
        swap = process_next_swap(state,
                                 test_agent,
                                 test_cex_dict[arb_cfg['exchange']],
                                 arb_cfg['tkn_pair'],
                                 arb_cfg['order_book'],
                                 arb_cfg['buffer'],
                                 max_liquidity[arb_cfg['exchange']],
                                 iters)
        if swap:
            swap['exchange'] = arb_cfg['exchange']
            all_swaps.append(swap)
        else:
            break
        new_arb_opps = get_arb_opps(state, test_cex_dict, config)
        if arb_opps and new_arb_opps and arb_opps[0][0] == new_arb_opps[0][0]:
            break
        arb_opps = new_arb_opps

    return flatten_swaps(all_swaps)


def get_arb_swaps_simple(op_state, cex_dict, config, max_liquidity=None, iters=20):
    if max_liquidity is None:
        max_liquidity = {k: {'cex': {}, 'dex': {}} for k in cex_dict}

    init_amt = 1000000000
    all_swaps = []
    state = op_state.copy()
    test_cex_dict = {exchange: cex_dict[exchange].copy() for exchange in cex_dict}
    holdings = {asset: init_amt for asset in state.asset_list}
    for ex in test_cex_dict:
        for asset in test_cex_dict[ex].asset_list:
            if asset not in holdings:
                holdings[asset] = init_amt
    test_agent = Agent(holdings=holdings, unique_id='bot')
    for arb_cfg in config:
        swap = None
        while swap != {}:
            swap = process_next_swap(state,
                                     test_agent,
                                     test_cex_dict[arb_cfg['exchange']],
                                     arb_cfg['tkn_pair'],
                                     arb_cfg['order_book'],
                                     arb_cfg['buffer'],
                                     max_liquidity[arb_cfg['exchange']],
                                     iters)
            if swap:
                swap['exchange'] = arb_cfg['exchange']
                all_swaps.append(swap)

    return flatten_swaps(all_swaps)


def calculate_arb_amount_bid(
        init_state: OmnipoolState,
        tkn, numeraire,
        bid: list,
        cex_fee: float = 0.0,
        buffer: float = 0.0,
        min_amt=1e-18,
        max_liq_tkn=float('inf'),
        max_liq_num=float('inf'),
        precision=1e-15,
        max_iters=None
):
    state = init_state.copy()
    init_amt = 1000000000
    holdings = {asset: init_amt for asset in [tkn, numeraire]}
    agent = Agent(holdings=holdings, unique_id='bot')

    asset_fee = state.asset_fee[tkn].compute(tkn=tkn)
    lrna_fee = state.lrna_fee[numeraire].compute(tkn=numeraire)
    cex_price = bid[0] * (1 - cex_fee - buffer)

    # If buying the min amount moves the price too much, return 0
    if min_amt < 1e-18:
        raise
    test_agent = agent.copy()
    test_state = state.copy()
    test_state.swap(test_agent, tkn_buy=tkn, tkn_sell=numeraire, buy_quantity=min_amt)
    op_spot = OmnipoolState.price(test_state, tkn, numeraire)
    buy_spot = op_spot / ((1 - lrna_fee) * (1 - asset_fee))
    if min_amt >= state.liquidity[tkn] or buy_spot > cex_price or test_state.fail != '':
        return 0
    if agent.holdings[numeraire] - test_agent.holdings[numeraire] > max_liq_num:
        return 0

    op_spot = OmnipoolState.price(state, tkn, numeraire)
    buy_spot = op_spot / ((1 - lrna_fee) * (1 - asset_fee))

    # we use binary search to find the amount that can be swapped
    amt_low = min_amt
    amt_high = min(max_liq_tkn, bid[1])
    amt = min(amt_high, state.liquidity[tkn])
    i = 0
    best_buy_spot = buy_spot
    while cex_price - best_buy_spot > precision:
        if amt < state.liquidity[tkn]:
            test_agent = agent.copy()
            test_state = state.copy()
            test_state.swap(test_agent, tkn_buy=tkn, tkn_sell=numeraire, buy_quantity=amt)
            op_spot = OmnipoolState.price(test_state, tkn, numeraire)
            buy_spot = op_spot / ((1 - lrna_fee) * (1 - asset_fee))
        if amt >= state.liquidity[tkn] or buy_spot > cex_price or test_state.fail != '':
            amt_high = amt
        elif buy_spot < cex_price:
            if agent.holdings[numeraire] - test_agent.holdings[numeraire] > max_liq_num:
                # best trade will involve trading max_liquidity, so can be calculated as a sell
                test_agent = agent.copy()
                test_state = state.copy()
                test_state.swap(test_agent, tkn_buy=tkn, tkn_sell=numeraire, sell_quantity=max_liq_num)
                return test_agent.holdings[tkn] - agent.holdings[tkn]
            else:
                amt_low = amt
                best_buy_spot = buy_spot

        if amt_high == amt_low:  # full amount can be traded
            break

        # only want to update amt if there will be another iteration
        if cex_price - best_buy_spot > precision:
            amt = amt_low + (amt_high - amt_low) / 2

        i += 1
        if max_iters is not None and i >= max_iters:
            break

    if amt_low == min_amt:
        return 0
    else:
        return amt_low


def calculate_arb_amount_ask(
        init_state: OmnipoolState,
        tkn,
        numeraire,
        ask: list,
        cex_fee: float = 0.0,
        buffer: float = 0.0,
        min_amt=1e-18,
        max_liq_tkn=float('inf'),
        max_liq_num=float('inf'),
        precision=1e-15,
        max_iters=None
):
    state = init_state.copy()
    init_amt = 1000000000
    holdings = {asset: init_amt for asset in [tkn, numeraire]}
    agent = Agent(holdings=holdings, unique_id='bot')

    asset_fee = state.asset_fee[numeraire].compute(tkn=numeraire)
    lrna_fee = state.lrna_fee[tkn].compute(tkn=tkn)
    cex_price = ask[0] * (1 + cex_fee + buffer)

    # If buying the min amount moves the price too much, return 0
    if min_amt < 1e-18:
        raise
    test_agent = agent.copy()
    test_state = state.copy()
    test_state.swap(test_agent, tkn_buy=numeraire, tkn_sell=tkn, sell_quantity=min_amt)
    op_spot = OmnipoolState.price(test_state, tkn, numeraire)
    sell_spot = op_spot * (1 - lrna_fee) * (1 - asset_fee)
    if min_amt >= state.liquidity[tkn] or sell_spot < cex_price or test_state.fail != '':
        return 0

    op_spot = OmnipoolState.price(state, tkn, numeraire)
    sell_spot = op_spot * (1 - lrna_fee) * (1 - asset_fee)

    # we use binary search to find the amount that can be swapped
    amt_low = min_amt
    amt_high = min(ask[1], max_liq_tkn)
    amt = amt_high
    i = 0
    best_sell_spot = sell_spot
    while best_sell_spot - cex_price > precision:
        test_agent = agent.copy()
        test_state = state.copy()
        test_state.swap(test_agent, tkn_buy=numeraire, tkn_sell=tkn, sell_quantity=amt)
        op_spot = OmnipoolState.price(test_state, tkn, numeraire)
        sell_spot = op_spot * (1 - lrna_fee) * (1 - asset_fee)
        if sell_spot < cex_price or test_state.fail != '':
            amt_high = amt
        elif sell_spot > cex_price:
            if test_agent.holdings[numeraire] - agent.holdings[numeraire] > max_liq_num:
                # best trade will involve trading max_liquidity, so can be calculated as a sell
                test_agent = agent.copy()
                test_state = state.copy()
                test_state.swap(test_agent, tkn_buy=numeraire, tkn_sell=tkn, buy_quantity=max_liq_num)
                return agent.holdings[tkn] - test_agent.holdings[tkn]
            else:
                amt_low = amt
                best_sell_spot = sell_spot

        if amt_high == amt_low:  # full amount can be traded
            break

        # only want to update amt if there will be another iteration
        if best_sell_spot - cex_price > precision:
            amt = amt_low + (amt_high - amt_low) / 2

        i += 1
        if max_iters is not None and i >= max_iters:
            break
    if amt_low == min_amt:
        return 0
    else:
        return amt_low


def execute_arb(dex, cex_dict, agent, all_swaps):
    exchanges = {
        'omnipool': dex,
        'kraken': cex_dict['kraken'],
        'binance': cex_dict['binance']
    }
    for swap in all_swaps:
        tkn_buy = swap['buy_asset']
        tkn_sell = swap['sell_asset']
        ex = exchanges[swap['exchange']]
        ex.fail = ''
        if swap['trade'] == 'buy':
            ex.swap(agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=swap['amount'])
        elif swap['trade'] == 'sell':
            # omnipool leg
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

    if sum([profit_asset[k] for k in profit_asset]) != sum([profit[k] for k in profit]):
        raise
    return profit


def combine_swaps(
        dex: OmnipoolState,
        cex: CentralizedMarket,
        agent: Agent,
        all_swaps: list[dict],
        asset_map: dict[str, str],
):
    # take the list of swaps and try to get the same result more efficiently
    # in particular, make sure to buy *at least* as much of each asset as the net from the original list
    exchanges = {'cex': cex, 'dex': dex}
    net_swaps = {'dex': {}, 'cex': {}}
    return_swaps = []

    for ex_name, ex in exchanges.items():

        test_agent = agent.copy()
        test_ex = ex.copy()
        default_swaps = list(filter(lambda s: s['exchange'] == ex_name, all_swaps))

        for swap in default_swaps:
            tkn_sell = swap['sell_asset']
            tkn_buy = swap['buy_asset']
            if swap['trade'] == 'buy':
                test_ex.swap(test_agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=swap['amount'])
            else:
                test_ex.swap(test_agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=swap['amount'])
        net_swaps[ex_name] = {tkn: test_agent.holdings[tkn] - agent.holdings[tkn] for tkn in ex.asset_list}
        # actual_swaps = {tkn: 0 for tkn in ex.asset_list}
        default_profit = calculate_profit(agent, test_agent, asset_map=asset_map)
        default_profit_usd = cex.value_assets(default_profit, asset_map)

        test_ex = ex.copy()
        test_agent = agent.copy()
        optimized_swaps = []

        buy_tkns = {tkn: 0 for tkn in ex.asset_list}
        buy_tkns.update({
            tkn: quantity for tkn, quantity in
            sorted(filter(lambda x: x[1] > 0, net_swaps[ex_name].items()), key=lambda x: x[1] * ex.buy_spot(x[0]))
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
                            x: test_ex.sell_spot(x, numeraire=tkn_buy)
                            or (
                                   test_ex.buy_spot(tkn_buy, numeraire=x)
                                   if test_ex.buy_spot(tkn_buy, numeraire=x) > 0 else 0
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
                        sell_quantity=sell_quantity
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
                        sell_tkns[tkn_sell] -= previous_tkn_sell - test_agent.holdings[tkn_sell]
                    else:
                        # buy enough to satisfy buy_quantity
                        test_ex.swap(test_agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=buy_tkns[tkn_buy])
                        sell_tkns[tkn_sell] -= previous_tkn_sell - test_agent.holdings[tkn_sell]
                        optimized_swaps.append({
                            'exchange': ex_name,
                            'trade': 'buy',
                            'buy_asset': tkn_buy,
                            'sell_asset': tkn_sell,
                            'amount': buy_tkns[tkn_buy]
                        })
                        buy_tkns[tkn_buy] = 0
                        break

        if sum(buy_tkns.values()) > 0:
            # try and sell everything remaining for USD, then use that to buy the remaining tokens
            for tkn_sell in sell_tkns:
                if sell_tkns[tkn_sell] > 0:
                    test_ex.swap(
                        agent=test_agent,
                        tkn_buy=ex.stablecoin,
                        tkn_sell=tkn_sell,
                        sell_quantity=sell_tkns[tkn_sell]
                    )
                    optimized_swaps.append({
                        'exchange': ex_name,
                        'trade': 'sell',
                        'buy_asset': ex.stablecoin,
                        'sell_asset': tkn_sell,
                        'amount': sell_tkns[tkn_sell]
                    })
            for tkn_buy in buy_tkns:
                if buy_tkns[tkn_buy] > 0:
                    test_ex.fail = ''
                    test_ex.swap(
                        agent=test_agent,
                        tkn_buy=tkn_buy,
                        tkn_sell=ex.stablecoin,
                        buy_quantity=buy_tkns[tkn_buy]
                    )
                    if not test_ex.fail:
                        optimized_swaps.append({
                            'exchange': ex_name,
                            'trade': 'buy',
                            'buy_asset': tkn_buy,
                            'sell_asset': ex.stablecoin,
                            'amount': buy_tkns[tkn_buy]
                        })
                    else:
                        for intermediate_tkn in ex.asset_list:
                            if ex.buy_spot(tkn_buy, intermediate_tkn) and ex.buy_spot(intermediate_tkn, ex.stablecoin):
                                buy_quantity = test_ex.calculate_sell_from_buy(
                                    tkn_buy=tkn_buy,
                                    tkn_sell=intermediate_tkn,
                                    buy_quantity=buy_tkns[tkn_buy]
                                )
                                test_ex.swap(
                                    agent=test_agent,
                                    tkn_buy=intermediate_tkn,
                                    tkn_sell=ex.stablecoin,
                                    buy_quantity=ex.calculate_sell_from_buy(
                                        tkn_buy=tkn_buy,
                                        tkn_sell=intermediate_tkn,
                                        buy_quantity=buy_quantity
                                    )
                                )
                                optimized_swaps.append({
                                    'exchange': ex_name,
                                    'trade': 'buy',
                                    'buy_asset': intermediate_tkn,
                                    'sell_asset': ex.stablecoin,
                                    'amount': ex.calculate_sell_from_buy(
                                        tkn_buy=tkn_buy,
                                        tkn_sell=intermediate_tkn,
                                        buy_quantity=buy_quantity
                                    )
                                })
                                test_ex.swap(
                                    agent=test_agent,
                                    tkn_buy=tkn_buy,
                                    tkn_sell=intermediate_tkn,
                                    buy_quantity=buy_tkns[tkn_buy]
                                )
                                optimized_swaps.append({
                                    'exchange': ex_name,
                                    'trade': 'buy',
                                    'buy_asset': tkn_buy,
                                    'sell_asset': intermediate_tkn,
                                    'amount': buy_tkns[tkn_buy]
                                })
                                break

        optimized_profit = calculate_profit(agent, test_agent, asset_map=asset_map)
        optimized_profit_usd = cex.value_assets(optimized_profit, asset_map)
        if optimized_profit_usd < default_profit_usd:
            return_swaps += default_swaps
        else:
            return_swaps += optimized_swaps

    return return_swaps
