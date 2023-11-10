import copy

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.centralized_market import CentralizedMarket
from hydradx.model.amm.arbitrage_agent import calculate_arb_amount_bid, calculate_arb_amount_ask


def get_arb_opps(op_state, order_book, order_book_map, cex_fee, buffer):
    arb_opps = []

    for tkn_pair in order_book_map:
        ob_tkn_pair = order_book_map[tkn_pair]
        pair_order_book = order_book[ob_tkn_pair]

        dex_spot_price = OmnipoolState.price(op_state, tkn_pair[0], tkn_pair[1])

        if len(pair_order_book.bids) > 0:
            bid_price = pair_order_book.bids[0][0]
            cex_sell_price = bid_price * (1 - cex_fee - buffer)

            numeraire_lrna_fee = op_state.lrna_fee[tkn_pair[1]].compute(tkn=tkn_pair[1])
            tkn_asset_fee = op_state.asset_fee[tkn_pair[0]].compute(tkn=tkn_pair[0])
            dex_buy_price = dex_spot_price / ((1 - tkn_asset_fee) * (1 - numeraire_lrna_fee))

            if dex_buy_price < cex_sell_price:  # buy from DEX, sell to CEX
                arb_opps.append(((cex_sell_price - dex_buy_price) / dex_buy_price, tkn_pair))

        if len(pair_order_book.asks) > 0:
            ask_price = pair_order_book.asks[0][0]
            cex_buy_price = ask_price * (1 + cex_fee + buffer)

            numeraire_asset_fee = op_state.asset_fee[tkn_pair[1]].compute(tkn=tkn_pair[1])
            tkn_lrna_fee = op_state.lrna_fee[tkn_pair[0]].compute(tkn=tkn_pair[0])
            dex_sell_price = dex_spot_price * (1 - numeraire_asset_fee) * (1 - tkn_lrna_fee)

            if dex_sell_price > cex_buy_price:  # buy from CEX, sell to DEX
                arb_opps.append(((dex_sell_price - cex_buy_price) / cex_buy_price, tkn_pair))

    arb_opps.sort(key=lambda x: x[0], reverse=True)
    return arb_opps


def get_arb_swaps(op_state, cex, order_book_map, buffer=0.0, max_liquidity={'cex': {}, 'dex': {}}, iters=20) -> list:
    cex_fee = cex.trade_fee
    dex_slippage_tolerance = buffer / 2
    cex_slippage_tolerance = buffer / 2

    arb_opps = get_arb_opps(op_state, cex.order_book, order_book_map, cex_fee, buffer)

    init_amt = 1000000000
    all_swaps = []
    state = op_state.copy()
    test_cex = cex.copy()
    holdings = {asset: init_amt for asset in test_cex.asset_list + state.asset_list}
    test_agent = Agent(holdings=holdings, unique_id='bot')
    order_book = test_cex.order_book
    while arb_opps:
        tkn_pair = arb_opps[0][1]
        ob_tkn_pair = order_book_map[tkn_pair]
        bids, asks = order_book[ob_tkn_pair].bids, order_book[ob_tkn_pair].asks
        tkn = tkn_pair[0]
        numeraire = tkn_pair[1]

        tkn_lrna_fee = state.lrna_fee[tkn].compute(tkn=tkn)
        numeraire_lrna_fee = state.lrna_fee[numeraire].compute(tkn=numeraire)
        tkn_asset_fee = state.asset_fee[tkn].compute(tkn=tkn)
        numeraire_asset_fee = state.asset_fee[numeraire].compute(tkn=numeraire)

        op_spot = OmnipoolState.price(state, tkn, numeraire)
        buy_spot = op_spot / ((1 - numeraire_lrna_fee) * (1 - tkn_asset_fee))
        sell_spot = op_spot * (1 - tkn_lrna_fee) * (1 - numeraire_asset_fee)

        if buy_spot < bids[0][0] * (1 - cex_fee - buffer):  # tkn is coming out of pool, numeraire going into pool
            bid = bids[0]
            max_liq_tkn = max_liquidity['cex'][tkn] if tkn in max_liquidity['cex'] else float('inf')
            max_liq_num = max_liquidity['dex'][numeraire] if numeraire in max_liquidity['dex'] else float('inf')
            amt = calculate_arb_amount_bid(state, tkn, numeraire, bid, cex_fee, buffer, max_liq_tkn=max_liq_tkn,
                                           max_liq_num=max_liq_num, precision=1e-10, max_iters=iters)

            if amt != 0:

                state.swap(test_agent, tkn_buy=tkn, tkn_sell=numeraire, buy_quantity=amt)
                amt_in = init_amt - test_agent.holdings[numeraire]
                test_cex.swap(test_agent, tkn_sell=ob_tkn_pair[0], tkn_buy=ob_tkn_pair[1], sell_quantity=amt)
                all_swaps += [{
                        'exchange': 'dex',
                        'trade': 'buy',
                        'buy_asset': tkn,
                        'sell_asset': numeraire,
                        'price': bid[0],
                        'amount': amt,
                        'max_sell': amt_in * (1 + dex_slippage_tolerance)
                    }, {
                        'exchange': 'cex',
                        'trade': 'sell',
                        'buy_asset': ob_tkn_pair[1],
                        'sell_asset': ob_tkn_pair[0],
                        'price': bid[0] * (1 - cex_slippage_tolerance),
                        'amount': amt
                    }]
                if tkn in max_liquidity['cex']:
                    max_liquidity['cex'][tkn] -= amt
                if numeraire in max_liquidity['cex']:
                    max_liquidity['cex'][numeraire] += amt_in
                if tkn in max_liquidity['dex']:
                    max_liquidity['dex'][tkn] += amt
                if numeraire in max_liquidity['dex']:
                    max_liquidity['dex'][numeraire] -= amt_in

        elif sell_spot > asks[0][0] * (1 + cex_fee + buffer):
            ask = asks[0]
            max_liq_tkn = max_liquidity['dex'][tkn] if tkn in max_liquidity['dex'] else float('inf')
            max_liq_num = max_liquidity['cex'][numeraire] if numeraire in max_liquidity['cex'] else float('inf')
            amt = calculate_arb_amount_ask(state, tkn, numeraire, ask, cex_fee, buffer, max_liq_tkn=max_liq_tkn,
                                           max_liq_num=max_liq_num, precision=1e-10, max_iters=iters)
            if amt != 0:
                state.swap(test_agent, tkn_buy=numeraire, tkn_sell=tkn, sell_quantity=amt)
                amt_out = test_agent.holdings[numeraire] - init_amt
                test_cex.swap(test_agent, tkn_sell=ob_tkn_pair[1], tkn_buy=ob_tkn_pair[0], buy_quantity=amt)

                all_swaps += [{
                        'exchange': 'dex',
                        'trade': 'sell',
                        'buy_asset': numeraire,
                        'sell_asset': tkn,
                        'price': ask[0],
                        'amount': amt,
                        'min_buy': amt_out * (1 - dex_slippage_tolerance)
                    }, {
                        'exchange': 'cex',
                        'trade': 'buy',
                        'buy_asset': ob_tkn_pair[0],
                        'sell_asset': ob_tkn_pair[1],
                        'price': ask[0] * (1 + cex_slippage_tolerance),
                        'amount': amt
                    }]
                if tkn in max_liquidity['cex']:
                    max_liquidity['cex'][tkn] += amt
                if numeraire in max_liquidity['cex']:
                    max_liquidity['cex'][numeraire] -= amt_out
                if tkn in max_liquidity['dex']:
                    max_liquidity['dex'][tkn] -= amt
                if numeraire in max_liquidity['dex']:
                    max_liquidity['dex'][numeraire] += amt_out
        new_arb_opps = get_arb_opps(state, order_book, order_book_map, cex_fee, buffer)
        if arb_opps and new_arb_opps and arb_opps[0][0] == new_arb_opps[0][0]:
            break
        arb_opps = new_arb_opps

    return all_swaps


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
                if tkn_buy == 'BNC':
                    er = 1
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
                        if test_agent.holdings[tkn_buy] - previous_tkn_buy != max_buy:
                            er = 1
                        else:
                            er = 2
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
            test_ex = ex.copy()
            test_agent = agent.copy()
            for swap in optimized_swaps:
                tkn_sell = swap['sell_asset']
                tkn_buy = swap['buy_asset']
                if swap['trade'] == 'buy':
                    test_ex.swap(test_agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=swap['amount'])
                else:
                    test_ex.swap(test_agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=swap['amount'])
            optimized_net_swaps = {tkn: test_agent.holdings[tkn] - agent.holdings[tkn] for tkn in ex.asset_list}
            test_ex = ex.copy()
            test_agent = agent.copy()
            for swap in default_swaps:
                tkn_sell = swap['sell_asset']
                tkn_buy = swap['buy_asset']
                if swap['trade'] == 'buy':
                    test_ex.swap(test_agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=swap['amount'])
                else:
                    test_ex.swap(test_agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=swap['amount'])
            default_net_swaps = {tkn: test_agent.holdings[tkn] - agent.holdings[tkn] for tkn in ex.asset_list}
            for tkn in ex.asset_list:
                if optimized_net_swaps[tkn] < default_net_swaps[tkn]:
                    er = 1
    return return_swaps


def execute_arb(dex, cex, agent, all_swaps):
    exchanges = {
        'dex': dex,
        'cex': cex
    }
    for swap in all_swaps:
        tkn_buy = swap['buy_asset']
        tkn_sell = swap['sell_asset']
        ex = exchanges[swap['exchange']]
        ex.fail = ''
        if swap['trade'] == 'buy':
            ex.swap(agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=swap['amount'])
            if ex.fail:
                er = 1
        elif swap['trade'] == 'sell':
            # omnipool leg
            ex.swap(agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=swap['amount'])
            if ex.fail:
                er = 1
        else:
            raise ValueError('Incorrect trade type.')


def calculate_profit(init_agent, agent, asset_map=None):
    if asset_map is None:
        asset_map = {}
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
