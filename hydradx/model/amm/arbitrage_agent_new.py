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


def get_arb_swaps(op_state, cex, order_book_map, buffer=0.0, max_liquidity={'cex': {}, 'dex': {}}, iters=20):
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
                op_spot = OmnipoolState.price(state, tkn, numeraire)
                all_swaps.append({'dex': {'trade': 'buy',
                                      'buy_asset': tkn,
                                      'sell_asset': numeraire,
                                      'price': bid[0],
                                      'amount': amt,
                                      'max_sell': amt_in * (1 + dex_slippage_tolerance)
                                      },
                              'cex': {'trade': 'sell',
                                      'buy_asset': ob_tkn_pair[1],
                                      'sell_asset': ob_tkn_pair[0],
                                      'price': bid[0] * (1 - cex_slippage_tolerance),
                                      'amount': amt
                                      }})
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
                op_spot = OmnipoolState.price(state, tkn, numeraire)

                all_swaps.append({'dex': {'trade': 'sell',
                                      'buy_asset': numeraire,
                                      'sell_asset': tkn,
                                      'price': ask[0],
                                      'amount': amt,
                                      'min_buy': amt_out * (1 - dex_slippage_tolerance)},
                              'cex': {'trade': 'buy',
                                      'buy_asset': ob_tkn_pair[0],
                                      'sell_asset': ob_tkn_pair[1],
                                      'price': ask[0] * (1 + cex_slippage_tolerance),
                                      'amount': amt
                                      }})
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


def combine_execute(
        dex: OmnipoolState,
        cex: CentralizedMarket,
        agent: Agent,
        all_swaps: list[dict],
):
    # take the list of swaps and try to get the same result more efficiently
    # in particular, make sure to buy *at least* as much of each asset as the net from the original list
    exchanges = (cex, 'cex'), (dex, 'dex')
    net_swaps = {'dex': {}, 'cex': {}}

    for ex, ex_name in exchanges:

        test_agent = agent.copy()
        test_ex = ex.copy()

        for swap in all_swaps:
            tkn_sell = swap[ex_name]['sell_asset']
            tkn_buy = swap[ex_name]['buy_asset']
            if swap[ex_name]['trade'] == 'buy':
                test_ex.swap(test_agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=swap[ex_name]['amount'])
            else:
                test_ex.swap(test_agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=swap[ex_name]['amount'])
        net_swaps[ex_name] = {tkn: test_agent.holdings[tkn] - agent.holdings[tkn] for tkn in ex.asset_list}
        # actual_swaps = {tkn: 0 for tkn in ex.asset_list}

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
                            x: ex.sell_spot(x, numeraire=tkn_buy)
                            or (ex.buy_spot(tkn_buy, numeraire=x) if ex.buy_spot(tkn_buy, numeraire=x) > 0 else 0)
                            for x in sell_tkns
                        }.items()
                    )
                }

                for tkn_sell in best_sell_tkns:
                    sell_quantity, price = best_sell_tkns[tkn_sell]
                    max_buy = ex.calculate_buy_from_sell(
                        tkn_sell=tkn_sell,
                        tkn_buy=tkn_buy,
                        sell_quantity=sell_quantity
                    )
                    if max_buy == 0:
                        continue
                    if max_buy <= buy_tkns[tkn_buy]:
                        # buy as much as we can without going over sell_quantity
                        previous_tkn_sell = agent.holdings[tkn_sell]
                        ex.swap(agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=max_buy)
                        buy_tkns[tkn_buy] -= max_buy
                        # actual_swaps[tkn_buy] -= max_buy
                        # actual_swaps[tkn_sell] -= agent.holdings[tkn_sell] - previous_tkn_sell
                        if tkn_sell in sell_tkns:
                            sell_tkns[tkn_sell] -= previous_tkn_sell - agent.holdings[tkn_sell]
                    else:
                        # buy enough to satisfy buy_quantity
                        previous_tkn_sell = agent.holdings[tkn_sell]
                        ex.swap(agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=buy_tkns[tkn_buy])
                        if tkn_sell in sell_tkns:
                            sell_tkns[tkn_sell] -= previous_tkn_sell - agent.holdings[tkn_sell]
                        # actual_swaps[tkn_buy] -= buy_tkns[tkn_buy]
                        # actual_swaps[tkn_sell] -= agent.holdings[tkn_sell] - previous_tkn_sell
                        buy_tkns[tkn_buy] = 0
                        break


def execute_arb(state, cex, agent, all_swaps, buffer=0.0):

    for swap in all_swaps:
        tkn_buy_dex = swap['dex']['buy_asset']
        tkn_sell_dex = swap['dex']['sell_asset']
        tkn_buy_cex = swap['cex']['buy_asset']
        tkn_sell_cex = swap['cex']['sell_asset']
        init_agent = agent.copy()
        if swap['dex']['trade'] == 'buy' and swap['cex']['trade'] == 'sell':
            # omnipool leg
            state.swap(agent, tkn_buy=tkn_buy_dex, tkn_sell=tkn_sell_dex, buy_quantity=swap['dex']['amount'])
            # CEX leg
            cex.swap(agent, tkn_buy=tkn_buy_cex, tkn_sell=tkn_sell_cex, sell_quantity=swap['cex']['amount'])
            dex_tkn_diff = agent.holdings[tkn_buy_dex] - init_agent.holdings[tkn_buy_dex]
            cex_tkn_diff = init_agent.holdings[tkn_sell_cex] - agent.holdings[tkn_sell_cex]
            if dex_tkn_diff != cex_tkn_diff:
                raise
            dex_numeraire_diff = agent.holdings[tkn_sell_dex] - init_agent.holdings[tkn_sell_dex]
            cex_numeraire_diff = agent.holdings[tkn_buy_cex] - init_agent.holdings[tkn_buy_cex]
            if dex_numeraire_diff + cex_numeraire_diff < 0:
                raise
        elif swap['dex']['trade'] == 'sell' and swap['cex']['trade'] == 'buy':
            # omnipool leg
            state.swap(agent, tkn_buy=tkn_buy_dex, tkn_sell=tkn_sell_dex, sell_quantity=swap['dex']['amount'])
            # CEX leg
            cex.swap(agent, tkn_buy=tkn_buy_cex, tkn_sell=tkn_sell_cex, buy_quantity=swap['cex']['amount'])
            dex_tkn_diff = agent.holdings[tkn_sell_dex] - init_agent.holdings[tkn_sell_dex]
            cex_tkn_diff = init_agent.holdings[tkn_buy_cex] - agent.holdings[tkn_buy_cex]
            if dex_tkn_diff != cex_tkn_diff:
                raise
            dex_numeraire_diff = agent.holdings[tkn_buy_dex] - init_agent.holdings[tkn_buy_dex]
            cex_numeraire_diff = agent.holdings[tkn_sell_cex] - init_agent.holdings[tkn_sell_cex]
            if dex_numeraire_diff + cex_numeraire_diff < 0:
                raise
        else:
            raise


def calculate_profit(init_agent, agent, asset_map={}):
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
