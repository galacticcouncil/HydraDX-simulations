from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.arbitrage_agent import calculate_arb_amount_bid, calculate_arb_amount_ask


def get_arb_opps(op_state, order_book, order_book_map, cex_fee, buffer):
    arb_opps = []

    for tkn_pair in order_book_map:
        ob_tkn_pair = order_book_map[tkn_pair]
        pair_order_book = order_book[ob_tkn_pair]

        dex_spot_price = OmnipoolState.price(op_state, tkn_pair[0], tkn_pair[1])

        if len(pair_order_book.bids) > 0:
            bid_price = pair_order_book.bids[0][0]
            cex_sell_price = bid_price * (1 - cex_fee - buffer[tkn_pair])

            numeraire_lrna_fee = op_state.lrna_fee[tkn_pair[1]].compute(tkn=tkn_pair[1])
            tkn_asset_fee = op_state.asset_fee[tkn_pair[0]].compute(tkn=tkn_pair[0])
            dex_buy_price = dex_spot_price / ((1 - tkn_asset_fee) * (1 - numeraire_lrna_fee))

            if dex_buy_price < cex_sell_price:  # buy from DEX, sell to CEX
                arb_opps.append(((cex_sell_price - dex_buy_price) / dex_buy_price, tkn_pair))

        if len(pair_order_book.asks) > 0:
            ask_price = pair_order_book.asks[0][0]
            cex_buy_price = ask_price * (1 + cex_fee + buffer[tkn_pair])

            numeraire_asset_fee = op_state.asset_fee[tkn_pair[1]].compute(tkn=tkn_pair[1])
            tkn_lrna_fee = op_state.lrna_fee[tkn_pair[0]].compute(tkn=tkn_pair[0])
            dex_sell_price = dex_spot_price * (1 - numeraire_asset_fee) * (1 - tkn_lrna_fee)

            if dex_sell_price > cex_buy_price:  # buy from CEX, sell to DEX
                arb_opps.append(((dex_sell_price - cex_buy_price) / cex_buy_price, tkn_pair))

    arb_opps.sort(key=lambda x: x[0], reverse=True)
    return arb_opps


def get_arb_swaps(op_state, cex, order_book_map, buffer={}, max_liquidity={'cex': {}, 'dex': {}}, iters=20):
    cex_fee = cex.trade_fee
    if type(buffer) == float:
        buffer_filled = {k: buffer for k in order_book_map}
    else:
        buffer_filled = {k: buffer[k] if k in buffer else 0 for k in order_book_map}
    dex_slippage_tolerance = {k: buffer_filled[k]/2 for k in order_book_map}
    cex_slippage_tolerance = {k: buffer_filled[k]/2 for k in order_book_map}

    arb_opps = get_arb_opps(op_state, cex.order_book, order_book_map, cex_fee, buffer_filled)

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

        if buy_spot < bids[0][0] * (1 - cex_fee - buffer_filled[tkn_pair]):  # tkn is coming out of pool, numeraire going into pool
            bid = bids[0]
            max_liq_tkn = max_liquidity['cex'][tkn] if tkn in max_liquidity['cex'] else float('inf')
            max_liq_num = max_liquidity['dex'][numeraire] if numeraire in max_liquidity['dex'] else float('inf')
            amt = calculate_arb_amount_bid(state, tkn, numeraire, bid, cex_fee, buffer_filled[tkn_pair], max_liq_tkn=max_liq_tkn,
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
                                      'max_sell': amt_in * (1 + dex_slippage_tolerance[tkn_pair])
                                      },
                              'cex': {'trade': 'sell',
                                      'buy_asset': ob_tkn_pair[1],
                                      'sell_asset': ob_tkn_pair[0],
                                      'price': bid[0] * (1 - cex_slippage_tolerance[tkn_pair]),
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


        elif sell_spot > asks[0][0] * (1 + cex_fee + buffer_filled[tkn_pair]):
            ask = asks[0]
            max_liq_tkn = max_liquidity['dex'][tkn] if tkn in max_liquidity['dex'] else float('inf')
            max_liq_num = max_liquidity['cex'][numeraire] if numeraire in max_liquidity['cex'] else float('inf')
            amt = calculate_arb_amount_ask(state, tkn, numeraire, ask, cex_fee, buffer_filled[tkn_pair], max_liq_tkn=max_liq_tkn,
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
                                      'min_buy': amt_out * (1 - dex_slippage_tolerance[tkn_pair])},
                              'cex': {'trade': 'buy',
                                      'buy_asset': ob_tkn_pair[0],
                                      'sell_asset': ob_tkn_pair[1],
                                      'price': ask[0] * (1 + cex_slippage_tolerance[tkn_pair]),
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
        new_arb_opps = get_arb_opps(state, order_book, order_book_map, cex_fee, buffer_filled)
        if arb_opps and new_arb_opps and arb_opps[0][0] == new_arb_opps[0][0]:
            break
        arb_opps = new_arb_opps

    return all_swaps
