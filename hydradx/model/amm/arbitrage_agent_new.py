from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.arbitrage_agent import process_next_swap


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
    # dex_slippage_tolerance = {k: buffer_filled[k]/2 for k in order_book_map}
    # cex_slippage_tolerance = {k: buffer_filled[k]/2 for k in order_book_map}

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
        swap = process_next_swap(state, test_agent, test_cex, tkn_pair, order_book_map, buffer_filled[tkn_pair],
                                 max_liquidity, iters)
        all_swaps.append(swap)
        new_arb_opps = get_arb_opps(state, order_book, order_book_map, cex_fee, buffer_filled)
        if arb_opps and new_arb_opps and arb_opps[0][0] == new_arb_opps[0][0]:
            break
        arb_opps = new_arb_opps

    return all_swaps
