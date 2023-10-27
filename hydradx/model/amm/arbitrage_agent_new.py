from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState


def get_arb_opps(op_state, order_book, order_book_map, cex_fee, buffer):
    arb_opps = []

    for tkn_pair in order_book_map:
        ob_tkn_pair = order_book_map[tkn_pair]
        pair_order_book = order_book[ob_tkn_pair]
        bid_price = pair_order_book.bids[0][0]
        cex_sell_price = bid_price * (1 - cex_fee - buffer)
        ask_price = pair_order_book.asks[0][0]
        cex_buy_price = ask_price * (1 + cex_fee + buffer)
        dex_spot_price = OmnipoolState.price(op_state, tkn_pair[0], tkn_pair[1])
        tkn_asset_fee = op_state.asset_fee[tkn_pair[0]].compute(tkn=tkn_pair[0])
        numeraire_asset_fee = op_state.asset_fee[tkn_pair[1]].compute(tkn=tkn_pair[1])
        tkn_lrna_fee = op_state.lrna_fee[tkn_pair[0]].compute(tkn=tkn_pair[0])
        numeraire_lrna_fee = op_state.lrna_fee[tkn_pair[1]].compute(tkn=tkn_pair[1])
        dex_buy_price = dex_spot_price / ((1 - tkn_asset_fee) * (1 - numeraire_lrna_fee))
        dex_sell_price = dex_spot_price * (1 - numeraire_asset_fee) * (1 - tkn_lrna_fee)

        if dex_sell_price > cex_buy_price:  # buy from CEX, sell to DEX
            arb_opps.append(((dex_sell_price - cex_buy_price) / cex_buy_price, tkn_pair))
        elif dex_buy_price < cex_sell_price:  # buy from DEX, sell to CEX
            arb_opps.append(((cex_sell_price - dex_buy_price) / dex_buy_price, tkn_pair))
    arb_opps.sort(key=lambda x: x[0], reverse=True)
    return arb_opps


def get_arb_swaps(op_state, cex, order_book_map, buffer=0.0, max_trades={}, iters=20):
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
    # for tkn_pair in order_book_map:
    while arb_opps:
        tkn_pair = arb_opps[0][1]
        ob_tkn_pair = order_book_map[tkn_pair]
        max_trade_tkn = max_trades[tkn_pair] if tkn_pair in max_trades else float('inf')
        pair_order_book = order_book[ob_tkn_pair]
        tkn = tkn_pair[0]
        numeraire = tkn_pair[1]
        bids = sorted(pair_order_book.bids, key=lambda x: x[0], reverse=True)
        asks = sorted(pair_order_book.asks, key=lambda x: x[0], reverse=False)

        tkn_lrna_fee = state.lrna_fee[tkn].compute(tkn=tkn)
        numeraire_lrna_fee = state.lrna_fee[numeraire].compute(tkn=numeraire)
        tkn_asset_fee = state.asset_fee[tkn].compute(tkn=tkn)
        numeraire_asset_fee = state.asset_fee[numeraire].compute(tkn=numeraire)

        op_spot = OmnipoolState.price(state, tkn, numeraire)
        buy_spot = op_spot / ((1 - numeraire_lrna_fee) * (1 - tkn_asset_fee))
        sell_spot = op_spot * (1 - tkn_lrna_fee) * (1 - numeraire_asset_fee)

        if buy_spot < bids[0][0] * (1 - cex_fee - buffer):
            bid = bids[0]
            if tkn in max_trades:
                amt = calculate_arb_amount_bid(state, tkn, numeraire, bid, cex_fee, buffer, precision=1e-10,
                                               max_iters=iters)
            else:
                amt = calculate_arb_amount_bid(state, tkn, numeraire, bid, cex_fee, buffer, precision=1e-10,
                                               max_iters=iters)
            if amt != 0:
                state.swap(test_agent, tkn_buy=tkn, tkn_sell=numeraire, buy_quantity=amt)
                amt_in = init_amt - test_agent.holdings[numeraire]
                test_cex.swap(test_agent, tkn_sell=ob_tkn_pair[0], tkn_buy=ob_tkn_pair[1], sell_quantity=amt)
                op_spot = OmnipoolState.price(state, tkn, numeraire)
                amt_adjusted = min(amt, max_trade_tkn)
                amt_in_adjusted = amt_in * (amt_adjusted / amt)
                all_swaps.append({'dex': {'trade': 'buy',
                                      'buy_asset': tkn,
                                      'sell_asset': numeraire,
                                      'price': bid[0],
                                      'amount': amt_adjusted,
                                      'max_sell': amt_in_adjusted * (1 + dex_slippage_tolerance)
                                      },
                              'cex': {'trade': 'sell',
                                      'buy_asset': ob_tkn_pair[1],
                                      'sell_asset': ob_tkn_pair[0],
                                      'price': bid[0] * (1 - cex_slippage_tolerance),
                                      'amount': amt_adjusted
                                      }})
                max_trade_tkn -= min(amt, max_trade_tkn)

        elif sell_spot > asks[0][0] * (1 + cex_fee + buffer):
            ask = asks[0]
            amt = calculate_arb_amount_ask(state, tkn, numeraire, ask, cex_fee, buffer, precision=1e-10, max_iters=iters)
            if amt != 0:
                state.swap(test_agent, tkn_buy=numeraire, tkn_sell=tkn, sell_quantity=amt)
                amt_out = test_agent.holdings[numeraire] - init_amt
                test_cex.swap(test_agent, tkn_sell=ob_tkn_pair[1], tkn_buy=ob_tkn_pair[0], buy_quantity=amt)
                op_spot = OmnipoolState.price(state, tkn, numeraire)

                amt_adjusted = min(amt, max_trade_tkn)
                amt_out_adjusted = amt_out * (amt_adjusted / amt)
                all_swaps.append({'dex': {'trade': 'sell',
                                      'buy_asset': numeraire,
                                      'sell_asset': tkn,
                                      'price': ask[0],
                                      'amount': amt_adjusted,
                                      'min_buy': amt_out_adjusted * (1 - dex_slippage_tolerance)},
                              'cex': {'trade': 'buy',
                                      'buy_asset': ob_tkn_pair[0],
                                      'sell_asset': ob_tkn_pair[1],
                                      'price': ask[0] * (1 + cex_slippage_tolerance),
                                      'amount': amt_adjusted
                                      }})
                max_trade_tkn -= min(amt, max_trade_tkn)

        new_arb_opps = get_arb_opps(state, order_book, order_book_map, cex_fee, buffer)
        if arb_opps[0][0] == new_arb_opps[0][0]:
            break
        arb_opps = new_arb_opps

    return all_swaps


def calculate_arb_amount_bid(
        init_state: OmnipoolState,
        tkn, numeraire,
        bid: list,
        cex_fee: float = 0.0,
        buffer: float = 0.0,
        min_amt=1e-18,
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

    op_spot = OmnipoolState.price(state, tkn, numeraire)
    buy_spot = op_spot / ((1 - lrna_fee) * (1 - asset_fee))

    # we use binary search to find the amount that can be swapped
    amt_low = min_amt
    amt_high = bid[1]
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
            if amt_high == amt_low:  # full amount can be traded
                break
        if amt >= state.liquidity[tkn] or buy_spot > cex_price or test_state.fail != '':
            amt_high = amt
        elif buy_spot < cex_price:
            amt_low = amt
            best_buy_spot = buy_spot

        # only want to update amt if there will be another iteration
        if cex_price - best_buy_spot > precision:
            amt = amt_low + (amt_high - amt_low) / 2

        i += 1
        if max_iters is not None and i >= max_iters:
            break

    return amt_low


def calculate_arb_amount_ask(
        init_state: OmnipoolState,
        tkn,
        numeraire,
        ask: list,
        cex_fee: float = 0.0,
        buffer: float = 0.0,
        min_amt=1e-18,
        precision=1e-15,
        max_iters=None
):
    state = init_state.copy()
    init_amt = 1000000000
    holdings = {asset: init_amt for asset in [tkn, numeraire]}
    agent = Agent(holdings=holdings, unique_id='bot')

    asset_fee = state.asset_fee[numeraire].compute(tkn=numeraire)
    lrna_fee = state.lrna_fee[tkn].compute(tkn=tkn)
    # cex_price = ask[0] / (1 - cex_fee)
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
    amt_high = ask[1]
    amt = amt_high
    i = 0
    best_sell_spot = sell_spot
    while best_sell_spot - cex_price > precision:
        test_agent = agent.copy()
        test_state = state.copy()
        test_state.swap(test_agent, tkn_buy=numeraire, tkn_sell=tkn, sell_quantity=amt)
        op_spot = OmnipoolState.price(test_state, tkn, numeraire)
        sell_spot = op_spot * (1 - lrna_fee) * (1 - asset_fee)
        if amt_high == amt_low:  # full amount can be traded
            break
        if sell_spot < cex_price or test_state.fail != '':
            amt_high = amt
        elif sell_spot > cex_price:
            amt_low = amt
            best_sell_spot = sell_spot

        # only want to update amt if there will be another iteration
        if best_sell_spot - cex_price > precision:
            amt = amt_low + (amt_high - amt_low) / 2

        i += 1
        if max_iters is not None and i >= max_iters:
            break

    return amt_low


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