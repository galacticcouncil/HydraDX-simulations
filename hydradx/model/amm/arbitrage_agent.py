from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState


# def get_arb_swaps(op_state, order_book, lrna_fee=0.0, asset_fee=0.0, cex_fee=0.0, iters=20):
def get_arb_swaps(op_state, cex, iters=20):
    order_book = cex.order_book
    cex_fee = cex.trade_fee

    all_swaps = {}
    state = op_state.copy()
    for tkn_pair in order_book:
        pair_order_book = order_book[tkn_pair]
        tkn = tkn_pair[0]
        numeraire = tkn_pair[1]
        bids = sorted(pair_order_book.bids, key=lambda x: x[0], reverse=True)
        asks = sorted(pair_order_book.asks, key=lambda x: x[0], reverse=False)

        tkn_lrna_fee = op_state.lrna_fee[tkn].compute(tkn=tkn)
        numeraire_lrna_fee = op_state.lrna_fee[numeraire].compute(tkn=numeraire)
        tkn_asset_fee = op_state.asset_fee[tkn].compute(tkn=tkn)
        numeraire_asset_fee = op_state.asset_fee[numeraire].compute(tkn=numeraire)

        op_spot = OmnipoolState.price(op_state, tkn, numeraire)
        buy_spot = op_spot / ((1 - numeraire_lrna_fee) * (1 - tkn_asset_fee))
        sell_spot = op_spot * (1 - tkn_lrna_fee) * (1 - numeraire_asset_fee)
        swaps = []

        if buy_spot < bids[0][0] * (1 - cex_fee):
            for bid in bids:
                test_agent = Agent(holdings={'USDT': 1000000000, 'DOT': 1000000000, 'HDX': 1000000000}, unique_id='bot')
                amt = calculate_arb_amount_bid(state, tkn, numeraire, bid, cex_fee, precision=1e-10, max_iters=iters)
                state.swap(test_agent, tkn_buy=tkn, tkn_sell=numeraire, buy_quantity=amt)
                op_spot = OmnipoolState.price(state, tkn, numeraire)
                if amt == 0:
                    break
                swaps.append(('buy', {'price': bid[0], 'amount': amt}))
                if amt != bid[1]:
                    break
        elif sell_spot > asks[0][0] / (1 - cex_fee):
            for ask in asks:
                test_agent = Agent(holdings={'USDT': 1000000000, 'DOT': 1000000000, 'HDX': 1000000000}, unique_id='bot')
                amt = calculate_arb_amount_ask(state, tkn, numeraire, ask, cex_fee, precision=1e-10, max_iters=iters)
                state.swap(test_agent, tkn_buy=numeraire, tkn_sell=tkn, sell_quantity=amt)
                op_spot = OmnipoolState.price(state, tkn, numeraire)
                if amt == 0:
                    break
                swaps.append(('sell', {'price': ask[0], 'amount': amt}))
                if amt != ask[1]:
                    break

        print(op_spot)
        all_swaps[tkn_pair] = swaps
    return all_swaps


def calculate_arb_amount_bid(init_state: OmnipoolState, tkn, numeraire, bid: list, cex_fee: float, min_amt = 1e-18, precision = 1e-15, max_iters=None):

    state = init_state.copy()
    agent = Agent(holdings={'USDT': 1000000000, 'DOT': 1000000000, 'HDX': 1000000000}, unique_id='bot')

    asset_fee = state.asset_fee[tkn].compute(tkn=tkn)
    lrna_fee = state.lrna_fee[numeraire].compute(tkn=numeraire)
    cex_price = bid[0] * (1 - cex_fee)

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

    return amt


def calculate_arb_amount_ask(init_state: OmnipoolState, tkn, numeraire, ask: list, cex_fee: float, min_amt = 1e-18, precision = 1e-15, max_iters=None):

    state = init_state.copy()
    agent = Agent(holdings={'USDT': 1000000000, 'DOT': 1000000000, 'HDX': 1000000000}, unique_id='bot')

    asset_fee = state.asset_fee[tkn].compute(tkn=tkn)
    lrna_fee = state.lrna_fee[numeraire].compute(tkn=numeraire)
    cex_price = ask[0] / (1 - cex_fee)

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

    return amt


def execute_arb(state, agent, all_swaps, cex_fee=0.0):

    for tkn_pair in all_swaps:
        swaps = all_swaps[tkn_pair]

        for swap in swaps:

            if swap[0] == 'buy':
                # omnipool leg
                state.swap(agent, tkn_buy=tkn_pair[0], tkn_sell=tkn_pair[1], buy_quantity=swap[1]['amount'])
                # CEX leg
                agent.holdings[tkn_pair[0]] -= swap[1]['amount']
                agent.holdings[tkn_pair[1]] += swap[1]['amount'] * swap[1]['price'] * (1 - cex_fee)
            elif swap[0] == 'sell':
                # omnipool leg
                state.swap(agent, tkn_buy=tkn_pair[1], tkn_sell=tkn_pair[0], sell_quantity=swap[1]['amount'])
                # CEX leg
                agent.holdings[tkn_pair[0]] += swap[1]['amount']
                agent.holdings[tkn_pair[1]] -= swap[1]['amount'] * swap[1]['price'] / (1 - cex_fee)


def calculate_profit(init_agent, agent):
    profit = {tkn: agent.holdings[tkn] - init_agent.holdings[tkn] for tkn in agent.holdings}
    return profit
