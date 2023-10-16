from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState


def get_arb_swaps(op_state, order_book, lrna_fee=0.0, asset_fee=0.0, cex_fee=0.0, iters=10):
    all_swaps = {}
    for tkn_pair in order_book:
        pair_order_book = order_book[tkn_pair]
        tkn = tkn_pair[0]
        numeraire = tkn_pair[1]
        bids = sorted(pair_order_book['bids'], key=lambda x: x['price'], reverse=True)
        asks = sorted(pair_order_book['asks'], key=lambda x: x['price'], reverse=False)

        agent = Agent(holdings={'USDT': 1000000, 'DOT': 1000000, 'HDX': 1000000}, unique_id='bot')

        op_spot = OmnipoolState.price(op_state, tkn, numeraire)
        buy_spot = op_spot / ((1 - lrna_fee) * (1 - asset_fee))
        sell_spot = op_spot * (1 - lrna_fee) * (1 - asset_fee)
        i = 0
        bid_executed = False
        ask_executed = False
        if buy_spot < bids[i]['price'] * (1 - cex_fee):
            bid_executed = True
            test_state = op_state
            test_agent = agent
            while i < len(bids) and buy_spot < bids[i]['price'] * (
                    1 - cex_fee):  # spot is lower than the highest bid, so we should buy from Omnipool and sell to CEX
                print('buy')
                op_state = test_state
                agent = test_agent
                test_state = op_state.copy()
                test_agent = agent.copy()
                test_state.swap(test_agent, tkn_buy=tkn, tkn_sell=numeraire, buy_quantity=bids[i]['amount'])
                op_spot = OmnipoolState.price(test_state, tkn, numeraire)
                buy_spot = op_spot / ((1 - lrna_fee) * (1 - asset_fee))
                i += 1
        elif sell_spot > asks[i]['price'] / (1 - cex_fee):
            ask_executed = True
            test_state = op_state
            test_agent = agent
            while i < len(bids) and sell_spot > asks[i]['price'] / (
                    1 - cex_fee):  # spot is higher than the lowest ask, so we should buy from CEX and sell to Omnipool
                print('sell')
                op_state = test_state
                agent = test_agent
                test_state = op_state.copy()
                test_agent = agent.copy()
                test_state.swap(test_agent, tkn_buy=numeraire, tkn_sell=tkn, sell_quantity=asks[i]['amount'])
                op_spot = OmnipoolState.price(test_state, tkn, numeraire)
                sell_spot = op_spot * (1 - lrna_fee) * (1 - asset_fee)
                i += 1

        swaps = []
        if bid_executed:
            if buy_spot < bids[i - 1]['price'] * (
                    1 - cex_fee):  # last trade can be fully executed without moving spot price too much
                swaps = [('buy', bids[j]) for j in range(i)]
            else:  # Use binary search to partially fill last order
                swaps = bids[0:i - 1]
                swaps = [('buy', bids[j]) for j in range(i - 1)]
                bid = bids[i - 1]
                amt_high = bid['amount']
                amt_low = 0

                for i in range(iters):
                    test_agent = agent.copy()
                    test_state = op_state.copy()
                    amt = amt_low + (amt_high - amt_low) / 2
                    test_state.swap(test_agent, tkn_buy=tkn, tkn_sell=numeraire, buy_quantity=amt)
                    op_spot = OmnipoolState.price(test_state, tkn, numeraire)
                    buy_spot = op_spot / ((1 - lrna_fee) * (1 - asset_fee))
                    if buy_spot > bid['price'] * (1 - cex_fee):
                        amt_high = amt
                    elif buy_spot < bid['price'] * (1 - cex_fee):
                        amt_low = amt
                    else:
                        break

                op_state.swap(agent, tkn_buy='DOT', tkn_sell=numeraire, buy_quantity=amt_low)
                op_spot = OmnipoolState.price(op_state, tkn, numeraire)
                swaps.append(('buy', {'price': bid['price'], 'amount': amt_low}))


        elif ask_executed:
            if sell_spot > asks[i - 1]['price'] / (
                    1 - cex_fee):  # last trade can be fully executed without moving spot price too much
                swaps = [('sell', asks[j]) for j in range(i)]
            else:  # Use binary search to partially fill last order
                swaps = [('sell', asks[j]) for j in range(i - 1)]
                ask = asks[i - 1]
                amt_high = ask['amount']
                amt_low = 0
                for i in range(iters):
                    test_agent = agent.copy()
                    test_state = op_state.copy()
                    amt = amt_low + (amt_high - amt_low) / 2
                    test_state.swap(test_agent, tkn_buy=numeraire, tkn_sell=tkn, sell_quantity=amt)
                    op_spot = OmnipoolState.price(test_state, tkn, numeraire)
                    sell_spot = op_spot * (1 - lrna_fee) * (1 - asset_fee)
                    if sell_spot < ask['price'] / (1 - cex_fee):
                        amt_high = amt
                    elif sell_spot > ask['price'] / (1 - cex_fee):
                        amt_low = amt
                    else:
                        break

                op_state.swap(agent, tkn_buy=numeraire, tkn_sell=tkn, buy_quantity=amt_low)
                op_spot = OmnipoolState.price(op_state, tkn, numeraire)
                swaps.append(('sell', {'price': ask['price'], 'amount': amt_low}))

        print(op_spot)
        all_swaps[tkn_pair] = swaps
    return all_swaps


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
                agent.holdings[tkn_pair[0]] += swap[1]['amount'] * (1 - cex_fee)
                agent.holdings[tkn_pair[1]] -= swap[1]['amount'] * swap[1]['price']


def calculate_profit(init_agent, agent):
    profit = {tkn: agent.holdings[tkn] - init_agent.holdings[tkn] for tkn in agent.holdings}
    return profit
