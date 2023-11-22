from enum import Enum

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState


# note that this function mutates state, test_agent, test_cex, and max_liquidity
def process_next_swap(text_dex, test_agent, test_cex, tkn_pair, ob_tkn_pair, buffer, max_liquidity_dex,
                      max_liquidity_cex, iters):
    bids, asks = test_cex.order_book[ob_tkn_pair].bids, test_cex.order_book[ob_tkn_pair].asks
    tkn = tkn_pair[0]
    numeraire = tkn_pair[1]
    cex_fee = test_cex.trade_fee
    dex_slippage_tolerance = buffer / 2
    cex_slippage_tolerance = buffer / 2

    tkn_lrna_fee = text_dex.lrna_fee[tkn].compute(tkn=tkn)
    numeraire_lrna_fee = text_dex.lrna_fee[numeraire].compute(tkn=numeraire)
    tkn_asset_fee = text_dex.asset_fee[tkn].compute(tkn=tkn)
    numeraire_asset_fee = text_dex.asset_fee[numeraire].compute(tkn=numeraire)

    op_spot = OmnipoolState.price(text_dex, tkn, numeraire)
    buy_spot = op_spot / ((1 - numeraire_lrna_fee) * (1 - tkn_asset_fee))
    sell_spot = op_spot * (1 - tkn_lrna_fee) * (1 - numeraire_asset_fee)
    swap = {}

    if bids and buy_spot < bids[0][0] * (1 - cex_fee - buffer):  # tkn is coming out of pool, numeraire going into pool
        bid = bids[0]
        max_liq_tkn = max_liquidity_cex[tkn] if tkn in max_liquidity_cex else float('inf')
        max_liq_num = max_liquidity_dex[numeraire] if numeraire in max_liquidity_dex else float('inf')
        amt = calculate_arb_amount_bid(text_dex, tkn, numeraire, bid, cex_fee, buffer, min_amt=1e-6,
                                       max_liq_tkn=max_liq_tkn, max_liq_num=max_liq_num, precision=1e-10,
                                       max_iters=iters)

        if amt != 0:
            init_amt = test_agent.holdings[numeraire]
            text_dex.swap(test_agent, tkn_buy=tkn, tkn_sell=numeraire, buy_quantity=amt)
            amt_in = init_amt - test_agent.holdings[numeraire]
            test_cex.swap(test_agent, tkn_sell=ob_tkn_pair[0], tkn_buy=ob_tkn_pair[1], sell_quantity=amt)
            op_spot = OmnipoolState.price(text_dex, tkn, numeraire)
            swap = {'dex': {'trade': 'buy',
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
                            }}
            if tkn in max_liquidity_cex:
                max_liquidity_cex[tkn] -= amt
            if numeraire in max_liquidity_cex:
                max_liquidity_cex[numeraire] += amt_in
            if tkn in max_liquidity_dex:
                max_liquidity_dex[tkn] += amt
            if numeraire in max_liquidity_dex:
                max_liquidity_dex[numeraire] -= amt_in

    elif asks and sell_spot > asks[0][0] * (1 + cex_fee + buffer):
        ask = asks[0]
        max_liq_tkn = max_liquidity_dex[tkn] if tkn in max_liquidity_dex else float('inf')
        max_liq_num = max_liquidity_cex[numeraire] if numeraire in max_liquidity_cex else float('inf')
        amt = calculate_arb_amount_ask(text_dex, tkn, numeraire, ask, cex_fee, buffer, min_amt=1e-6,
                                       max_liq_tkn=max_liq_tkn, max_liq_num=max_liq_num, precision=1e-10,
                                       max_iters=iters)
        if amt != 0:
            init_amt = test_agent.holdings[numeraire]
            text_dex.swap(test_agent, tkn_buy=numeraire, tkn_sell=tkn, sell_quantity=amt)
            amt_out = test_agent.holdings[numeraire] - init_amt
            test_cex.swap(test_agent, tkn_sell=ob_tkn_pair[1], tkn_buy=ob_tkn_pair[0], buy_quantity=amt)
            op_spot = OmnipoolState.price(text_dex, tkn, numeraire)

            swap = {'dex': {'trade': 'sell',
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
                            }}
            if tkn in max_liquidity_cex:
                max_liquidity_cex[tkn] += amt
            if numeraire in max_liquidity_cex:
                max_liquidity_cex[numeraire] -= amt_out
            if tkn in max_liquidity_dex:
                max_liquidity_dex[tkn] -= amt
            if numeraire in max_liquidity_dex:
                max_liquidity_dex[numeraire] += amt_out

    return swap


def get_buffers(bids, asks, ex1_agent_holdings, ex2_agent_holdings, ex1_tkn_pair, ex2_tkn_pair, buffer,
                execution_risk_buffer, asset_config1, asset_config2, tkn_lrna_fee, numeraire_lrna_fee, tkn_asset_fee,
                numeraire_asset_fee, cex_fee, op_spot):

    buy_spot = op_spot / ((1 - numeraire_lrna_fee) * (1 - tkn_asset_fee))
    sell_spot = op_spot * (1 - tkn_lrna_fee) * (1 - numeraire_asset_fee)

    if bids and buy_spot < bids[0][0]:
        buy_flag = True
        dex_fee = tkn_asset_fee + numeraire_lrna_fee
    elif asks and sell_spot > asks[0][0]:
        buy_flag = False
        dex_fee = numeraire_asset_fee + tkn_lrna_fee
    else:
        raise

    class location(Enum):
        low = 0
        boundary = 1
        high = 2

    iter_obj = [
        [ex1_agent_holdings, ex1_tkn_pair, asset_config1, buy_flag, dex_fee],
        [ex2_agent_holdings, ex2_tkn_pair, asset_config2, not buy_flag, cex_fee]
    ]

    buffers = []

    for (holdings, tkn_pair, asset_config, tkn_buy_flag, fee) in iter_obj:
        if holdings[tkn_pair[0]] < asset_config[tkn_pair[0]]:  # tkn omnipool holdings is low
            tkn_pos = location.low
        elif holdings[tkn_pair[0]] > asset_config[tkn_pair[0]]:  # tkn omnipool holdings is sufficiently high
            tkn_pos = location.high
        else:  # tkn omnipool holdings is on boundary
            tkn_pos = location.boundary

        if holdings[tkn_pair[1]] < asset_config[tkn_pair[1]]:  # tkn omnipool holdings is low
            numeraire_pos = location.low
        elif holdings[tkn_pair[1]] > asset_config[tkn_pair[1]]:  # tkn omnipool holdings is sufficiently high
            numeraire_pos = location.high
        else:  # tkn omnipool holdings is on boundary
            numeraire_pos = location.boundary

        if ((tkn_pos == location.high or (tkn_pos == location.boundary and tkn_buy_flag)) and
            (numeraire_pos == location.high or (numeraire_pos == location.boundary and not tkn_buy_flag))):
            tkn_buffer = execution_risk_buffer / 2
        elif ((tkn_pos in (location.low, location.boundary) and not tkn_buy_flag)
              or (numeraire_pos in (location.low, location.boundary) and tkn_buy_flag)):
            tkn_buffer = buffer / 2
        elif (tkn_pos == location.low and tkn_buy_flag) or (numeraire_pos == location.low and not tkn_buy_flag):
            tkn_buffer = -fee
        else:
            raise

        buffers.append(tkn_buffer)

    return buffers


def process_next_swap_inventory(test_dex, dex_agent, test_cex, cex_agent, tkn_pair, ob_tkn_pair, buffer,
                                asset_config_dex, asset_config_cex, max_liquidity_dex, max_liquidity_cex, iters):
    bids, asks = test_cex.order_book[ob_tkn_pair].bids, test_cex.order_book[ob_tkn_pair].asks
    tkn = tkn_pair[0]
    numeraire = tkn_pair[1]
    cex_fee = test_cex.trade_fee
    dex_slippage_tolerance = buffer / 2
    cex_slippage_tolerance = buffer / 2

    execution_risk_buffer = 0.001

    tkn_lrna_fee = test_dex.lrna_fee[tkn].compute(tkn=tkn)
    numeraire_lrna_fee = test_dex.lrna_fee[numeraire].compute(tkn=numeraire)
    tkn_asset_fee = test_dex.asset_fee[tkn].compute(tkn=tkn)
    numeraire_asset_fee = test_dex.asset_fee[numeraire].compute(tkn=numeraire)

    op_spot = OmnipoolState.price(test_dex, tkn, numeraire)
    buy_spot = op_spot / ((1 - numeraire_lrna_fee) * (1 - tkn_asset_fee))
    sell_spot = op_spot * (1 - tkn_lrna_fee) * (1 - numeraire_asset_fee)
    swap = {}

    buffer = get_buffers(test_cex.order_book[ob_tkn_pair].bids,
                         test_cex.order_book[ob_tkn_pair].asks,
                         dex_agent.holdings,
                         cex_agent.holdings,
                         tkn_pair,
                         ob_tkn_pair,
                         buffer,
                         execution_risk_buffer,
                         asset_config_dex,
                         asset_config_cex,
                         tkn_lrna_fee,
                         numeraire_lrna_fee,
                         tkn_asset_fee,
                         numeraire_asset_fee,
                         cex_fee,
                         op_spot)

    if bids and buy_spot < bids[0][0] * (1 - cex_fee - buffer):  # tkn is coming out of pool, numeraire going into pool
        bid = bids[0]
        max_liq_tkn = max_liquidity_cex[tkn] if tkn in max_liquidity_cex else float('inf')
        max_liq_num = max_liquidity_dex[numeraire] if numeraire in max_liquidity_dex else float('inf')
        amt = calculate_arb_amount_bid(test_dex, tkn, numeraire, bid, cex_fee, buffer, min_amt=1e-6,
                                       max_liq_tkn=max_liq_tkn, max_liq_num=max_liq_num, precision=1e-10,
                                       max_iters=iters)

        if amt != 0:
            init_amt = dex_agent.holdings[numeraire]
            test_dex.swap(dex_agent, tkn_buy=tkn, tkn_sell=numeraire, buy_quantity=amt)
            amt_in = init_amt - dex_agent.holdings[numeraire]
            test_cex.swap(cex_agent, tkn_sell=ob_tkn_pair[0], tkn_buy=ob_tkn_pair[1], sell_quantity=amt)
            op_spot = OmnipoolState.price(test_dex, tkn, numeraire)
            swap = {'dex': {'trade': 'buy',
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
                            }}
            if tkn in max_liquidity_cex:
                max_liquidity_cex[tkn] -= amt
            if numeraire in max_liquidity_cex:
                max_liquidity_cex[numeraire] += amt_in
            if tkn in max_liquidity_dex:
                max_liquidity_dex[tkn] += amt
            if numeraire in max_liquidity_dex:
                max_liquidity_dex[numeraire] -= amt_in

    elif asks and sell_spot > asks[0][0] * (1 + cex_fee + buffer):
        ask = asks[0]
        max_liq_tkn = max_liquidity_dex[tkn] if tkn in max_liquidity_dex else float('inf')
        max_liq_num = max_liquidity_cex[numeraire] if numeraire in max_liquidity_cex else float('inf')
        amt = calculate_arb_amount_ask(test_dex, tkn, numeraire, ask, cex_fee, buffer, min_amt=1e-6,
                                       max_liq_tkn=max_liq_tkn, max_liq_num=max_liq_num, precision=1e-10,
                                       max_iters=iters)
        if amt != 0:
            init_amt = dex_agent.holdings[numeraire]
            test_dex.swap(dex_agent, tkn_buy=numeraire, tkn_sell=tkn, sell_quantity=amt)
            amt_out = dex_agent.holdings[numeraire] - init_amt
            test_cex.swap(cex_agent, tkn_sell=ob_tkn_pair[1], tkn_buy=ob_tkn_pair[0], buy_quantity=amt)
            op_spot = OmnipoolState.price(test_dex, tkn, numeraire)

            swap = {'dex': {'trade': 'sell',
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
                            }}
            if tkn in max_liquidity_cex:
                max_liquidity_cex[tkn] += amt
            if numeraire in max_liquidity_cex:
                max_liquidity_cex[numeraire] -= amt_out
            if tkn in max_liquidity_dex:
                max_liquidity_dex[tkn] -= amt
            if numeraire in max_liquidity_dex:
                max_liquidity_dex[numeraire] += amt_out

    return swap


def get_arb_opps(op_state, cex_dict, config):
    arb_opps = []

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


def does_max_liquidity_allow_trade(dex_tkn_pair, cex_tkn_pair, max_liquidity_dex, max_liquidity_cex):
    if dex_tkn_pair[0] in max_liquidity_dex and max_liquidity_dex[dex_tkn_pair[0]] <= 0:
        return False
    if dex_tkn_pair[1] in max_liquidity_dex and max_liquidity_dex[dex_tkn_pair[1]] <= 0:
        return False
    if cex_tkn_pair[0] in max_liquidity_cex and max_liquidity_cex[cex_tkn_pair[0]] <= 0:
        return False
    if cex_tkn_pair[1] in max_liquidity_cex and max_liquidity_cex[cex_tkn_pair[1]] <= 0:
        return False
    return True


def get_arb_swaps(op_state, cex_dict, config, max_liquidity=None, iters=20):
    arb_opps = get_arb_opps(op_state, cex_dict, config)

    if max_liquidity is None:
        max_liquidity = {'cex': {exchange: {} for exchange in cex_dict}, 'dex': {}}

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
        while not does_max_liquidity_allow_trade(
                arb_cfg['tkn_pair'],
                arb_cfg['order_book'],
                max_liquidity['dex'],
                max_liquidity['cex'][arb_cfg['exchange']]
        ):
            arb_opps.pop(0)
            if not arb_opps:
                return all_swaps
            arb_cfg = config[arb_opps[0][1]]
        swap = process_next_swap(state,
                                 test_agent,
                                 test_cex_dict[arb_cfg['exchange']],
                                 arb_cfg['tkn_pair'],
                                 arb_cfg['order_book'],
                                 arb_cfg['buffer'],
                                 max_liquidity['dex'],
                                 max_liquidity['cex'][arb_cfg['exchange']],
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

    return all_swaps


def get_arb_swaps_inventory(op_state, cex_dict, agent_dict, config, asset_config, max_liquidity=None, iters=20):
    arb_opps = get_arb_opps(op_state, cex_dict, config)

    if max_liquidity is None:
        max_liquidity = {'cex': {exchange: {} for exchange in cex_dict}, 'dex': {}}

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
        while not does_max_liquidity_allow_trade(
                arb_cfg['tkn_pair'],
                arb_cfg['order_book'],
                max_liquidity['dex'],
                max_liquidity['cex'][arb_cfg['exchange']]
        ):
            arb_opps.pop(0)
            if not arb_opps:
                return all_swaps
            arb_cfg = config[arb_opps[0][1]]
        swap = process_next_swap_inventory(state,
                                           agent_dict['omnipool'],
                                           test_cex_dict[arb_cfg['exchange']],
                                           agent_dict[arb_cfg['exchange']],
                                           arb_cfg['tkn_pair'],
                                           arb_cfg['order_book'],
                                           arb_cfg['buffer'],
                                           asset_config['omnipool'],
                                           asset_config[arb_cfg['exchange']],
                                           max_liquidity['dex'],
                                           max_liquidity['cex'][arb_cfg['exchange']],
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

    return all_swaps


def get_arb_swaps_simple(op_state, cex_dict, config, max_liquidity=None, iters=20):
    if max_liquidity is None:
        max_liquidity = {'cex': {exchange: {} for exchange in cex_dict}, 'dex': {}}

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
                                     max_liquidity['dex'],
                                     max_liquidity['cex'][arb_cfg['exchange']],
                                     iters)
            if swap:
                swap['exchange'] = arb_cfg['exchange']
                all_swaps.append(swap)

    return all_swaps


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


def execute_arb(state, cex_dict, agent, all_swaps):
    for swap in all_swaps:
        tkn_buy_dex = swap['dex']['buy_asset']
        tkn_sell_dex = swap['dex']['sell_asset']
        tkn_buy_cex = swap['cex']['buy_asset']
        tkn_sell_cex = swap['cex']['sell_asset']
        init_agent = agent.copy()
        cex = cex_dict[swap['exchange']]
        if swap['dex']['trade'] == 'buy' and swap['cex']['trade'] == 'sell':
            # omnipool leg
            state.swap(agent, tkn_buy=tkn_buy_dex, tkn_sell=tkn_sell_dex, buy_quantity=swap['dex']['amount'])
            # CEX leg
            cex.swap(agent, tkn_buy=tkn_buy_cex, tkn_sell=tkn_sell_cex, sell_quantity=swap['cex']['amount'])
            dex_tkn_diff = agent.holdings[tkn_buy_dex] - init_agent.holdings[tkn_buy_dex]
            cex_tkn_diff = init_agent.holdings[tkn_sell_cex] - agent.holdings[tkn_sell_cex]
            if dex_tkn_diff != cex_tkn_diff:
                print("Error")
            dex_numeraire_diff = agent.holdings[tkn_sell_dex] - init_agent.holdings[tkn_sell_dex]
            cex_numeraire_diff = agent.holdings[tkn_buy_cex] - init_agent.holdings[tkn_buy_cex]
            if dex_numeraire_diff + cex_numeraire_diff < 0:
                print("Error")
        elif swap['dex']['trade'] == 'sell' and swap['cex']['trade'] == 'buy':
            # omnipool leg
            state.swap(agent, tkn_buy=tkn_buy_dex, tkn_sell=tkn_sell_dex, sell_quantity=swap['dex']['amount'])
            # CEX leg
            cex.swap(agent, tkn_buy=tkn_buy_cex, tkn_sell=tkn_sell_cex, buy_quantity=swap['cex']['amount'])
            dex_tkn_diff = agent.holdings[tkn_sell_dex] - init_agent.holdings[tkn_sell_dex]
            cex_tkn_diff = init_agent.holdings[tkn_buy_cex] - agent.holdings[tkn_buy_cex]
            if dex_tkn_diff != cex_tkn_diff:
                print("Error")
            dex_numeraire_diff = agent.holdings[tkn_buy_dex] - init_agent.holdings[tkn_buy_dex]
            cex_numeraire_diff = agent.holdings[tkn_sell_cex] - init_agent.holdings[tkn_sell_cex]
            if dex_numeraire_diff + cex_numeraire_diff < 0:
                print("Error")
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
