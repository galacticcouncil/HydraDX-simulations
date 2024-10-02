import copy

from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm import omnix
from hydradx.model.amm.agents import Agent

def calculate_price_slippage_to_impact(omnipool: OmnipoolState, intent: dict) -> float:
    tkn_buy, tkn_sell = intent['tkn_buy'], intent['tkn_sell']
    sell_price = omnipool.price(omnipool, tkn_sell, tkn_buy)
    buy_liquidity, sell_liquidity = omnipool.liquidity[tkn_buy], omnipool.liquidity[tkn_sell]
    buy_lrna, sell_lrna = omnipool.lrna[tkn_buy], omnipool.lrna[tkn_sell]

    if 'sell_quantity' in intent:
        if sell_lrna <= buy_lrna:
            impact = intent['sell_quantity'] / sell_liquidity
        else:
            impact = intent['sell_quantity'] * sell_price / buy_liquidity
        limit_price = intent['buy_limit'] / intent['sell_quantity']
    else:
        if sell_lrna <= buy_lrna:
            impact = intent['buy_quantity'] / sell_price / sell_liquidity
        else:
            impact = intent['buy_quantity'] / buy_liquidity
        limit_price = intent['buy_quantity'] / intent['sell_limit']
    slippage = (sell_price - limit_price) / sell_price

    return slippage / impact


def calculate_price_slippage(omnipool: OmnipoolState, intent: dict) -> float:
    sell_price = omnipool.price(omnipool, intent['tkn_sell'], intent['tkn_buy'])

    if 'sell_quantity' in intent:
        limit_price = intent['buy_limit'] / intent['sell_quantity']
    else:
        limit_price = intent['buy_quantity'] / intent['sell_limit']

    return (sell_price - limit_price) / sell_price


def _calculate_price_slippage_to_impact_sign_tuple(omnipool: OmnipoolState, intent: dict) -> tuple:
    x = calculate_price_slippage_to_impact(omnipool, intent)
    return (1, x) if x >= 0 else (-1, -x)


def _calculate_price_slippage_sign_tuple(omnipool: OmnipoolState, intent: dict) -> tuple:
    x = calculate_price_slippage(omnipool, intent)
    return (1, x) if x >= 0 else (-1, -x)


def get_sorted_intents(omnipool: OmnipoolState, intents: list) -> list:
    return sorted(intents, key=lambda intent: _calculate_price_slippage_to_impact_sign_tuple(omnipool, intent),
                  reverse=True)


def get_sorted_intents_slippage(omnipool: OmnipoolState, intents: list) -> list:
    return sorted(intents, key=lambda intent: _calculate_price_slippage_sign_tuple(omnipool, intent),
                  reverse=True)


def try_swap(omnipool: OmnipoolState, swap_intent: dict) -> dict:
    '''
    Does swap (mutating omnipool and agent) if swap_intent can be executed respecting price limits.
    Returns the token differences in Omnipool for both tokens and LRNA
    '''
    init_buy_lrna = omnipool.lrna[swap_intent['tkn_buy']]
    init_sell_lrna = omnipool.lrna[swap_intent['tkn_sell']]
    if 'sell_quantity' in swap_intent:
        sell_amt = swap_intent['sell_quantity']
        buy_amt = omnipool.calculate_buy_from_sell(swap_intent['tkn_buy'], swap_intent['tkn_sell'],
                                                   swap_intent['sell_quantity'])
        if buy_amt < swap_intent['buy_limit']:
            return {'buy_tkn': 0, 'sell_tkn': 0, 'buy_lrna': 0, 'sell_lrna': 0}
        if not swap_intent['agent'].is_holding(swap_intent['tkn_sell'], sell_amt):
            return {'buy_tkn': 0, 'sell_tkn': 0, 'buy_lrna': 0, 'sell_lrna': 0}
        omnipool.swap(swap_intent['agent'], swap_intent['tkn_buy'], swap_intent['tkn_sell'],
                      sell_quantity=swap_intent['sell_quantity'])
    else:
        buy_amt = swap_intent['buy_quantity']
        sell_amt = omnipool.calculate_sell_from_buy(swap_intent['tkn_buy'], swap_intent['tkn_sell'],
                                                    swap_intent['buy_quantity'])
        if sell_amt > swap_intent['sell_limit']:
            return {'buy_tkn': 0, 'sell_tkn': 0, 'buy_lrna': 0, 'sell_lrna': 0}
        if not swap_intent['agent'].is_holding(swap_intent['tkn_sell'], sell_amt):
            return {'buy_tkn': 0, 'sell_tkn': 0, 'buy_lrna': 0, 'sell_lrna': 0}
        omnipool.swap(swap_intent['agent'], swap_intent['tkn_buy'], swap_intent['tkn_sell'],
                      buy_quantity=swap_intent['buy_quantity'])
    return {
        'buy_tkn': buy_amt,
        'sell_tkn': sell_amt,
        'buy_lrna': omnipool.lrna[swap_intent['tkn_buy']] - init_buy_lrna,
        'sell_lrna': init_sell_lrna - omnipool.lrna[swap_intent['tkn_sell']]
    }


def calculate_deltas(init_intents: list, intents: list) -> dict:
    deltas = {}
    for i in range(len(intents)):
        intent, init_intent = intents[i], init_intents[i]
        buy_amt = intent['agent'].holdings[intent['tkn_buy']] - init_intent['agent'].holdings[init_intent['tkn_buy']]
        sell_amt = init_intent['agent'].holdings[init_intent['tkn_sell']] - intent['agent'].holdings[intent['tkn_sell']]
        if intent['tkn_buy'] not in deltas:
            deltas[intent['tkn_buy']] = {"in": 0, "out": 0}
        deltas[intent['tkn_buy']]["out"] += buy_amt
        if intent['tkn_sell'] not in deltas:
            deltas[intent['tkn_sell']] = {"in": 0, "out": 0}
        deltas[intent['tkn_sell']]["in"] += sell_amt
    return deltas


def calculate_prices(omnipool: OmnipoolState, tkn_deltas: dict, lrna_deltas: dict) -> tuple:
    buy_prices = {}
    sell_prices = {}
    for tkn in tkn_deltas:
        if tkn_deltas[tkn]["in"] > tkn_deltas[tkn]["out"]:
            buy_prices[tkn] = omnipool.price(omnipool, tkn, "LRNA")
            # matched amount being sold is traded at final spot too
            lrna_for_matched = buy_prices[tkn] * tkn_deltas[tkn]["out"]
            # net price is total lrna divided by total asset in
            if lrna_deltas[tkn]["out"] <= lrna_deltas[tkn]["in"]:
                raise Exception("lrna_deltas[tkn][out] <= lrna_deltas[tkn][in]")
            sell_prices[tkn] = (lrna_deltas[tkn]["out"] - lrna_deltas[tkn]["in"] + lrna_for_matched) / tkn_deltas[tkn][
                "in"]
        elif tkn_deltas[tkn]["out"] > tkn_deltas[tkn]["in"]:
            sell_prices[tkn] = omnipool.price(omnipool, tkn, "LRNA")
            # matched amount being bought is traded at final spot too
            lrna_for_matched = sell_prices[tkn] * tkn_deltas[tkn]["in"]
            # net price is total lrna divided by total asset out
            assert lrna_deltas[tkn]["in"] > lrna_deltas[tkn]["out"]
            buy_prices[tkn] = (lrna_deltas[tkn]["in"] - lrna_deltas[tkn]["out"] + lrna_for_matched) / tkn_deltas[tkn][
                "out"]
        else:
            sell_prices[tkn] = omnipool.price(omnipool, tkn, "LRNA")
            buy_prices[tkn] = omnipool.price(omnipool, tkn, "LRNA")

    return buy_prices, sell_prices


def try_swaps(omnipool, sorted_intents, skip_indices=None):
    if skip_indices is None:
        skip_indices = []
    amts = []
    tkn_deltas = {}
    lrna_deltas = {}
    for i in range(len(sorted_intents)):
        intent = sorted_intents[i]
        if i in skip_indices:
            deltas = {
                'buy_tkn': 0,
                'sell_tkn': 0,
                'buy_lrna': 0,
                'sell_lrna': 0
            }
        else:
            deltas = try_swap(omnipool, intent)
        if intent['tkn_buy'] not in tkn_deltas:
            tkn_deltas[intent['tkn_buy']] = {"in": 0, "out": 0}
        tkn_deltas[intent['tkn_buy']]["out"] += deltas['buy_tkn']
        if intent['tkn_sell'] not in tkn_deltas:
            tkn_deltas[intent['tkn_sell']] = {"in": 0, "out": 0}
        tkn_deltas[intent['tkn_sell']]["in"] += deltas['sell_tkn']
        if intent['tkn_buy'] not in lrna_deltas:
            lrna_deltas[intent['tkn_buy']] = {"in": 0, "out": 0}
        lrna_deltas[intent['tkn_buy']]["in"] += deltas['buy_lrna']
        if intent['tkn_sell'] not in lrna_deltas:
            lrna_deltas[intent['tkn_sell']] = {"in": 0, "out": 0}
        lrna_deltas[intent['tkn_sell']]["out"] += deltas['sell_lrna']

        if "sell_quantity" in intent:
            amt = deltas['sell_tkn']
        else:
            amt = deltas['buy_tkn']

        amts.append(amt)
        # if amt > 0:
        #     exec_trade_indices.append(i)

    return tkn_deltas, lrna_deltas, amts


def calculate_price_limits(i: int, intents: list, amounts: list):
    price_limits = {
        'compound_same_limit': None,
        'compound_inverse_limit': None,
        'buy_tkn_same_limit': None,
        'buy_tkn_inverse_limit': None,
        'sell_tkn_same_limit': None,
        'sell_tkn_inverse_limit': None
    }
    intent = intents[i]
    for j in range(i):  # iterate through already-processed intents
        if amounts[j] > 0:  # only worry about intents which are being executed
            included_intent = intents[j]
            if 'sell_quantity' in included_intent:
                price_limit = included_intent['buy_limit'] / included_intent['sell_quantity']
            else:
                price_limit = included_intent['buy_quantity'] / included_intent['sell_limit']

            if included_intent['tkn_buy'] == intent['tkn_buy']:
                if included_intent['tkn_sell'] == intent['tkn_sell']:
                    if price_limits['compound_same_limit'] is None:
                        price_limits['compound_same_limit'] = price_limit
                    else:
                        price_limits['compound_same_limit'] = max(price_limits['compound_same_limit'], price_limit)
                else:
                    if price_limits['buy_tkn_same_limit'] is None:
                        price_limits['buy_tkn_same_limit'] = price_limit
                    else:
                        price_limits['buy_tkn_same_limit'] = max(price_limits['buy_tkn_same_limit'], price_limit)
            elif included_intent['tkn_buy'] == intent['tkn_sell']:
                if included_intent['tkn_sell'] == intent['tkn_buy']:
                    if price_limits['compound_inverse_limit'] is None:
                        price_limits['compound_inverse_limit'] = 1 / price_limit
                    else:
                        price_limits['compound_inverse_limit'] = min(price_limits['compound_inverse_limit'],
                                                                     1 / price_limit)
                else:
                    if price_limits['sell_tkn_inverse_limit'] is None:
                        price_limits['sell_tkn_inverse_limit'] = 1 / price_limit
                    else:
                        price_limits['sell_tkn_inverse_limit'] = min(price_limits['sell_tkn_inverse_limit'],
                                                                     1 / price_limit)
            elif included_intent['tkn_sell'] == intent['tkn_sell']:
                if price_limits['sell_tkn_same_limit'] is None:
                    price_limits['sell_tkn_same_limit'] = price_limit
                else:
                    price_limits['sell_tkn_same_limit'] = max(price_limits['sell_tkn_same_limit'], price_limit)
            elif included_intent['tkn_sell'] == intent['tkn_buy']:
                if price_limits['buy_tkn_inverse_limit'] is None:
                    price_limits['buy_tkn_inverse_limit'] = 1 / price_limit
                else:
                    price_limits['buy_tkn_inverse_limit'] = min(price_limits['buy_tkn_inverse_limit'], 1 / price_limit)
    return price_limits


def calculate_net_intents(
        intents: list,
        tkn_buy: str,
        tkn_sell: str,
        buy_prices: dict,
        sell_prices: dict
) -> tuple:
    # keys will be [<tkn_sell>_<tkn_buy>][0/1] where 0 indicates sell is specified
    net_intent_dict = {}
    price_limits = {
        'min_tkn_buy_tkn_sell_price': 0,
        'max_tkn_buy_tkn_sell_price': float('inf'),
        'min_tkn_sell_price': 0,
        'max_tkn_sell_price': float('inf'),
        'min_tkn_buy_price': 0,
        'max_tkn_buy_price': float('inf')
    }

    for intent in intents:
        if intent['tkn_sell'] == tkn_sell:
            tb = tkn_buy if intent['tkn_buy'] == tkn_buy else "LRNA"
            ts = tkn_sell
        elif intent['tkn_sell'] == tkn_buy:
            tb = tkn_sell if intent['tkn_buy'] == tkn_sell else "LRNA"
            ts = tkn_buy
        elif intent['tkn_buy'] == tkn_buy:
            tb = tkn_buy
            ts = "LRNA"
        elif intent['tkn_buy'] == tkn_sell:
            tb = tkn_sell
            ts = "LRNA"
        else:
            continue

        if 'sell_quantity' in intent:
            is_buy = 0
            amt_key = 'sell_quantity'
            amt = intent['sell_quantity']
            if ts == "LRNA":  # convert to LRNA if necessary
                amt *= sell_prices[intent['tkn_sell']]
        else:
            is_buy = 1
            amt_key = 'buy_quantity'
            amt = intent['buy_quantity']
            if tb == "LRNA":  # convert to LRNA if necessary
                amt *= buy_prices[intent['tkn_buy']]

        k = ts + "_" + tb
        if k not in net_intent_dict:
            net_intent_dict[k] = {}
        if is_buy not in net_intent_dict[k]:
            net_intent_dict[k][is_buy] = {
                amt_key: 0,
                'tkn_sell': ts,
                'tkn_buy': tb
            }
        net_intent_dict[k][is_buy][amt_key] += amt

        # adjust price limits
        if intent['tkn_sell'] == tkn_sell and intent['tkn_buy'] == tkn_buy:
            # need to update buy/sell price
            if 'buy_quantity' in intent:
                buy_amt = intent['buy_quantity']
                sell_amt = intent['sell_limit']
            else:
                buy_amt = intent['buy_limit']
                sell_amt = intent['sell_quantity']
            price_limit = buy_amt / sell_amt
            price_limits['min_tkn_buy_tkn_sell_price'] = max(price_limits['min_tkn_buy_tkn_sell_price'], price_limit)
        elif intent['tkn_sell'] == tkn_sell:
            if 'buy_quantity' in intent:
                buy_amt = intent['buy_quantity'] * buy_prices[intent['tkn_buy']]
                sell_amt = intent['sell_limit']
            else:
                buy_amt = intent['buy_limit'] * buy_prices[intent['tkn_buy']]
                sell_amt = intent['sell_quantity']
            price_limit = buy_amt / sell_amt
            price_limits['min_tkn_sell_price'] = max(price_limits['min_tkn_sell_price'], price_limit)
        elif intent['tkn_buy'] == tkn_buy:
            if 'buy_quantity' in intent:
                buy_amt = intent['buy_quantity']
                sell_amt = intent['sell_limit'] * sell_prices[intent['tkn_sell']]
            else:
                buy_amt = intent['buy_limit']
                sell_amt = intent['sell_quantity'] * sell_prices[intent['tkn_sell']]
            price_limit = sell_amt / buy_amt
            price_limits['max_tkn_buy_price'] = min(price_limits['max_tkn_buy_price'], price_limit)
        elif intent['tkn_buy'] == tkn_sell and intent['tkn_sell'] == tkn_buy:
            if 'buy_quantity' in intent:
                buy_amt = intent['buy_quantity']
                sell_amt = intent['sell_limit']
            else:
                buy_amt = intent['buy_limit']
                sell_amt = intent['sell_quantity']
            price_limit = sell_amt / buy_amt
            price_limits['max_tkn_buy_tkn_sell_price'] = min(price_limits['max_tkn_buy_tkn_sell_price'], price_limit)
        elif intent['tkn_buy'] == tkn_sell:
            if 'buy_quantity' in intent:
                buy_amt = intent['buy_quantity']
                sell_amt = intent['sell_limit'] * sell_prices[intent['tkn_sell']]
            else:
                buy_amt = intent['buy_limit']
                sell_amt = intent['sell_quantity'] * sell_prices[intent['tkn_sell']]
            price_limit = sell_amt / buy_amt
            price_limits['max_tkn_sell_price'] = min(price_limits['max_tkn_sell_price'], price_limit)
        elif intent['tkn_sell'] == tkn_buy:
            if 'buy_quantity' in intent:
                buy_amt = intent['buy_quantity'] * buy_prices[intent['tkn_buy']]
                sell_amt = intent['sell_limit']
            else:
                buy_amt = intent['buy_limit'] * buy_prices[intent['tkn_buy']]
                sell_amt = intent['sell_quantity']
            price_limit = buy_amt / sell_amt
            price_limits['min_tkn_buy_price'] = max(price_limits['min_tkn_buy_price'], price_limit)

    net_intents = []
    for k in net_intent_dict:
        for i in net_intent_dict[k]:
            net_intents.append(net_intent_dict[k][i])
    return net_intents, price_limits


def calculate_transfers(
        intents: list,  # aggregate intents
        buy_prices: dict,  # key: token, value: price
        sell_prices: dict,  # key: token, value: price
) -> (list, dict):
    # we only expect 3 tokens, tkn_buy, tkn_sell, and LRNA
    # intents should be fully processed since they represent net processed amounts
    transfers = []
    deltas = {tkn: {"in": 0, "out": 0} for tkn in buy_prices}
    deltas['LRNA'] = {"in": 0, "out": 0}
    buy_prices = {k: buy_prices[k] for k in buy_prices}
    sell_prices = {k: sell_prices[k] for k in sell_prices}
    buy_prices['LRNA'] = 1
    sell_prices['LRNA'] = 1
    for i in range(len(intents)):
        intent = intents[i]
        if intent['tkn_buy'] not in deltas:
            deltas[intent['tkn_buy']] = {"in": 0, "out": 0}
        if intent['tkn_sell'] not in deltas:
            deltas[intent['tkn_sell']] = {"in": 0, "out": 0}
        if 'sell_quantity' in intent:
            amt = intent['sell_quantity']
            buy_amt = amt * sell_prices[intent['tkn_sell']] / buy_prices[intent['tkn_buy']]
            transfer = {
                'buy_quantity': buy_amt,
                'sell_quantity': amt,
                'tkn_buy': intent['tkn_buy'],
                'tkn_sell': intent['tkn_sell']
            }
            transfers.append(transfer)
            deltas[intent['tkn_buy']]["out"] += buy_amt
            deltas[intent['tkn_sell']]["in"] += amt
        elif 'buy_quantity' in intent:
            amt = intent['buy_quantity']
            sell_amt = amt * buy_prices[intent['tkn_buy']] / sell_prices[intent['tkn_sell']]
            transfer = {
                'buy_quantity': amt,
                'sell_quantity': sell_amt,
                'tkn_buy': intent['tkn_buy'],
                'tkn_sell': intent['tkn_sell']
            }
            transfers.append(transfer)
            deltas[intent['tkn_buy']]["out"] += amt
            deltas[intent['tkn_sell']]["in"] += sell_amt

    return transfers, deltas


def calculate_solution_first_trade(
        omnipool: OmnipoolState,
        new_intent: dict
):
    omnipool_new = omnipool.copy()
    if 'sell_quantity' in new_intent:
        sell_amt = new_intent['sell_quantity']
        buy_amt = omnipool.calculate_buy_from_sell(new_intent['tkn_buy'], new_intent['tkn_sell'], sell_amt)
        if buy_amt >= new_intent['buy_limit']:
            omnipool_new.swap(new_intent['agent'].copy(), new_intent['tkn_buy'], new_intent['tkn_sell'],
                              sell_quantity=sell_amt)
            amt = sell_amt
        else:
            amt = 0
    elif 'buy_quantity' in new_intent:
        buy_amt = new_intent['buy_quantity']
        sell_amt = omnipool.calculate_sell_from_buy(new_intent['tkn_buy'], new_intent['tkn_sell'], buy_amt)
        if sell_amt <= new_intent['sell_limit']:
            omnipool_new.swap(new_intent['agent'].copy(), new_intent['tkn_buy'], new_intent['tkn_sell'],
                              buy_quantity=buy_amt)
            amt = buy_amt
        else:
            amt = 0
    else:
        raise Exception("Invalid intent")

    delta_tkn_buy = omnipool.liquidity[new_intent['tkn_buy']] - omnipool_new.liquidity[new_intent['tkn_buy']]
    delta_lrna_tkn_buy = omnipool.lrna[new_intent['tkn_buy']] - omnipool_new.lrna[new_intent['tkn_buy']]
    delta_tkn_sell = omnipool.liquidity[new_intent['tkn_sell']] - omnipool_new.liquidity[new_intent['tkn_sell']]
    delta_lrna_tkn_sell = omnipool.lrna[new_intent['tkn_sell']] - omnipool_new.lrna[new_intent['tkn_sell']]
    buy_prices = {new_intent['tkn_buy']: abs(delta_lrna_tkn_buy / delta_tkn_buy), new_intent['tkn_sell']: 0}
    sell_prices = {new_intent['tkn_sell']: abs(delta_lrna_tkn_sell / delta_tkn_sell), new_intent['tkn_buy']: 0}

    return [new_intent], [amt], buy_prices, sell_prices, omnipool_new


def add_nonintersecting_intent(
        omnipool: OmnipoolState,
        intents: list,
        amt_processed: list,  # list of amt processed for each intent
        buy_prices: dict,  # key: token, value: price
        sell_prices: dict,  # key: token, value: price
        omnipool_after_intents: OmnipoolState,
        new_intent: dict
):
    omnipool_new = omnipool_after_intents.copy()
    if 'sell_quantity' in new_intent:
        sell_amt = new_intent['sell_quantity']
        buy_amt = omnipool_new.calculate_buy_from_sell(new_intent['tkn_buy'], new_intent['tkn_sell'], sell_amt)
        if buy_amt >= new_intent['buy_limit']:
            omnipool_new.swap(new_intent['agent'].copy(), new_intent['tkn_buy'], new_intent['tkn_sell'],
                              sell_quantity=sell_amt)
            amt = sell_amt
        else:
            amt = 0
    elif 'buy_quantity' in new_intent:
        buy_amt = new_intent['buy_quantity']
        sell_amt = omnipool_new.calculate_sell_from_buy(new_intent['tkn_buy'], new_intent['tkn_sell'], buy_amt)
        if sell_amt <= new_intent['sell_limit']:
            omnipool_new.swap(new_intent['agent'].copy(), new_intent['tkn_buy'], new_intent['tkn_sell'],
                              buy_quantity=buy_amt)
            amt = buy_amt
        else:
            amt = 0
    else:
        raise Exception("Invalid intent")

    delta_tkn_buy = omnipool.liquidity[new_intent['tkn_buy']] - omnipool_new.liquidity[new_intent['tkn_buy']]
    delta_lrna_tkn_buy = omnipool.lrna[new_intent['tkn_buy']] - omnipool_new.lrna[new_intent['tkn_buy']]
    delta_tkn_sell = omnipool.liquidity[new_intent['tkn_sell']] - omnipool_new.liquidity[new_intent['tkn_sell']]
    delta_lrna_tkn_sell = omnipool.lrna[new_intent['tkn_sell']] - omnipool_new.lrna[new_intent['tkn_sell']]
    buy_prices_new = {k: v for k, v in buy_prices.items()}
    sell_prices_new = {k: v for k, v in sell_prices.items()}
    buy_prices_new[new_intent['tkn_buy']] = abs(delta_lrna_tkn_buy / delta_tkn_buy)
    buy_prices_new[new_intent['tkn_sell']] = 0
    sell_prices_new[new_intent['tkn_sell']] = abs(delta_lrna_tkn_sell / delta_tkn_sell)
    sell_prices_new[new_intent['tkn_buy']] = 0

    return intents + [new_intent], amt_processed + [amt], buy_prices_new, sell_prices_new, omnipool_new


def calculate_price_targets(
        deltas: dict,  # key: token, value: delta_change of token w.r.t. Omnipool
        lrna_deltas: dict,  # key: token, value: delta_change of LRNA w.r.t. Omnipool in tkn trade
        buy_prices: dict,  # key: token, value: price
        sell_prices: dict,  # key: token, value: price
        spot_prices: dict
):
    price_targets = {}
    new_sell_prices = {tkn: sell_prices[tkn] for tkn in sell_prices}
    new_buy_prices = {tkn: buy_prices[tkn] for tkn in buy_prices}
    for tkn, delta in deltas.items():  # calculate net of Omnipool leg and other leg in same direction
        price_targets[tkn] = {'buy': 0, 'sell': 0}
        if delta["out"] > delta["in"]:  # LRNA being sold to Omnipool for TKN
            agent_sell_amt = delta['in']
            omnipool_sell_amt = delta["out"] - delta['in']
            agent_lrna_amt = agent_sell_amt * sell_prices[tkn]
            omnipool_lrna_amt = lrna_deltas[tkn]
            net_price = (agent_lrna_amt + omnipool_lrna_amt) / (agent_sell_amt + omnipool_sell_amt)
            price_targets[tkn]['buy'] = net_price
            price_targets[tkn]['sell'] = spot_prices['sell'][tkn]
            new_buy_prices[tkn] = net_price
            new_sell_prices[tkn] = spot_prices['sell'][tkn]
            # diffs[tkn]['buy'] = (buy_prices[tkn] - net_price)/net_price
        elif delta["out"] < delta["in"]:  # LRNA being bought from Omnipool for TKN
            agent_buy_amt = delta['out']
            omnipool_buy_amt = delta['in'] - delta["out"]
            agent_lrna_amt = agent_buy_amt * buy_prices[tkn]
            omnipool_lrna_amt = -lrna_deltas[tkn]
            net_price = (agent_lrna_amt + omnipool_lrna_amt) / (agent_buy_amt + omnipool_buy_amt)
            price_targets[tkn]['sell'] = net_price
            price_targets[tkn]['buy'] = spot_prices['buy'][tkn]
            new_sell_prices[tkn] = net_price
            new_buy_prices[tkn] = spot_prices['buy'][tkn]
            # diffs[tkn]['sell'] = (sell_prices[tkn] - net_price)/net_price
        elif delta["out"] != 0:
            price_targets[tkn]['buy'] = spot_prices['buy'][tkn]
            price_targets[tkn]['sell'] = spot_prices['sell'][tkn]
            new_buy_prices[tkn] = spot_prices['buy'][tkn]
            new_sell_prices[tkn] = spot_prices['sell'][tkn]
            # diffs[tkn]['buy'] = (buy_prices[tkn] - spot_price) / spot_price
            # diffs[tkn]['sell'] = (sell_prices[tkn] - spot_price) / spot_price

    return price_targets, new_buy_prices, new_sell_prices


def calculate_price_targets2(
        deltas: dict,  # key: token, value: delta_change of token w.r.t. Omnipool
        lrna_deltas: dict,  # key: token, value: delta_change of LRNA w.r.t. Omnipool in tkn trade
        buy_prices: dict,  # key: token, value: price
        sell_prices: dict,  # key: token, value: price
        spot_prices: dict,
        net_intents: list,
        allocated_leftover: dict,
        tkn_buy: str,
        tkn_sell: str
):

    net_intent_dict = {
        'LRNA_' + tkn_buy: [0,0],
        tkn_buy + '_LRNA': [0,0],
        tkn_sell + '_' + tkn_buy: [0,0],
        tkn_buy + '_' + tkn_sell: [0,0],
        'LRNA_' + tkn_sell: [0,0],
        tkn_sell + '_LRNA': [0,0]
    }
    for intent in net_intents:
        ts, tb = intent['tkn_sell'], intent['tkn_buy']
        k = ts + "_" + tb
        is_buy = 1 if 'buy_quantity' in intent else 0
        amt = intent['buy_quantity'] if is_buy else intent['sell_quantity']
        net_intent_dict[k][is_buy] += amt

    # for tkn_buy
    remaining_lrna = allocated_leftover[tkn_buy]
    tkn_out = net_intent_dict[tkn_sell + "_" + tkn_buy][1] + net_intent_dict["LRNA_" + tkn_buy][1]
    tkn_in = net_intent_dict[tkn_buy + "_" + tkn_sell][0] + net_intent_dict[tkn_buy + "_LRNA"][0]
    lrna_out = net_intent_dict[tkn_buy + "_LRNA"][1] + net_intent_dict[tkn_buy + "_" + tkn_sell][1] * buy_prices[tkn_sell]
    lrna_in = net_intent_dict["LRNA_" + tkn_buy][0] + net_intent_dict[tkn_sell + "_" + tkn_buy][0] * sell_prices[tkn_sell]


    get_new_prices_for_tkn(allocated_leftover[tkn_buy], tkn_out, tkn_in, lrna_out, lrna_in, buy_prices[tkn_buy],
                           sell_prices[tkn_buy], lrna_deltas[tkn_buy], deltas[tkn_buy]["in"] - deltas[tkn_buy]["out"])

    # def get_new_prices_for_tkn(
    #         remaining_lrna: float,
    #         tkn_out: float,
    #         tkn_in: float,
    #         lrna_out: float,
    #         lrna_in: float,
    #         buy_price: float,
    #         sell_price: float,
    #         lrna_to_omnipool: float,  # negative value indicates LRNA is leaving Omnipool
    #         tkn_to_omnipool: float,  # negative value indicates tkn is leaving Omnipool
    # ):

    price_targets = {}
    new_sell_prices = {tkn: sell_prices[tkn] for tkn in sell_prices}
    new_buy_prices = {tkn: buy_prices[tkn] for tkn in buy_prices}
    for tkn, delta in deltas.items():  # calculate net of Omnipool leg and other leg in same direction
        price_targets[tkn] = {'buy': 0, 'sell': 0}
        if delta["out"] > delta["in"]:  # LRNA being sold to Omnipool for TKN
            agent_sell_amt = delta['in']
            omnipool_sell_amt = delta["out"] - delta['in']
            agent_lrna_amt = agent_sell_amt * sell_prices[tkn]
            omnipool_lrna_amt = lrna_deltas[tkn]
            net_price = (agent_lrna_amt + omnipool_lrna_amt) / (agent_sell_amt + omnipool_sell_amt)
            price_targets[tkn]['buy'] = net_price
            price_targets[tkn]['sell'] = spot_prices['sell'][tkn]
            new_buy_prices[tkn] = net_price
            new_sell_prices[tkn] = spot_prices['sell'][tkn]
            # diffs[tkn]['buy'] = (buy_prices[tkn] - net_price)/net_price
        elif delta["out"] < delta["in"]:  # LRNA being bought from Omnipool for TKN
            agent_buy_amt = delta['out']
            omnipool_buy_amt = delta['in'] - delta["out"]
            agent_lrna_amt = agent_buy_amt * buy_prices[tkn]
            omnipool_lrna_amt = -lrna_deltas[tkn]
            net_price = (agent_lrna_amt + omnipool_lrna_amt) / (agent_buy_amt + omnipool_buy_amt)
            price_targets[tkn]['sell'] = net_price
            price_targets[tkn]['buy'] = spot_prices['buy'][tkn]
            new_sell_prices[tkn] = net_price
            new_buy_prices[tkn] = spot_prices['buy'][tkn]
            # diffs[tkn]['sell'] = (sell_prices[tkn] - net_price)/net_price
        elif delta["out"] != 0:
            price_targets[tkn]['buy'] = spot_prices['buy'][tkn]
            price_targets[tkn]['sell'] = spot_prices['sell'][tkn]
            new_buy_prices[tkn] = spot_prices['buy'][tkn]
            new_sell_prices[tkn] = spot_prices['sell'][tkn]
            # diffs[tkn]['buy'] = (buy_prices[tkn] - spot_price) / spot_price
            # diffs[tkn]['sell'] = (sell_prices[tkn] - spot_price) / spot_price

    return price_targets, new_buy_prices, new_sell_prices


def is_remainder_nonnegative(pool_agent: Agent):
    for tkn, amt in pool_agent.holdings.items():
        if amt < 0:
            return False
    return True


def allocate_lrna_remainder(net_intents: list, buy_prices: dict, sell_prices: dict, tkn1: str, tkn2: str) -> dict:
    # calculate weights for allocating LRNA remainder, by allocating to assets with potential for adjustment
    # note that delta_Q = TKN_in * price_sell - TKN_out * price_buy
    # so having larger TKN_in, TKN_out amounts indicates larger ability to adjust LRNA
    # so we will allocate by relative values of TKN_in * price_sell + TKN_out * price_buy
    # tkn1
    tkn1_in, tkn1_out = 0, 0
    tkn2_in, tkn2_out = 0, 0
    for intent in net_intents:
        if 'buy_quantity' in intent:
            if intent['tkn_buy'] == tkn1:
                tkn1_out += intent['buy_quantity']
            elif intent['tkn_buy'] == tkn2:
                tkn2_out += intent['buy_quantity']
        elif 'sell_quantity' in intent:
            if intent['tkn_sell'] == tkn1:
                tkn1_in += intent['sell_quantity']
            elif intent['tkn_sell'] == tkn2:
                tkn2_in += intent['sell_quantity']
    tkn1_weight = tkn1_in * sell_prices[tkn1] + tkn1_out * buy_prices[tkn1]
    tkn2_weight = tkn2_in * sell_prices[tkn2] + tkn2_out * buy_prices[tkn2]
    total_weight = tkn1_weight + tkn2_weight
    weights = {tkn1: tkn1_weight / total_weight, tkn2: tkn2_weight / total_weight}
    return weights



def validate_solution_returning_price_targets(
        omnipool: OmnipoolState,
        intents: list,
        net_intents: list,  # swap desired to be processed
        amt_processed: list,  # list of amt processed for each intent
        buy_prices: dict,  # key: token, value: price
        sell_prices: dict,  # key: token, value: price
        tolerance: float = 0.01
):
    test_omnipool = omnipool.copy()
    test_intents = copy.deepcopy(intents)

    assert buy_prices.keys() == sell_prices.keys(), "buy_prices and sell_prices are not provided for same tokens"

    omnix.validate_intents(test_intents, amt_processed)
    transfers, deltas = omnix.calculate_transfers(test_intents, amt_processed, buy_prices, sell_prices)
    # net_intents = calculate_net_intents(test_intents, test_omnipool.liquidity.keys(), buy_prices, sell_prices)
    omnix.validate_transfer_amounts(transfers)
    pool_agent, lrna_deltas = omnix.execute_solution(test_omnipool, transfers, deltas)

    # if not is_remainder_nonnegative(pool_agent):
    #     raise Exception("agent has negative holdings")
    lrna_leftover = pool_agent.holdings["LRNA"]
    if 0 < lrna_leftover < 1:
        lrna_leftover = 0
    # adjust price based on leftover LRNA.
    # distribute lrna_leftover proportionally across deltas for tokens in new intent
    tkn_buy = intents[-1]['tkn_buy']
    tkn_sell = intents[-1]['tkn_sell']
    weights = allocate_lrna_remainder(net_intents, buy_prices, sell_prices, tkn_buy, tkn_sell)
    allocated_leftover = {tkn: lrna_leftover * weights[tkn] for tkn in weights}

    # # convert agent deltas to lrna
    # lrna_buy_in = deltas[tkn_buy]['in'] * sell_prices[tkn_buy]
    # lrna_buy_out = deltas[tkn_buy]['out'] * buy_prices[tkn_buy]
    # lrna_buy_matched = min(lrna_buy_in, lrna_buy_out)
    # lrna_sell_in = deltas[tkn_sell]['in'] * sell_prices[tkn_sell]
    # lrna_sell_out = deltas[tkn_sell]['out'] * buy_prices[tkn_sell]
    # lrna_sell_matched = min(lrna_sell_in, lrna_sell_out)
    # denom = lrna_buy_matched + lrna_sell_matched
    # if denom != 0:  # allocate leftover LRNA based on how value is matched in each asset
    #     lrna_buy_leftover = lrna_buy_matched / denom * lrna_leftover
    #     lrna_sell_leftover = lrna_sell_matched / denom * lrna_leftover
    #     # need to capture this additional LRNA from trade against Omnipool
    #     if lrna_buy_in > lrna_buy_out:
    #         lrna_buy_in += lrna_buy_leftover
    #     else:
    #         lrna_buy_out -= lrna_buy_leftover
    #     if lrna_sell_in > lrna_sell_out:
    #         lrna_sell_in += lrna_sell_leftover
    #     else:
    #         lrna_sell_out -= lrna_sell_leftover
    # # # distribute lrna_leftover across agent_lrna_deltas
    # # lrna_buy_in += lrna_buy_in / denom * lrna_leftover
    # # lrna_buy_out -= lrna_buy_out / denom * lrna_leftover
    # # lrna_sell_in += lrna_sell_in / denom * lrna_leftover
    # # lrna_sell_out -= lrna_sell_out / denom * lrna_leftover
    # # recompute buy and sell prices
    # new_sell_prices = copy.deepcopy(sell_prices)
    # new_buy_prices = copy.deepcopy(buy_prices)
    # new_sell_prices[tkn_buy] = lrna_buy_in / deltas[tkn_buy]['in'] if deltas[tkn_buy]['in'] > 0 else sell_prices[tkn_buy]
    # new_buy_prices[tkn_buy] = lrna_buy_out / deltas[tkn_buy]['out'] if deltas[tkn_buy]['out'] > 0 else buy_prices[tkn_buy]
    # new_sell_prices[tkn_sell] = lrna_sell_in / deltas[tkn_sell]['in'] if deltas[tkn_sell]['in'] > 0 else sell_prices[tkn_sell]
    # new_buy_prices[tkn_sell] = lrna_sell_out / deltas[tkn_sell]['out'] if deltas[tkn_sell]['out'] > 0 else buy_prices[tkn_sell]
    #
    # test_omnipool2 = omnipool.copy()
    #
    # # calculate new values
    # transfers2, deltas2 = omnix.calculate_transfers(test_intents, amt_processed, new_buy_prices, new_sell_prices)
    # # omnix.validate_transfer_amounts(transfers)
    # pool_agent2, lrna_deltas2 = omnix.execute_solution(test_omnipool2, transfers2, deltas2)
    spot_prices = {
        'buy': {tkn: test_omnipool.price(test_omnipool, tkn, "LRNA") for tkn in test_omnipool.liquidity},
        'sell': {tkn: test_omnipool.price(test_omnipool, tkn, "LRNA") for tkn in test_omnipool.liquidity}
    }

    price_targets, new_buy_prices, new_sell_prices = \
        calculate_price_targets2(deltas, lrna_deltas, buy_prices, sell_prices, spot_prices, net_intents,
                                 allocated_leftover, tkn_buy, tkn_sell)

    # update_intents(intents, transfers)

    return price_targets, pool_agent, new_buy_prices, new_sell_prices


def validate_solution_returning_price_targets2(
        omnipool: OmnipoolState,
        intents: list,  # swap desired to be processed
        amt_processed: list,  # list of amt processed for each intent
        buy_prices: dict,  # key: token, value: price
        sell_prices: dict,  # key: token, value: price
        tolerance: float = 0.01
):
    test_omnipool = omnipool.copy()
    test_intents = copy.deepcopy(intents)

    assert buy_prices.keys() == sell_prices.keys(), "buy_prices and sell_prices are not provided for same tokens"

    omnix.validate_intents(test_intents, amt_processed)
    transfers, deltas = omnix.calculate_transfers(test_intents, amt_processed, buy_prices, sell_prices)
    omnix.validate_transfer_amounts(transfers)
    pool_agent, lrna_deltas = omnix.execute_solution(test_omnipool, transfers, deltas)

    # if not is_remainder_nonnegative(pool_agent):
    #     raise Exception("agent has negative holdings")
    lrna_leftover = pool_agent.holdings["LRNA"]
    if 0 < lrna_leftover < 1:
        lrna_leftover = 0
    # adjust price based on leftover LRNA.
    # distribute lrna_leftover proportionally across deltas for tokens in new intent
    tkn_buy = intents[-1]['tkn_buy']
    tkn_sell = intents[-1]['tkn_sell']
    # convert agent deltas to lrna
    lrna_buy_in = deltas[tkn_buy]['in'] * sell_prices[tkn_buy]
    lrna_buy_out = deltas[tkn_buy]['out'] * buy_prices[tkn_buy]
    lrna_buy_matched = min(lrna_buy_in, lrna_buy_out)
    lrna_sell_in = deltas[tkn_sell]['in'] * sell_prices[tkn_sell]
    lrna_sell_out = deltas[tkn_sell]['out'] * buy_prices[tkn_sell]
    lrna_sell_matched = min(lrna_sell_in, lrna_sell_out)
    denom = lrna_buy_matched + lrna_sell_matched
    if denom != 0:  # allocate leftover LRNA based on how value is matched in each asset
        lrna_buy_leftover = lrna_buy_matched / denom * lrna_leftover
        lrna_sell_leftover = lrna_sell_matched / denom * lrna_leftover
        # need to capture this additional LRNA from trade against Omnipool
        if lrna_buy_in > lrna_buy_out:
            lrna_buy_in += lrna_buy_leftover
        else:
            lrna_buy_out -= lrna_buy_leftover
        if lrna_sell_in > lrna_sell_out:
            lrna_sell_in += lrna_sell_leftover
        else:
            lrna_sell_out -= lrna_sell_leftover
    # # distribute lrna_leftover across agent_lrna_deltas
    # lrna_buy_in += lrna_buy_in / denom * lrna_leftover
    # lrna_buy_out -= lrna_buy_out / denom * lrna_leftover
    # lrna_sell_in += lrna_sell_in / denom * lrna_leftover
    # lrna_sell_out -= lrna_sell_out / denom * lrna_leftover
    # recompute buy and sell prices
    new_sell_prices = copy.deepcopy(sell_prices)
    new_buy_prices = copy.deepcopy(buy_prices)
    new_sell_prices[tkn_buy] = lrna_buy_in / deltas[tkn_buy]['in'] if deltas[tkn_buy]['in'] > 0 else sell_prices[tkn_buy]
    new_buy_prices[tkn_buy] = lrna_buy_out / deltas[tkn_buy]['out'] if deltas[tkn_buy]['out'] > 0 else buy_prices[tkn_buy]
    new_sell_prices[tkn_sell] = lrna_sell_in / deltas[tkn_sell]['in'] if deltas[tkn_sell]['in'] > 0 else sell_prices[tkn_sell]
    new_buy_prices[tkn_sell] = lrna_sell_out / deltas[tkn_sell]['out'] if deltas[tkn_sell]['out'] > 0 else buy_prices[tkn_sell]

    test_omnipool2 = omnipool.copy()

    # calculate new values
    transfers2, deltas2 = omnix.calculate_transfers(test_intents, amt_processed, new_buy_prices, new_sell_prices)
    # omnix.validate_transfer_amounts(transfers)
    pool_agent2, lrna_deltas2 = omnix.execute_solution(test_omnipool2, transfers2, deltas2)
    spot_prices = {
        'buy': {tkn: test_omnipool.price(test_omnipool2, tkn, "LRNA") for tkn in test_omnipool2.liquidity},
        'sell': {tkn: test_omnipool.price(test_omnipool2, tkn, "LRNA") for tkn in test_omnipool2.liquidity}
    }

    price_targets, new_buy_prices2, new_sell_prices2 = calculate_price_targets(deltas2, lrna_deltas2, new_buy_prices, new_sell_prices, spot_prices)

    # update_intents(intents, transfers)

    return price_targets, pool_agent2, new_buy_prices2, new_sell_prices2


def get_new_prices_for_tkn(
        remaining_lrna: float,
        tkn_out: float,
        tkn_in: float,
        lrna_out: float,
        lrna_in: float,
        buy_price: float,
        sell_price: float,
        lrna_to_omnipool: float,  # negative value indicates LRNA is leaving Omnipool
        tkn_to_omnipool: float,  # negative value indicates tkn is leaving Omnipool
):
    if tkn_to_omnipool < 0:
        assert lrna_to_omnipool > 0, "Invalid input"
        a = tkn_in * (tkn_in - tkn_to_omnipool) / tkn_out - tkn_in
        b = (buy_price - (sell_price * tkn_in + remaining_lrna) / tkn_out) \
            * (tkn_in - tkn_to_omnipool) + tkn_in / tkn_out * lrna_out - (lrna_out + lrna_to_omnipool)
        c = (buy_price - (sell_price * tkn_in + remaining_lrna) / tkn_out) * lrna_out
        if a != 0:
            new_sell_price = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
        elif b != 0:
            new_sell_price = -c / b
        else:
            raise
        new_buy_price = buy_price + (tkn_in * (new_sell_price - sell_price) + remaining_lrna) / tkn_out
        return new_buy_price, new_sell_price


def add_intent_to_solution(
        omnipool: OmnipoolState,
        intents: list,
        amt_processed: list,  # list of amt processed for each intent
        buy_prices: dict,  # key: token, value: price
        sell_prices: dict,  # key: token, value: price
        omnipool_after_intents: OmnipoolState,
        new_intent: dict,
        validation_tolerance=0.0001,
        solution_tolerance=None
):
    # TODO cut down intents using amt_processed

    if not intents:  # new solution is to simply execute against Omnipool
        return calculate_solution_first_trade(omnipool, new_intent)

    if solution_tolerance is None:
        solution_tolerance = validation_tolerance / 2
    else:
        assert solution_tolerance <= validation_tolerance, "solution_tolerance must be less than or equal to validation_tolerance"

    net_intents, price_limits = calculate_net_intents(intents, new_intent['tkn_buy'], new_intent['tkn_sell'],
                                                      buy_prices, sell_prices)
    transfers, deltas = calculate_transfers(net_intents, buy_prices, sell_prices)

    # add new_intent to net_intents
    added = False
    for intent in net_intents:
        if intent['tkn_buy'] == new_intent['tkn_buy'] and intent['tkn_sell'] == new_intent['tkn_sell']:
            if 'buy_quantity' in new_intent and 'buy_quantity' in intent:
                intent['buy_quantity'] += new_intent['buy_quantity']
                added = True
                break
            elif 'sell_quantity' in new_intent and 'sell_quantity' in intent:
                intent['sell_quantity'] += new_intent['sell_quantity']
                added = True
                break
    if not added:
        new_net_intent = {'tkn_buy': new_intent['tkn_buy'], 'tkn_sell': new_intent['tkn_sell']}
        if 'buy_quantity' in new_intent:
            new_net_intent['buy_quantity'] = new_intent['buy_quantity']
        else:
            new_net_intent['sell_quantity'] = new_intent['sell_quantity']
        net_intents.append(new_net_intent)

    # come up with prices that are in appropriate bounds
    # apply new intent to omnipool_after_intents
    # new prices should be between original Omnipool spot, and execution price
    test_omnipool = omnipool_after_intents.copy()
    if 'buy_quantity' in new_intent:
        buy_amt = new_intent['buy_quantity']
        test_omnipool.swap(new_intent['agent'].copy(), new_intent['tkn_buy'], new_intent['tkn_sell'], buy_quantity=buy_amt)
        sell_amt = test_omnipool.liquidity[new_intent['tkn_sell']] - omnipool_after_intents.liquidity[new_intent['tkn_sell']]
        amt = buy_amt
    elif 'sell_quantity' in new_intent:
        sell_amt = new_intent['sell_quantity']
        test_omnipool.swap(new_intent['agent'].copy(), new_intent['tkn_buy'], new_intent['tkn_sell'], sell_quantity=sell_amt)
        buy_amt = omnipool_after_intents.liquidity[new_intent['tkn_buy']] - test_omnipool.liquidity[new_intent['tkn_buy']]
        amt = sell_amt
    else:
        raise Exception("Invalid intent")
    lrna_amt = test_omnipool.lrna[new_intent['tkn_buy']] - omnipool_after_intents.lrna[new_intent['tkn_buy']]
    buy_price_max = lrna_amt / buy_amt
    sell_price_min = lrna_amt / sell_amt
    buy_price_min = buy_prices[new_intent['tkn_buy']] if new_intent['tkn_buy'] in buy_prices else 0
    sell_price_max = sell_prices[new_intent['tkn_sell']] if new_intent['tkn_sell'] in sell_prices else float('inf')

    price_limits['min_tkn_buy_price'] = max(price_limits['min_tkn_buy_price'], buy_price_min)
    price_limits['max_tkn_buy_price'] = min(price_limits['max_tkn_buy_price'], buy_price_max)
    price_limits['min_tkn_sell_price'] = max(price_limits['min_tkn_sell_price'], sell_price_min)
    price_limits['max_tkn_sell_price'] = min(price_limits['max_tkn_sell_price'], sell_price_max)

    # start at price_limits['max_tkn_buy_price'] and price_limits['min_tkn_sell_price']

    test_intents = intents + [new_intent]
    test_amts = amt_processed + [amt]
    test_buy_prices = copy.deepcopy(buy_prices)
    test_sell_prices = copy.deepcopy(sell_prices)
    test_buy_prices[new_intent['tkn_buy']] = price_limits['max_tkn_buy_price']
    if new_intent['tkn_buy'] not in test_sell_prices:
        test_sell_prices[new_intent['tkn_buy']] = 0
    test_sell_prices[new_intent['tkn_sell']] = price_limits['min_tkn_sell_price']
    if new_intent['tkn_sell'] not in test_buy_prices:
        test_buy_prices[new_intent['tkn_sell']] = 0

    # calculate_prices_outcome(omnipool, test_intents, test_amts, test_buy_prices, test_sell_prices)
    for i in range(50):
        price_targets, remainder_agent, test_buy_prices, test_sell_prices = validate_solution_returning_price_targets(omnipool, test_intents, net_intents, test_amts, test_buy_prices, test_sell_prices, 0.0001)

        diff_buy_price_buy_token = price_targets[new_intent['tkn_buy']]['buy'] - test_buy_prices[new_intent['tkn_buy']]
        diff_sell_price_sell_token = price_targets[new_intent['tkn_sell']]['sell'] - test_sell_prices[new_intent['tkn_sell']]
        diff_sell_price_buy_token = price_targets[new_intent['tkn_buy']]['sell'] - test_sell_prices[new_intent['tkn_buy']]
        diff_buy_price_sell_token = price_targets[new_intent['tkn_sell']]['buy'] - test_buy_prices[new_intent['tkn_sell']]

        max_error = abs(diff_buy_price_buy_token) / price_targets[new_intent['tkn_buy']]['buy']
        max_error = max(max_error, abs(diff_sell_price_sell_token) / price_targets[new_intent['tkn_sell']]['sell'])
        max_error = max(max_error, abs(diff_sell_price_buy_token) / price_targets[new_intent['tkn_buy']]['sell'])
        max_error = max(max_error, abs(diff_buy_price_sell_token) / price_targets[new_intent['tkn_sell']]['buy'])
        error1 = abs(diff_buy_price_buy_token) / price_targets[new_intent['tkn_buy']]['buy']
        error2 = abs(diff_sell_price_sell_token) / price_targets[new_intent['tkn_sell']]['sell']
        error3 = abs(diff_sell_price_buy_token) / price_targets[new_intent['tkn_buy']]['sell']
        error4 = abs(diff_buy_price_sell_token) / price_targets[new_intent['tkn_sell']]['buy']
        if max_error < solution_tolerance and remainder_agent.holdings['LRNA'] >= 0:
            break
        # elif max_error >= solution_tolerance:

        omnix.validate_solution(omnipool, test_intents, test_amts, test_buy_prices, test_sell_prices,
                                validation_tolerance)
        return test_intents, test_amts, test_buy_prices, test_sell_prices, omnipool


def add_intent_to_solution2(
        omnipool: OmnipoolState,
        intents: list,
        amt_processed: list,  # list of amt processed for each intent
        buy_prices: dict,  # key: token, value: price
        sell_prices: dict,  # key: token, value: price
        omnipool_after_intents: OmnipoolState,
        new_intent: dict,
        validation_tolerance=0.0001,
        solution_tolerance=None
):
    # TODO cut down intents using amt_processed
    tkn_buy_seen, tkn_sell_seen = False, False
    for intent in intents:
        if new_intent['tkn_buy'] in [intent['tkn_buy'], intent['tkn_sell']]:
            tkn_buy_seen = True
        if new_intent['tkn_sell'] in [intent['tkn_buy'], intent['tkn_sell']]:
            tkn_sell_seen = True
    if not tkn_buy_seen and not tkn_sell_seen:  # new solution is to simply execute against Omnipool
        return calculate_solution_first_trade(omnipool, new_intent)

    if solution_tolerance is None:
        solution_tolerance = validation_tolerance / 2
    else:
        assert solution_tolerance <= validation_tolerance, "solution_tolerance must be less than or equal to validation_tolerance"

    net_intents, price_limits = calculate_net_intents(intents, new_intent['tkn_buy'], new_intent['tkn_sell'],
                                                      buy_prices, sell_prices)
    transfers, deltas = calculate_transfers(net_intents, buy_prices, sell_prices)

    # add new_intent to net_intents
    added = False
    for intent in net_intents:
        if intent['tkn_buy'] == new_intent['tkn_buy'] and intent['tkn_sell'] == new_intent['tkn_sell']:
            if 'buy_quantity' in new_intent and 'buy_quantity' in intent:
                intent['buy_quantity'] += new_intent['buy_quantity']
                added = True
                break
            elif 'sell_quantity' in new_intent and 'sell_quantity' in intent:
                intent['sell_quantity'] += new_intent['sell_quantity']
                added = True
                break
    if not added:
        new_net_intent = {'tkn_buy': new_intent['tkn_buy'], 'tkn_sell': new_intent['tkn_sell']}
        if 'buy_quantity' in new_intent:
            new_net_intent['buy_quantity'] = new_intent['buy_quantity']
        else:
            new_net_intent['sell_quantity'] = new_intent['sell_quantity']
        net_intents.append(new_net_intent)

    # come up with prices that are in appropriate bounds
    # apply new intent to omnipool_after_intents
    # new prices should be between original Omnipool spot, and execution price
    test_omnipool = omnipool_after_intents.copy()
    if 'buy_quantity' in new_intent:
        buy_amt = new_intent['buy_quantity']
        test_omnipool.swap(new_intent['agent'].copy(), new_intent['tkn_buy'], new_intent['tkn_sell'],
                           buy_quantity=buy_amt)
        sell_amt = test_omnipool.liquidity[new_intent['tkn_sell']] - omnipool_after_intents.liquidity[
            new_intent['tkn_sell']]
        amt = buy_amt
    elif 'sell_quantity' in new_intent:
        sell_amt = new_intent['sell_quantity']
        test_omnipool.swap(new_intent['agent'].copy(), new_intent['tkn_buy'], new_intent['tkn_sell'],
                           sell_quantity=sell_amt)
        buy_amt = omnipool_after_intents.liquidity[new_intent['tkn_buy']] - test_omnipool.liquidity[
            new_intent['tkn_buy']]
        amt = sell_amt
    else:
        raise Exception("Invalid intent")
    lrna_amt = test_omnipool.lrna[new_intent['tkn_buy']] - omnipool_after_intents.lrna[new_intent['tkn_buy']]
    buy_price_max = lrna_amt / buy_amt
    sell_price_min = lrna_amt / sell_amt
    buy_price_min = buy_prices[new_intent['tkn_buy']] if new_intent['tkn_buy'] in buy_prices else 0
    sell_price_max = sell_prices[new_intent['tkn_sell']] if new_intent['tkn_sell'] in sell_prices else float('inf')

    price_limits['min_tkn_buy_price'] = max(price_limits['min_tkn_buy_price'], buy_price_min)
    price_limits['max_tkn_buy_price'] = min(price_limits['max_tkn_buy_price'], buy_price_max)
    price_limits['min_tkn_sell_price'] = max(price_limits['min_tkn_sell_price'], sell_price_min)
    price_limits['max_tkn_sell_price'] = min(price_limits['max_tkn_sell_price'], sell_price_max)

    # start at price_limits['max_tkn_buy_price'] and price_limits['min_tkn_sell_price']

    test_intents = intents + [new_intent]
    test_amts = amt_processed + [amt]
    test_buy_prices = copy.deepcopy(buy_prices)
    test_sell_prices = copy.deepcopy(sell_prices)
    test_buy_prices[new_intent['tkn_buy']] = price_limits['max_tkn_buy_price']
    if new_intent['tkn_buy'] not in test_sell_prices:
        test_sell_prices[new_intent['tkn_buy']] = 0
    test_sell_prices[new_intent['tkn_sell']] = price_limits['min_tkn_sell_price']
    if new_intent['tkn_sell'] not in test_buy_prices:
        test_buy_prices[new_intent['tkn_sell']] = 0

    # calculate_prices_outcome(omnipool, test_intents, test_amts, test_buy_prices, test_sell_prices)
    for i in range(50):
        price_targets, remainder_agent, test_buy_prices, test_sell_prices = validate_solution_returning_price_targets(
            omnipool, test_intents, net_intents, test_amts, test_buy_prices, test_sell_prices, 0.0001)

        diff_buy_price_buy_token = price_targets[new_intent['tkn_buy']]['buy'] - test_buy_prices[
            new_intent['tkn_buy']]
        diff_sell_price_sell_token = price_targets[new_intent['tkn_sell']]['sell'] - test_sell_prices[
            new_intent['tkn_sell']]
        diff_sell_price_buy_token = price_targets[new_intent['tkn_buy']]['sell'] - test_sell_prices[
            new_intent['tkn_buy']]
        diff_buy_price_sell_token = price_targets[new_intent['tkn_sell']]['buy'] - test_buy_prices[
            new_intent['tkn_sell']]

        max_error = abs(diff_buy_price_buy_token) / price_targets[new_intent['tkn_buy']]['buy']
        max_error = max(max_error, abs(diff_sell_price_sell_token) / price_targets[new_intent['tkn_sell']]['sell'])
        max_error = max(max_error, abs(diff_sell_price_buy_token) / price_targets[new_intent['tkn_buy']]['sell'])
        max_error = max(max_error, abs(diff_buy_price_sell_token) / price_targets[new_intent['tkn_sell']]['buy'])
        error1 = abs(diff_buy_price_buy_token) / price_targets[new_intent['tkn_buy']]['buy']
        error2 = abs(diff_sell_price_sell_token) / price_targets[new_intent['tkn_sell']]['sell']
        error3 = abs(diff_sell_price_buy_token) / price_targets[new_intent['tkn_buy']]['sell']
        error4 = abs(diff_buy_price_sell_token) / price_targets[new_intent['tkn_sell']]['buy']
        if max_error < solution_tolerance and remainder_agent.holdings['LRNA'] >= 0:
            break
        # elif max_error >= solution_tolerance:


    omnix.validate_solution(omnipool, test_intents, test_amts, test_buy_prices, test_sell_prices, validation_tolerance)
    return test_intents, test_amts, test_buy_prices, test_sell_prices, omnipool


def calculate_solution_iteratively(
        omnipool: OmnipoolState,
        intents: list,
        validation_tolerance=0.0001,
        solution_tolerance=None
):
    if solution_tolerance is not None:
        assert solution_tolerance <= validation_tolerance, "solution_tolerance must be less than or equal to validation_tolerance"
    else:
        solution_tolerance = validation_tolerance / 2
    sorted_intents = get_sorted_intents_slippage(omnipool, intents)
    included_intents = []
    amts = []
    buy_prices = {}
    sell_prices = []
    omnipool_new = omnipool.copy()
    for intent in sorted_intents:
        # included_intents, amts, buy_prices, sell_prices, omnipool_new = add_intent_to_solution(omnipool, included_intents, amts, buy_prices, sell_prices, omnipool_new, intent, validation_tolerance, solution_tolerance)
        included_intents, amts, buy_prices, sell_prices, omnipool_new = (
            add_intent_to_solution2(omnipool, included_intents, amts, buy_prices, sell_prices, omnipool_new, intent,
                                    validation_tolerance, solution_tolerance))
    return included_intents, amts, buy_prices, sell_prices, omnipool_new


def calculate_prices_outcome(
        omnipool: OmnipoolState,
        intents: list,
        amounts: list,
        buy_prices: dict,
        sell_prices: dict
) -> dict:
    # calculate net intents
    # call calculate_transfers to identify net transfers
    # construct Omnipool trades from differences in transfers
    # collect assets in
    # trade assets in for LRNA
    # trade LRNA for assets out
    # distribute assets out
    # return pool agent
    pass
