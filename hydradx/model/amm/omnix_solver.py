import copy

from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.omnix import validate_and_execute_solution


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


def _calculate_price_slippage_to_impact_sign_tuple(omnipool: OmnipoolState, intent: dict) -> tuple:
    x = calculate_price_slippage_to_impact(omnipool, intent)
    return (1, x) if x >= 0 else (-1, -x)


def get_sorted_intents(omnipool: OmnipoolState, intents: list) -> list:
    return sorted(intents, key=lambda intent: _calculate_price_slippage_to_impact_sign_tuple(omnipool, intent),
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


def construct_solution_old(init_omnipool: OmnipoolState, intents: list):
    init_sorted_intents = get_sorted_intents(init_omnipool, intents)

    # try to do swaps in order
    omnipool = init_omnipool.copy()
    sorted_intents = copy.deepcopy(init_sorted_intents)

    tkn_deltas, lrna_deltas, amounts = try_swaps(omnipool, sorted_intents)
    buy_prices, sell_prices = calculate_prices(omnipool, tkn_deltas, lrna_deltas)
    # check if prices satisfy all intents
    limit_violation_indices = []
    for i in range(len(sorted_intents)):
        intent = sorted_intents[i]
        exec_price = sell_prices[intent['tkn_sell']] / buy_prices[intent['tkn_buy']]
        if "sell_quantity" in intent:
            limit_price = intent['buy_limit'] / intent['sell_quantity']
            if exec_price < limit_price:
                limit_violation_indices.append(i)
        else:
            limit_price = intent['buy_quantity'] / intent['sell_limit']
            if exec_price < limit_price:
                limit_violation_indices.append(i)

    first_intent_skipped_index = len(init_sorted_intents)
    skipped_indices = []
    # if not, throw out worst intent and try again
    while first_intent_skipped_index > 0 and len(limit_violation_indices) > 0:
        first_intent_skipped_index -= 1
        skipped_indices.append(first_intent_skipped_index)

        omnipool = init_omnipool.copy()
        sorted_intents = copy.deepcopy(init_sorted_intents)

        tkn_deltas, lrna_deltas, amounts = try_swaps(omnipool, sorted_intents, skip_indices=skipped_indices)
        buy_prices, sell_prices = calculate_prices(omnipool, tkn_deltas, lrna_deltas)
        # check if prices satisfy all intents
        limit_violation_indices = []
        for i in range(len(sorted_intents)):
            intent = sorted_intents[i]
            exec_price = sell_prices[intent['tkn_sell']] / buy_prices[intent['tkn_buy']]
            if "sell_quantity" in intent:
                limit_price = intent['buy_limit'] / intent['sell_quantity']
                if exec_price < limit_price:
                    limit_violation_indices.append(i)
            else:
                limit_price = intent['buy_quantity'] / intent['sell_limit']
                if exec_price < limit_price:
                    limit_violation_indices.append(i)

    return buy_prices, sell_prices, amounts


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


def construct_solution(init_omnipool: OmnipoolState, intents: list):
    init_sorted_intents = get_sorted_intents(init_omnipool, intents)

    # try to do swaps in order
    omnipool = init_omnipool.copy()

    amounts = [0] * len(init_sorted_intents)
    buy_prices = {}
    sell_prices = {}
    asset_deltas = {}

    for i in range(len(init_sorted_intents)):
        intent = init_sorted_intents[i]
        # if neither asset in intent is present yet, solution should just try to execute against Omnipool
        if intent['tkn_buy'] not in asset_deltas and intent['tkn_sell'] not in asset_deltas:
            deltas = try_swap(omnipool, intent)
            amounts[i] = deltas['sell_tkn'] if 'sell_quantity' in intent else deltas['buy_tkn']
            buy_prices[intent['tkn_buy']] = deltas['buy_lrna'] / deltas['buy_tkn']
            sell_prices[intent['tkn_sell']] = deltas['sell_lrna'] / deltas['sell_tkn']
            asset_deltas[intent['tkn_buy']] = {
                'tkn_in': 0,
                'tkn_out': deltas['buy_tkn'],
                'lrna_in': deltas['buy_lrna'],
                'lrna_out': 0
            }
            asset_deltas[intent['tkn_sell']] = {
                'tkn_in': deltas['sell_tkn'],
                'tkn_out': 0,
                'lrna_in': 0,
                'lrna_out': deltas['sell_lrna']
            }
        else:
            # want to know most restrictive buy and sell limit prices for all assets
            # we assume that other prices will not change
            # price_limits = calculate_price_limits(i, init_sorted_intents, amounts)
            # now that we have the price limits, we want to know if we can include next intent while staying within the limits
            # assets break down into several categories
            # 1. assets that are not involved in any
            temp_asset_deltas = copy.deepcopy(asset_deltas)
            temp_omnipool = init_omnipool.copy()
            if 'sell_quantity' in intent:  # sell asset is specified
                pass
            else:  # buy asset is specified
                pass
            print('done')


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
    buy_prices = {new_intent['tkn_buy']: abs(delta_lrna_tkn_buy / delta_tkn_buy)}
    sell_prices = {new_intent['tkn_sell']: abs(delta_lrna_tkn_sell / delta_tkn_sell)}

    return [new_intent], [amt], buy_prices, sell_prices, omnipool_new


def add_intent_to_solution(
        omnipool: OmnipoolState,
        intents: list,
        amt_processed: list,  # list of amt processed for each intent
        buy_prices: dict,  # key: token, value: price
        sell_prices: dict,  # key: token, value: price
        omnipool_after_intents: OmnipoolState,
        new_intent: dict
):
    # TODO cut down intents using amt_processed

    if not intents:  # new solution is to simply execute against Omnipool
        return calculate_solution_first_trade(omnipool, new_intent)

    net_intents, price_limits = calculate_net_intents(intents, new_intent['tkn_buy'], new_intent['tkn_sell'],
                                                      buy_prices, sell_prices)
    transfers, deltas = calculate_transfers(net_intents, buy_prices, sell_prices)

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
    buy_price_min = buy_prices[new_intent['tkn_buy']]
    sell_price_max = sell_prices[new_intent['tkn_sell']]

    # start at buy_price_max and sell_price_min

    test_intents = intents + [new_intent]
    test_amts = amt_processed + [amt]
    test_buy_prices = copy.deepcopy(buy_prices)
    test_sell_prices = copy.deepcopy(sell_prices)
    test_buy_prices[new_intent['tkn_buy']] = buy_price_max
    if new_intent['tkn_buy'] not in test_sell_prices:
        test_sell_prices[new_intent['tkn_buy']] = 0
    test_sell_prices[new_intent['tkn_sell']] = sell_price_min
    if new_intent['tkn_sell'] not in test_buy_prices:
        test_buy_prices[new_intent['tkn_sell']] = 0

    # calculate_prices_outcome(omnipool, test_intents, test_amts, test_buy_prices, test_sell_prices)

    validate_and_execute_solution(omnipool, test_intents, test_amts, test_buy_prices, test_sell_prices, 0.0001)

    print('done')


def calculate_solution_iteratively(
        omnipool: OmnipoolState,
        intents: list
):
    included_intents = []
    amts = []
    buy_prices = {}
    sell_prices = []
    omnipool_new = omnipool.copy()
    for intent in intents:
        included_intents, amts, buy_prices, sell_prices, omnipool_new = add_intent_to_solution(omnipool, included_intents, amts, buy_prices, sell_prices, omnipool_new, intent)
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
