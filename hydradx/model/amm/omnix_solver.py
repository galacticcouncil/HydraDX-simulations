import copy

from hydradx.model.amm.omnipool_amm import OmnipoolState


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
        omnipool.swap(swap_intent['agent'], swap_intent['tkn_buy'], swap_intent['tkn_sell'],
                      sell_quantity=swap_intent['sell_quantity'])
    else:
        buy_amt = swap_intent['buy_quantity']
        sell_amt = omnipool.calculate_sell_from_buy(swap_intent['tkn_buy'], swap_intent['tkn_sell'],
                                                    swap_intent['buy_quantity'])
        if sell_amt > swap_intent['sell_limit']:
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


def construct_solution(init_omnipool: OmnipoolState, intents: list):
    init_sorted_intents = get_sorted_intents(init_omnipool, intents)

    # try to do swaps in order
    omnipool = init_omnipool.copy()
    sorted_intents = copy.deepcopy(init_sorted_intents)
    trade_amts = []
    # exec_trade_indices = []
    intent_deltas = []
    for i in range(len(sorted_intents)):
        deltas = try_swap(omnipool, sorted_intents[i])
        intent_deltas.append(deltas)
        # if amt > 0:
        #     exec_trade_indices.append(i)


