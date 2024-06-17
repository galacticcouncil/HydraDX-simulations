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


def order_initial_intents(intents: list):
    # intents = [
    #     {'agent': agent_alice, 'buy_quantity': 10000, 'sell_limit': 81000, 'tkn_buy': 'DOT', 'tkn_sell': 'USDT'},
    #     {'agent': agent_bob, 'sell_quantity': 10000, 'buy_limit': 75500, 'tkn_buy': 'USDT', 'tkn_sell': 'DOT'}
    # ]

    pass
