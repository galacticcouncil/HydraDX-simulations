from hydradx.model.amm.fixed_price import FixedPriceExchange
from hydradx.model.amm.agents import Agent

def test_initialize_and_swap():
    agent = Agent(holdings={'USD': 5000})
    exchange = FixedPriceExchange(
        tokens={
            'USD': 1,
            'DOT': 5,
            'ETH': 2500
        },
        unique_id='fixed_price_exchange'
    ).swap(
        agent, tkn_sell='USD', tkn_buy='ETH', buy_quantity=2
    ).swap(
        agent, tkn_sell='ETH', tkn_buy='DOT', sell_quantity=1
    )
    if agent.holdings['USD'] != 0:
        raise AssertionError('Agent still has USD')
    elif agent.holdings['DOT'] != 500:
        raise AssertionError('Dot holdings incorrect.')
    elif agent.holdings['ETH'] != 1:
        raise AssertionError('ETH holdings incorrect.')


def test_price():
    exchange = FixedPriceExchange(
        tokens={
            'USD': 1,
            'DOT': 5,
            'ETH': 2500
        },
        unique_id='fixed_price_exchange'
    )
    if exchange.price('USD', 'ETH') != 0.0004:
        raise AssertionError('price incorrect')
    elif exchange.price('ETH', 'DOT') != 500:
        raise AssertionError('price incorrect')
    elif exchange.price('DOT', 'USD') != exchange.price('DOT'):
        raise AssertionError('price incorrect')


def test_liquidity_tracking():
    agent = Agent(holdings={'USD': 5000})
    exchange = FixedPriceExchange(
        tokens={
            'USD': 1,
            'DOT': 5,
            'ETH': 2500
        },
        unique_id='fixed_price_exchange'
    ).swap(
        agent, tkn_sell='USD', tkn_buy='ETH', buy_quantity=2
    ).swap(
        agent, tkn_sell='ETH', tkn_buy='DOT', sell_quantity=1
    )
    if exchange.liquidity['ETH'] != -1:
        raise AssertionError('ETH liquidity incorrect.')
    elif exchange.liquidity['USD'] != 5000:
        raise AssertionError('USD liquidity incorrect.')
    elif exchange.liquidity['DOT'] != -500:
        raise AssertionError('DOT liquidity incorrect.')