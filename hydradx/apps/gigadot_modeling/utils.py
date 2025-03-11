from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.agents import Agent

# Function to create a custom scenario from a chosen baseline.
# This function returns a tuple: (custom_omnipool, custom_pool)
def create_custom_scenario(omnipool, gigaDOT = None, op_mult = 1, pool_mult = 1):
    new_tokens = {
        tkn: {'liquidity': omnipool.liquidity[tkn], 'LRNA': omnipool.lrna[tkn]}
        for tkn in omnipool.liquidity
    }
    new_tokens['DOT']['liquidity'] *= op_mult
    new_tokens['DOT']['LRNA'] *= op_mult
    if 'vDOT' in new_tokens:
        new_tokens['vDOT']['liquidity'] *= op_mult
        new_tokens['vDOT']['LRNA'] *= op_mult

    custom_omnipool = OmnipoolState(tokens=new_tokens, lrna_fee=omnipool.last_lrna_fee, asset_fee=omnipool.last_fee)

    if gigaDOT is not None:
        new_tokens = {tkn: amt * pool_mult for tkn, amt in gigaDOT.liquidity.items()}
        custom_gigaDOT = StableSwapPoolState(
            tokens=new_tokens, amplification=gigaDOT.amplification, trade_fee=gigaDOT.trade_fee, unique_id=gigaDOT.unique_id
        )
    else:
        custom_gigaDOT = None

    return custom_omnipool, custom_gigaDOT


def get_omnipool_minus_vDOT(omnipool, op_dot_tvl_mult=1):
    omnipool_gigadot_liquidity = {tkn: value for tkn, value in omnipool.liquidity.items()}
    del omnipool_gigadot_liquidity['vDOT']
    omnipool_gigadot_lrna = {tkn: value for tkn, value in omnipool.lrna.items()}
    del omnipool_gigadot_lrna['vDOT']

    omnipool_gigadot_liquidity['DOT'] *= op_dot_tvl_mult
    omnipool_gigadot_lrna['DOT'] *= op_dot_tvl_mult

    tokens = {
        tkn: {'liquidity': omnipool_gigadot_liquidity[tkn], 'LRNA': omnipool_gigadot_lrna[tkn]}
        for tkn in omnipool_gigadot_liquidity
    }

    omnipool_gigadot = OmnipoolState(tokens=tokens, lrna_fee=omnipool.last_lrna_fee, asset_fee=omnipool.last_fee)
    return omnipool_gigadot

def set_up_gigaDOT_3pool(omnipool, amp: float, gigaDOT_tvl_mult=1):
    op_vDOT = omnipool.liquidity['vDOT']
    gigaDOT_dot = omnipool.liquidity['DOT'] * omnipool.lrna['vDOT'] / omnipool.lrna['DOT']
    gigadot_tokens = {
        'vDOT': op_vDOT / 3 * gigaDOT_tvl_mult,
        'DOT': gigaDOT_dot / 3 * gigaDOT_tvl_mult,
        'aDOT': gigaDOT_dot / 3 * gigaDOT_tvl_mult
    }
    vDOT_peg = gigadot_tokens['DOT'] / gigadot_tokens['vDOT']
    gigadot_pool = StableSwapPoolState(
        tokens=gigadot_tokens, amplification=amp, trade_fee=0.0002, unique_id='gigaDOT', peg=[1, vDOT_peg]
    )
    return gigadot_pool

def set_up_gigaDOT_2pool(omnipool, amp: float, gigaDOT_tvl_mult=1):
    op_vDOT = omnipool.liquidity['vDOT']
    gigaDOT_dot = omnipool.liquidity['DOT'] * omnipool.lrna['vDOT'] / omnipool.lrna['DOT']
    gigadot_tokens = {
        'aDOT': gigaDOT_dot / 2 * gigaDOT_tvl_mult,
        'vDOT': op_vDOT / 2 * gigaDOT_tvl_mult
    }
    vDOT_peg = gigadot_tokens['aDOT'] / gigadot_tokens['vDOT']
    gigadot_pool = StableSwapPoolState(
        tokens=gigadot_tokens, amplification=amp, trade_fee=0.0002, unique_id='gigaDOT', peg=vDOT_peg
    )
    return gigadot_pool

# dummy money market aDOT wrapper & unwrapper
def money_market_swap(agent, tkn_buy, tkn_sell, quantity):
    assert quantity > 0
    assert tkn_buy != tkn_sell
    assert tkn_buy in ['DOT', 'aDOT']
    assert tkn_sell in ['DOT', 'aDOT']
    if not agent.validate_holdings(tkn_sell, quantity):
        raise ValueError("Insufficient holdings.")
    agent.add(tkn_buy, quantity)
    agent.remove(tkn_sell, quantity)

def simulate_route(
        omnipool: OmnipoolState,
        stableswap: StableSwapPoolState,
        agent: Agent,
        buy_quantity: float,
        route: list
):
    trade_amt = buy_quantity
    new_omnipool = omnipool.copy()
    new_stableswap = stableswap.copy() if stableswap is not None else None
    new_agent = agent.copy()
    for trade in reversed(route):
        if trade['pool'] == "omnipool":
            new_omnipool.swap(new_agent, tkn_buy=trade['tkn_buy'], tkn_sell=trade['tkn_sell'], buy_quantity=trade_amt)
            if new_omnipool.fail:
                raise AssertionError("Swap failed in Omnipool.")
        elif trade['pool'] == "gigaDOT":
            new_stableswap.swap(new_agent, tkn_buy=trade['tkn_buy'], tkn_sell=trade['tkn_sell'], buy_quantity=trade_amt)
            if new_stableswap.fail:
                raise AssertionError("Swap failed in GigaDOT.")
        elif trade['pool'] == "money market":
            money_market_swap(new_agent, trade['tkn_buy'], trade['tkn_sell'], trade_amt)
        else:
            raise ValueError(f"Unknown pool type: {trade['pool']}")
        trade_amt = -new_agent.get_holdings(trade['tkn_sell'])
    return new_omnipool, new_stableswap, new_agent