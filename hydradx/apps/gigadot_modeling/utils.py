import streamlit as st

from matplotlib import pyplot as plt

from hydradx.model.amm.stableswap_amm import StableSwapPoolState, simulate_buy_shares
from hydradx.model.amm.stableswap_amm import simulate_swap as simulate_stableswap_swap, simulate_withdraw_asset
from hydradx.model.amm.omnipool_amm import OmnipoolState, DynamicFee
from hydradx.model.amm.omnipool_amm import simulate_swap as simulate_omnipool_swap
from hydradx.model.amm.agents import Agent


# Custom function to display actual values instead of percentages
def actual_value_labels(pct, all_values):
    absolute = pct / 100.0 * sum(all_values)  # Convert % to actual value
    if absolute >= 1_000_000:  # If value is 1 million or more
        return f"${absolute / 1_000_000:.3g}m"
    elif absolute >= 1_000:  # If value is 1 thousand or more
        return f"${absolute / 1_000:.3g}k"
    else:  # If value is below 1,000
        return f"${absolute:.3g}"

def display_liquidity_usd(ax, usd_dict, title):
    labels = list(usd_dict.keys())
    sizes = list(usd_dict.values())

    ax.pie(sizes, labels=labels, autopct=lambda pct: actual_value_labels(pct, sizes), startangle=140)
    ax.set_title(title)

def display_liquidity(ax, lrna_dict, lrna_price, title):
    tvls = {tkn: lrna_dict[tkn] * lrna_price for tkn in lrna_dict}
    display_liquidity_usd(ax, tvls, title)

def display_op_and_ss(omnipool_lrna, ss_liquidity, prices, title, x_size, y_size):

    omnipool_tvl = sum(omnipool_lrna.values()) * prices['LRNA']
    stableswap_usd = {tkn: ss_liquidity[tkn] * prices[tkn] for tkn in ss_liquidity}
    stableswap_tvl = sum(stableswap_usd.values())
    scaling_factor = (stableswap_tvl / omnipool_tvl) ** 0.5
    ss_x_size, ss_y_size = x_size * scaling_factor, y_size * scaling_factor

    total_x_size, total_y_size = ss_x_size + x_size, ss_y_size + y_size
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(total_x_size, total_y_size))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    # fig.subplots_adjust(top=1.35)
    fig.subplots_adjust(hspace=-0.5)  # Adjust the spacing (lower values reduce the gap)
    display_liquidity(ax1, omnipool_lrna, prices['LRNA'], "Omnipool")
    display_liquidity_usd(ax2, stableswap_usd, "gigaDOT")
    ax2.set_position([ax2.get_position().x0, ax2.get_position().y0, ax2.get_position().width * scaling_factor, ax2.get_position().height * scaling_factor])

    st.sidebar.pyplot(fig)

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
    gigadot_pool = StableSwapPoolState(
        tokens=gigadot_tokens, amplification=amp, trade_fee=0.0002, unique_id='gigaDOT'
    )
    return gigadot_pool

def set_up_gigaDOT_2pool(omnipool, amp: float, gigaDOT_tvl_mult=1):
    op_vDOT = omnipool.liquidity['vDOT']
    gigaDOT_dot = omnipool.liquidity['DOT'] * omnipool.lrna['vDOT'] / omnipool.lrna['DOT']
    gigadot_tokens = {
        'vDOT': op_vDOT / 2 * gigaDOT_tvl_mult,
        'aDOT': gigaDOT_dot / 2 * gigaDOT_tvl_mult
    }
    gigadot_pool = StableSwapPoolState(
        tokens=gigadot_tokens, amplification=amp, trade_fee=0.0002, unique_id='gigaDOT'
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