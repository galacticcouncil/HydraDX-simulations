import copy

from matplotlib import pyplot as plt
import sys, os
import streamlit as st

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from hydradx.model.amm.stableswap_amm import StableSwapPoolState, simulate_buy_shares
from hydradx.model.amm.stableswap_amm import simulate_swap as simulate_stableswap_swap
from hydradx.model.amm.omnipool_amm import OmnipoolState, DynamicFee
from hydradx.model.amm.omnipool_amm import simulate_swap as simulate_omnipool_swap
from hydradx.model.amm.agents import Agent


# current Omnipool numbers

lrna_price = 25  # price of LRNA in USDT
current_omnipool_liquidity = {'HDX': 85_500_000, '2-Pool': 15_000_000,
                      'DOT': 2_717_000, 'vDOT': 905_800,
                      'WETH': 905.5, 'ASTR': 50_080_000,
                      'GLMR': 14_080_000, '4-Pool': 1_265_000,
                      'BNC': 5_972_000, 'tBTC': 10.79,
                      'CFG': 5_476_000, 'iBTC': 10.42,
                      'WBTC': 10.29, 'PHA': 5_105_000,
                      'KSM': 32_790, 'INTR': 67_330_000,
                      'vASTR': 7_165_000, 'KILT': 4_490_000,
                      'AAVE': 965.4, 'SOL': 754.4,
                      'ZTG': 8_229_000, 'CRU': 442_600,
                      '2-Pool-Btc': 0.6296, 'RING': 32_760_000}

usd_values = {'HDX': 1_000_000, '2-Pool': 15_000_000, 'DOT': 13_250_000,
              'vDOT': 6_475_000, 'WETH': 2_050_000, 'ASTR': 1_900_000,
              'GLMR': 1_660_000, '4-Pool': 1_290_000, 'BNC': 1_085_000,
              'tBTC': 900_000, 'CFG': 870_000, 'iBTC': 870_000,
              'WBTC': 860_000, 'PHA': 760_000, 'KSM': 640_000,
              'INTR': 370_000, 'vASTR': 320_000, 'KILT': 250_000,
              'AAVE': 190_000, 'SOL': 100_000, 'ZTG': 94_000,
              'CRU': 71_000, '2-Pool-Btc': 55_000, 'RING': 47_000}

usd_prices = {tkn: value / current_omnipool_liquidity[tkn] for tkn, value in usd_values.items()}
usd_prices['aDOT'] = usd_prices['DOT']

lrna_amounts = {key: value / lrna_price for key, value in usd_values.items()}

tokens = {
    tkn: {'liquidity': current_omnipool_liquidity[tkn], 'LRNA': lrna_amounts[tkn]}
    for tkn in current_omnipool_liquidity
}

lrna_fee = DynamicFee(
    minimum=0.0005, maximum=0.0010, amplification=1,  # TODO get these numbers right
    decay=0.000001
)
asset_fee = DynamicFee(
    minimum=0.0025, maximum=0.01, amplification=2,  # TODO get these numbers right
    decay=0.00001
)
tokens_2_pool = {'USDT': 9_000_000, 'USDC': 8_000_000}
tokens_4_pool = {
    'USDT': 362000, 'USDC2': 262000,
    'DAI': 368000, 'USDT2': 323000
}

current_ss_2_pool = StableSwapPoolState(
    tokens=tokens_2_pool, amplification=100, trade_fee=0.0002, unique_id='2-Pool'  # TODO get these numbers right
)
current_ss_4_pool = StableSwapPoolState(
    tokens=tokens_4_pool, amplification=320, trade_fee=0.0002, unique_id='4-Pool'
)

st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            min-width: 400px;
            max-width: 800px;
        }
    </style>
""", unsafe_allow_html=True)
st.sidebar.subheader("Liquidity distribution")

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

def display_liquidity(ax, lrna_dict, title):
    tvls = {tkn: lrna_dict[tkn] * lrna_price for tkn in lrna_dict}
    display_liquidity_usd(ax, tvls, title)


x_size, y_size = 10, 10
fig, ax = plt.subplots(figsize=(x_size, y_size))
display_liquidity(ax, lrna_amounts, "Current Omnipool")
st.sidebar.pyplot(fig)

# current pools

current_omnipool = OmnipoolState(
    tokens=tokens, lrna_fee=copy.deepcopy(lrna_fee), asset_fee=copy.deepcopy(asset_fee)
)


# vDOT, DOT moved to gigaDOT pool

usd_value_vdot = usd_values['vDOT']
usd_value_dot = usd_values['DOT']
tokens_gigadot_pool = {
    'vDOT': current_omnipool_liquidity['vDOT'] * 2/3,
    'DOT': current_omnipool_liquidity['DOT'] * usd_value_vdot / usd_value_dot * 2/3,
    'aDOT': current_omnipool_liquidity['DOT'] * usd_value_vdot / usd_value_dot * 2/3,
}

omnipool_gigadot_tokens = {tkn: value for tkn, value in current_omnipool_liquidity.items()}
omnipool_gigadot_tokens['DOT'] -= current_omnipool_liquidity['DOT'] * usd_value_vdot / usd_value_dot
del omnipool_gigadot_tokens['vDOT']

# scale LRNA by diff in tokens
gigadot_lrna_amounts = {tkn: omnipool_gigadot_tokens[tkn] / current_omnipool_liquidity[tkn] * lrna_amounts[tkn] for tkn in omnipool_gigadot_tokens}

tokens = {
    tkn: {'liquidity': omnipool_gigadot_tokens[tkn], 'LRNA': gigadot_lrna_amounts[tkn]}
    for tkn in omnipool_gigadot_tokens
}

gigadot_pool = StableSwapPoolState(
    tokens=tokens_gigadot_pool, amplification=100, trade_fee=0.0002, unique_id='gigaDOT'
)

omnipool_gigadot = OmnipoolState(
    tokens=tokens, lrna_fee=copy.deepcopy(lrna_fee), asset_fee=copy.deepcopy(asset_fee)
)


def display_op_and_ss(omnipool_lrna, ss_liquidity, prices, title):

    omnipool_tvl = sum(omnipool_lrna.values()) * lrna_price
    stableswap_usd = {tkn: ss_liquidity[tkn] * prices[tkn] for tkn in ss_liquidity}
    stableswap_tvl = sum(stableswap_usd.values())
    scaling_factor = (stableswap_tvl / omnipool_tvl) ** 0.5
    ss_x_size, ss_y_size = x_size * scaling_factor, y_size * scaling_factor


    total_x_size, total_y_size = ss_x_size + x_size, ss_y_size + y_size
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(total_x_size, total_y_size))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    # fig.subplots_adjust(top=1.35)
    fig.subplots_adjust(hspace=-0.4)  # Adjust the spacing (lower values reduce the gap)
    display_liquidity(ax1, omnipool_lrna, "Omnipool")
    display_liquidity_usd(ax2, stableswap_usd, "gigaDOT")
    ax2.set_position([ax2.get_position().x0, ax2.get_position().y0, ax2.get_position().width * scaling_factor, ax2.get_position().height * scaling_factor])

    st.sidebar.pyplot(fig)

display_op_and_ss(omnipool_gigadot.lrna, gigadot_pool.liquidity, usd_prices, "Removing DOT & vDOT")

# mini gigaDOT

usd_value_vdot = usd_values['vDOT']
usd_value_dot = usd_values['DOT']
tokens_minigigadot_pool = {
    'vDOT': current_omnipool_liquidity['vDOT'] * 1/3,
    'DOT': current_omnipool_liquidity['DOT'] * usd_value_vdot / usd_value_dot * 1/3,
    'aDOT': current_omnipool_liquidity['DOT'] * usd_value_vdot / usd_value_dot * 1/3,
}

omnipool_minigigadot_tokens = {tkn: value for tkn, value in current_omnipool_liquidity.items()}
del omnipool_minigigadot_tokens['vDOT']

# scale LRNA by diff in tokens
minigigadot_lrna_amounts = {tkn: omnipool_minigigadot_tokens[tkn] / current_omnipool_liquidity[tkn] * lrna_amounts[tkn] for tkn in omnipool_minigigadot_tokens}

tokens = {
    tkn: {'liquidity': omnipool_minigigadot_tokens[tkn], 'LRNA': minigigadot_lrna_amounts[tkn]}
    for tkn in omnipool_minigigadot_tokens
}

minigigadot_pool = StableSwapPoolState(
    tokens=tokens_minigigadot_pool, amplification=100, trade_fee=0.0002, unique_id='gigaDOT'
)

omnipool_minigigadot = OmnipoolState(
    tokens=tokens, lrna_fee=copy.deepcopy(lrna_fee), asset_fee=copy.deepcopy(asset_fee)
)

display_op_and_ss(omnipool_minigigadot.lrna, minigigadot_pool.liquidity, usd_prices,"Removing only vDOT")


# gigaDOT in Omnipool

gigadot_price = gigadot_pool.share_price('DOT') * usd_values['DOT'] / current_omnipool_liquidity['DOT']
usd_prices['gigaDOT'] = gigadot_price
tokens_op_with_gigadot = {tkn: value for tkn, value in omnipool_gigadot_tokens.items()}
tokens_op_with_gigadot['gigaDOT'] = gigadot_pool.shares
gigaDOT_lrna = gigadot_price * gigadot_pool.shares / lrna_price
lrna_op_with_gigadot = {tkn: value for tkn, value in gigadot_lrna_amounts.items()}
lrna_op_with_gigadot['gigaDOT'] = gigaDOT_lrna
tokens = {
    tkn: {'liquidity': tokens_op_with_gigadot[tkn], 'LRNA': lrna_op_with_gigadot[tkn]}
    for tkn in tokens_op_with_gigadot
}
omnipool_with_gigadot = OmnipoolState(
    tokens=tokens, lrna_fee=copy.deepcopy(lrna_fee), asset_fee=copy.deepcopy(asset_fee)
)

# display_liquidity(omnipool_with_gigadot.lrna, "Omnipool with gigaDOT", x_size, y_size)
display_op_and_ss(omnipool_with_gigadot.lrna, gigadot_pool.liquidity, usd_prices, "gigaDOT in Omnipool")


# gigaDOT as 2-pool
gigaDOT_tokens = {tkn: value for tkn, value in gigadot_pool.liquidity.items()}
usd_prices['aDOT'] = usd_prices['DOT']
gigaDOT_tvl = sum([gigaDOT_tokens[tkn] * usd_prices[tkn] for tkn in gigaDOT_tokens])
gigaDOT2_adot = gigaDOT_tvl / usd_prices['DOT'] / 2
gigaDOT2_vdot = gigaDOT_tvl / usd_prices['vDOT'] / 2
gigaDOT2_tokens = {'vDOT': gigaDOT2_vdot, 'aDOT': gigaDOT2_adot}
gigaDOT2_pool = StableSwapPoolState(
    tokens=gigaDOT2_tokens, amplification=100, trade_fee=0.0002, unique_id='gigaDOT'
)

display_op_and_ss(omnipool_gigadot.lrna, gigaDOT2_tokens, usd_prices, "gigaDOT with 2 assets")


# dummy money market aDOT wrapper & unwrapper
def money_market_swap(agent, tkn_buy, tkn_sell, quantity):
    assert quantity > 0
    assert tkn_buy != tkn_sell
    assert tkn_buy in ['DOT', 'aDOT']
    assert tkn_sell in ['DOT', 'aDOT']
    if not agent.validate_holdings(tkn_sell, quantity):
        raise ValueError("Insufficient holdings.")
    agent.transfer_to(tkn_buy, quantity)
    agent.transfer_from(tkn_sell, quantity)

# model

agent = Agent(enforce_holdings=False)
buy_sizes = [1, 10, 100, 1000, 10000]  # buying DOT with vDOT, DOT with USDT, vDOT with USDT
buy_sizes.sort()
sell_amts_omnipool = []
sell_amts_gigadot = []
sell_amts_minigigadot = []
sell_amts_gigadot_in_omnipool = []
sell_amts_gigadot2 = []
for buy_size in buy_sizes:
    sell_amts_omnipool_dict = {}
    sell_amts_gigadot_dict = {}
    sell_amts_gigadot_in_omnipool_dict = {}
    sell_amts_minigigadot_dict = {}
    sell_amts_gigadot2_dict = {}

    # DOT -> vDOT
    tkn_sell, tkn_buy = 'DOT', 'vDOT'
    # current Omnipool
    new_state, new_agent = simulate_omnipool_swap(
        current_omnipool, agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=buy_size
    )
    sell_amts_omnipool_dict[(tkn_sell, tkn_buy)] = -new_agent.get_holdings(tkn_sell)
    # gigadot
    new_state, new_agent = simulate_stableswap_swap(
        gigadot_pool, agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=buy_size
    )
    sell_amts_gigadot_dict[(tkn_sell, tkn_buy)] = -new_agent.get_holdings(tkn_sell)
    sell_amts_gigadot_in_omnipool_dict[(tkn_sell, tkn_buy)] = -new_agent.get_holdings(tkn_sell)
    # mini gigadot
    new_state, new_agent = simulate_stableswap_swap(
        minigigadot_pool, agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=buy_size
    )
    sell_amts_minigigadot_dict[(tkn_sell, tkn_buy)] = -new_agent.get_holdings(tkn_sell)
    # gigaDOT as 2-pool
    new_state, new_agent = simulate_stableswap_swap(
        gigaDOT2_pool, agent, tkn_buy=tkn_buy, tkn_sell='aDOT', buy_quantity=buy_size
    )
    money_market_swap(new_agent, 'aDOT', tkn_sell, -new_agent.get_holdings('aDOT'))
    sell_amts_gigadot2_dict[(tkn_sell, tkn_buy)] = -new_agent.get_holdings(tkn_sell)

    # DOT -> USDT
    tkn_sell, tkn_buy = 'DOT', '2-Pool'
    # current Omnipool
    new_state, new_agent = simulate_omnipool_swap(
        current_omnipool, agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=buy_size
    )
    sell_amts_omnipool_dict[(tkn_sell, tkn_buy)] = -new_agent.get_holdings(tkn_sell)
    # gigadot Omnipool
    new_state, new_agent = simulate_omnipool_swap(
        omnipool_gigadot, agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=buy_size
    )
    sell_amts_gigadot_dict[(tkn_sell, tkn_buy)] = -new_agent.get_holdings(tkn_sell)
    sell_amts_gigadot2_dict[(tkn_sell, tkn_buy)] = -new_agent.get_holdings(tkn_sell)  # gigaDOT as 2-pool
    # gigadot in Omnipool
    new_state, new_agent = simulate_omnipool_swap(
        omnipool_with_gigadot, agent, tkn_buy='2-Pool', tkn_sell='gigaDOT', buy_quantity=buy_size
    )
    withdraw_amt = -new_agent.get_holdings('gigaDOT')
    new_state, new_agent = simulate_buy_shares(
        gigadot_pool, agent, quantity=withdraw_amt, tkn_add='DOT'
    )
    sell_amts_gigadot_in_omnipool_dict[(tkn_sell, tkn_buy)] = -new_agent.get_holdings(tkn_sell)
    # mini gigadot
    new_state, new_agent = simulate_omnipool_swap(
        omnipool_minigigadot, agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=buy_size
    )
    sell_amts_minigigadot_dict[(tkn_sell, tkn_buy)] = -new_agent.get_holdings(tkn_sell)

    # vDOT -> USDT
    tkn_sell, tkn_buy = 'vDOT', '2-Pool'
    # current Omnipool
    new_state, new_agent = simulate_omnipool_swap(
        current_omnipool, agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=buy_size
    )
    sell_amts_omnipool_dict[(tkn_sell, tkn_buy)] = -new_agent.get_holdings(tkn_sell)
    # gigadot Omnipool, route through DOT
    new_state, new_agent = simulate_omnipool_swap(
        omnipool_gigadot, agent, tkn_buy=tkn_buy, tkn_sell='DOT', buy_quantity=buy_size
    )
    new_state, new_agent = simulate_stableswap_swap(
        gigadot_pool, agent, tkn_buy='DOT', tkn_sell=tkn_sell, buy_quantity=-new_agent.get_holdings('DOT')
    )
    sell_amts_gigadot_dict[(tkn_sell, tkn_buy)] = -new_agent.get_holdings(tkn_sell)
    # gigadot in Omnipool
    new_state, new_agent = simulate_omnipool_swap(
        omnipool_with_gigadot, agent, tkn_buy='2-Pool', tkn_sell='gigaDOT', buy_quantity=buy_size
    )
    withdraw_amt = -new_agent.get_holdings('gigaDOT')
    new_state, new_agent = simulate_buy_shares(
        gigadot_pool, agent, quantity=withdraw_amt, tkn_add='vDOT'
    )
    sell_amts_gigadot_in_omnipool_dict[(tkn_sell, tkn_buy)] = -new_agent.get_holdings(tkn_sell)
    # minigigadot
    new_state, new_agent = simulate_omnipool_swap(
        omnipool_minigigadot, agent, tkn_buy=tkn_buy, tkn_sell='DOT', buy_quantity=buy_size
    )
    new_state, new_agent = simulate_stableswap_swap(
        minigigadot_pool, agent, tkn_buy='DOT', tkn_sell=tkn_sell, buy_quantity=-new_agent.get_holdings('DOT')
    )
    sell_amts_minigigadot_dict[(tkn_sell, tkn_buy)] = -new_agent.get_holdings(tkn_sell)
    # gigaDOT as 2-pool
    new_state, new_agent = simulate_omnipool_swap(
        omnipool_gigadot, agent, tkn_buy=tkn_buy, tkn_sell='DOT', buy_quantity=buy_size
    )
    money_market_swap(new_agent, 'DOT', 'aDOT', buy_size)
    new_state, new_agent = simulate_stableswap_swap(
        gigaDOT2_pool, agent, tkn_buy='aDOT', tkn_sell=tkn_sell, buy_quantity=-new_agent.get_holdings('aDOT')
    )
    sell_amts_gigadot2_dict[(tkn_sell, tkn_buy)] = -new_agent.get_holdings(tkn_sell)

    sell_amts_omnipool.append(sell_amts_omnipool_dict)
    sell_amts_gigadot.append(sell_amts_gigadot_dict)
    sell_amts_gigadot_in_omnipool.append(sell_amts_gigadot_in_omnipool_dict)
    sell_amts_minigigadot.append(sell_amts_minigigadot_dict)
    sell_amts_gigadot2.append(sell_amts_gigadot2_dict)


# graph slippage

current_slippage = {}
gigadot_slippage = {}
gigadot_in_op_slippage = {}
minigigadot_slippage = {}
gigadot2_slippage = {}
for tkn_pair in [('DOT', 'vDOT'), ('DOT', '2-Pool'), ('vDOT', '2-Pool')]:
    current_prices = [sell_amts_omnipool[i][tkn_pair] / buy_sizes[i] for i in range(len(buy_sizes))]
    gigadot_prices = [sell_amts_gigadot[i][tkn_pair] / buy_sizes[i] for i in range(len(buy_sizes))]
    gigadot_in_op_prices = [sell_amts_gigadot_in_omnipool[i][tkn_pair] / buy_sizes[i] for i in range(len(buy_sizes))]
    minigigadot_prices = [sell_amts_minigigadot[i][tkn_pair] / buy_sizes[i] for i in range(len(buy_sizes))]
    gigadot2_prices = [sell_amts_gigadot2[i][tkn_pair] / buy_sizes[i] for i in range(len(buy_sizes))]
    lowest_current_price = current_prices[0]
    lowest_gigadot_price = gigadot_prices[0]
    lowest_gigadot_in_op_price = gigadot_in_op_prices[0]
    lowest_minigigadot_price = minigigadot_prices[0]
    lowest_gigadot2_price = gigadot2_prices[0]
    current_slippage[tkn_pair] = [(current_prices[i] - lowest_current_price) / lowest_current_price for i in range(len(buy_sizes))]
    gigadot_slippage[tkn_pair] = [(gigadot_prices[i] - lowest_gigadot_price) / lowest_gigadot_price for i in range(len(buy_sizes))]
    gigadot_in_op_slippage[tkn_pair] = [(gigadot_in_op_prices[i] - lowest_gigadot_in_op_price) / lowest_gigadot_in_op_price for i in range(len(buy_sizes))]
    minigigadot_slippage[tkn_pair] = [(minigigadot_prices[i] - lowest_minigigadot_price) / lowest_minigigadot_price for i in range(len(buy_sizes))]
    gigadot2_slippage[tkn_pair] = [(gigadot2_prices[i] - lowest_gigadot2_price) / lowest_gigadot2_price for i in range(len(buy_sizes))]

# Define markers for better visibility
markers = ['o', 's', '^', 'D', 'x']  # Circle, square, triangle, diamond, cross

# Mapping token pairs to marker indices
for idx, tkn_pair in enumerate([('DOT', 'vDOT'), ('DOT', '2-Pool'), ('vDOT', '2-Pool')]):
    fig, ax = plt.subplots(figsize=(10, 6))  # Create figure

    # Plot different slippage data with markers
    ax.plot(buy_sizes, current_slippage[tkn_pair], marker=markers[0], linestyle='-', label='Current', linewidth=2)
    ax.plot(buy_sizes, gigadot_slippage[tkn_pair], marker=markers[1], linestyle='--', label='GigaDOT', linewidth=2)
    ax.plot(buy_sizes, gigadot_in_op_slippage[tkn_pair], marker=markers[2], linestyle='-.', label='GigaDOT in Omnipool', linewidth=2)
    ax.plot(buy_sizes, minigigadot_slippage[tkn_pair], marker=markers[3], linestyle=':', label='Removing only vDOT', linewidth=2)
    ax.plot(buy_sizes, gigadot2_slippage[tkn_pair], marker=markers[4], linestyle='-', label='GigaDOT as 2-pool', linewidth=2)

    # Add grid, labels, legend, and title
    ax.grid(True, linestyle='--', alpha=0.6)  # Add light grid lines for better readability
    ax.legend()
    ax.set_title(f"{tkn_pair} Slippage")
    ax.set_xlabel('Buy size')
    ax.set_ylabel('Slippage')

    # Display the plot in Streamlit
    st.pyplot(fig)
