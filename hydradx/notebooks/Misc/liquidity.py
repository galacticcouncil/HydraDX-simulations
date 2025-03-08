import copy

from matplotlib import pyplot as plt
import sys, os
import streamlit as st

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from hydradx.model.amm.stableswap_amm import StableSwapPoolState, simulate_buy_shares
from hydradx.model.amm.stableswap_amm import simulate_swap as simulate_stableswap_swap, simulate_withdraw_asset
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

lrna_fee = 0.0005
asset_fee = 0.0015

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




# current pools

current_omnipool = OmnipoolState(tokens=tokens, lrna_fee=lrna_fee, asset_fee=asset_fee)

amp_2pool = 320
amp_3pool = amp_2pool * (4 / 27)

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

    omnipool_gigadot = OmnipoolState(tokens=tokens, lrna_fee=lrna_fee, asset_fee=asset_fee)
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

omnipool_minus_vDOT = get_omnipool_minus_vDOT(current_omnipool)
gigadot_pool = set_up_gigaDOT_3pool(current_omnipool, amp_3pool)
gigadot2_pool = set_up_gigaDOT_2pool(current_omnipool, amp_2pool)


def display_op_and_ss(omnipool_lrna, ss_liquidity, prices, title, x_size, y_size):

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
    fig.subplots_adjust(hspace=-0.5)  # Adjust the spacing (lower values reduce the gap)
    display_liquidity(ax1, omnipool_lrna, "Omnipool")
    display_liquidity_usd(ax2, stableswap_usd, "gigaDOT")
    ax2.set_position([ax2.get_position().x0, ax2.get_position().y0, ax2.get_position().width * scaling_factor, ax2.get_position().height * scaling_factor])

    st.sidebar.pyplot(fig)

x_size, y_size = 10, 10
display_op_and_ss(omnipool_minus_vDOT.lrna, gigadot_pool.liquidity, usd_prices, "gigaDOT with 3 assets", x_size, y_size)
display_op_and_ss(omnipool_minus_vDOT.lrna, gigadot2_pool.liquidity, usd_prices, "gigaDOT with 2 assets", x_size, y_size)
fig, ax = plt.subplots(figsize=(x_size, y_size))
display_liquidity(ax, lrna_amounts, "Current Omnipool")
st.sidebar.pyplot(fig)

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
# current Omnipool
for buy_size in buy_sizes:
    sell_amts_omnipool_dict = {}
    for (tkn_sell, tkn_buy) in [('DOT', 'vDOT'), ('2-Pool', 'DOT'), ('2-Pool', 'vDOT')]:
        new_state, new_agent = simulate_omnipool_swap(
            current_omnipool, agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=buy_size
        )
        sell_amts_omnipool_dict[(tkn_sell, tkn_buy)] = -new_agent.get_holdings(tkn_sell)
    sell_amts_omnipool.append(sell_amts_omnipool_dict)


sell_amts_gigadot = []
sell_amts_gigadot2 = []
for buy_size in buy_sizes:
    sell_amts_gigadot_dict = {}
    sell_amts_gigadot2_dict = {}

    # DOT -> vDOT
    tkn_sell, tkn_buy = 'DOT', 'vDOT'
    # gigadot
    new_state, new_agent = simulate_stableswap_swap(
        gigadot_pool, agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=buy_size
    )
    sell_amts_gigadot_dict[(tkn_sell, tkn_buy)] = -new_agent.get_holdings(tkn_sell)
    # gigaDOT as 2-pool
    new_state, new_agent = simulate_stableswap_swap(
        gigadot2_pool, agent, tkn_buy=tkn_buy, tkn_sell='aDOT', buy_quantity=buy_size
    )
    money_market_swap(new_agent, 'aDOT', tkn_sell, -new_agent.get_holdings('aDOT'))
    sell_amts_gigadot2_dict[(tkn_sell, tkn_buy)] = -new_agent.get_holdings(tkn_sell)

    # 2-Pool -> DOT
    tkn_sell, tkn_buy = '2-Pool', 'DOT'
    # gigadot Omnipool
    new_state, new_agent = simulate_omnipool_swap(
        omnipool_minus_vDOT, agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=buy_size
    )
    sell_amts_gigadot_dict[(tkn_sell, tkn_buy)] = -new_agent.get_holdings(tkn_sell)
    sell_amts_gigadot2_dict[(tkn_sell, tkn_buy)] = -new_agent.get_holdings(tkn_sell)  # gigaDOT as 2-pool

    # 2-Pool -> vDOT
    tkn_sell, tkn_buy = '2-Pool', 'vDOT'
    # gigadot Omnipool, route through DOT
    new_state, new_agent = simulate_stableswap_swap(  # buy vDOT with DOT
        gigadot_pool, agent, tkn_buy=tkn_buy, tkn_sell='DOT', buy_quantity=buy_size
    )
    new_state, new_agent = simulate_omnipool_swap(  # buy DOT with 2-Pool
        omnipool_minus_vDOT, agent, tkn_buy='DOT', tkn_sell=tkn_sell, buy_quantity=-new_agent.get_holdings('DOT')
    )
    sell_amts_gigadot_dict[(tkn_sell, tkn_buy)] = -new_agent.get_holdings(tkn_sell)
    # gigaDOT as 2-pool
    new_state, new_agent = simulate_stableswap_swap(  # buy vDOT with aDOT
        gigadot2_pool, agent, tkn_buy=tkn_buy, tkn_sell='aDOT', buy_quantity=buy_size
    )
    money_market_swap(new_agent, 'aDOT', 'DOT', -new_agent.get_holdings('aDOT'))  # buy aDOT with DOT
    new_state, new_agent = simulate_omnipool_swap(  # buy DOT with 2-Pool
        omnipool_minus_vDOT, agent, tkn_buy='DOT', tkn_sell=tkn_sell, buy_quantity=-new_agent.get_holdings('DOT')
    )
    sell_amts_gigadot2_dict[(tkn_sell, tkn_buy)] = -new_agent.get_holdings(tkn_sell)

    sell_amts_gigadot.append(sell_amts_gigadot_dict)
    sell_amts_gigadot2.append(sell_amts_gigadot2_dict)


# graph slippage

current_slippage = {}
gigadot_slippage = {}
gigadot2_slippage = {}
for tkn_pair in [('DOT', 'vDOT'), ('2-Pool', 'DOT'), ('2-Pool', 'vDOT')]:
    current_prices = [sell_amts_omnipool[i][tkn_pair] / buy_sizes[i] for i in range(len(buy_sizes))]
    gigadot_prices = [sell_amts_gigadot[i][tkn_pair] / buy_sizes[i] for i in range(len(buy_sizes))]
    gigadot2_prices = [sell_amts_gigadot2[i][tkn_pair] / buy_sizes[i] for i in range(len(buy_sizes))]
    assert min(gigadot2_prices) >= 0
    lowest_current_price = current_prices[0]
    lowest_gigadot_price = gigadot_prices[0]
    lowest_gigadot2_price = gigadot2_prices[0]
    current_slippage[tkn_pair] = [(current_prices[i] - lowest_current_price) / lowest_current_price for i in range(len(buy_sizes))]
    gigadot_slippage[tkn_pair] = [(gigadot_prices[i] - lowest_gigadot_price) / lowest_gigadot_price for i in range(len(buy_sizes))]
    gigadot2_slippage[tkn_pair] = [(gigadot2_prices[i] - lowest_gigadot2_price) / lowest_gigadot2_price for i in range(len(buy_sizes))]

# Define markers for better visibility
markers = ['o', 's', '^', 'D', 'x']  # Circle, square, triangle, diamond, cross

# Mapping token pairs to marker indices
for idx, tkn_pair in enumerate([('DOT', 'vDOT'), ('2-Pool', 'DOT'), ('2-Pool', 'vDOT')]):
    st.subheader(f"Options for {tkn_pair} Slippage")

    options = ["Current", "GigaDOT as 3-pool", "GigaDOT as 2-pool"]
    # The multiselect returns a list of selected series; by default, all are selected.
    selected_series = st.multiselect(f"Select series to display for {tkn_pair}", options, default=options, key=f"multiselect_{tkn_pair}")

    # Create the figure and axis for this graph
    fig, ax = plt.subplots(figsize=(10, 6))

    # Conditionally plot each series based on checkbox values
    if "Current" in selected_series:
        ax.plot(buy_sizes, current_slippage[tkn_pair],
                marker=markers[0], linestyle='-', label='Current', linewidth=2)
    if "GigaDOT as 3-pool" in selected_series:
        ax.plot(buy_sizes, gigadot_slippage[tkn_pair],
                marker=markers[1], linestyle='--', label='GigaDOT', linewidth=2)
    if "GigaDOT as 2-pool" in selected_series:
        ax.plot(buy_sizes, gigadot2_slippage[tkn_pair],
                marker=markers[4], linestyle='-', label='GigaDOT as 2-pool', linewidth=2)

    # Add grid, legend, title, and axis labels
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    ax.set_title(f"{tkn_pair} Slippage")
    ax.set_xlabel('Buy size')
    ax.set_ylabel('Slippage')

    # Display the plot in Streamlit
    st.pyplot(fig)
