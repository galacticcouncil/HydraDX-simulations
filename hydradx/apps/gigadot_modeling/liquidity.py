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
# Add custom scenario inputs at the top of the sidebar:
st.sidebar.subheader("Custom Scenario Setup")

baseline_option = st.sidebar.selectbox(
    "Choose baseline scenario for custom scenario",
    ["Current Omnipool", "gigaDOT with 3 assets", "gigaDOT with 2 assets"],
    key="baseline_custom"
)

# Multiplier for the Omnipool DOT/vDOT amounts:
op_multiplier = st.sidebar.number_input(
    "DOT/vDOT in Omnipool multiplier",
    min_value=0.1, value=1.0, step=0.1, key="op_mult"
)

# Multiplier for the size of the gigaDOT pool:
pool_multiplier = st.sidebar.number_input(
    "gigaDOT pool size multiplier",
    min_value=0.1, value=1.0, step=0.1, key="pool_mult"
)



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

    custom_omnipool = OmnipoolState(tokens=new_tokens, lrna_fee=lrna_fee, asset_fee=asset_fee)

    if gigaDOT is not None:
        new_tokens = {tkn: amt * pool_mult for tkn, amt in gigaDOT.liquidity.items()}
        custom_gigaDOT = StableSwapPoolState(
            tokens=new_tokens, amplification=gigaDOT.amplification, trade_fee=gigaDOT.trade_fee, unique_id=gigaDOT.unique_id
        )
    else:
        custom_gigaDOT = None

    return custom_omnipool, custom_gigaDOT

st.sidebar.subheader("Liquidity distribution")

x_size, y_size = 10, 10


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
scenario_dict = {
    "Current Omnipool": (current_omnipool, None),
    "gigaDOT with 3 assets": (omnipool_minus_vDOT, gigadot_pool),
    "gigaDOT with 2 assets": (omnipool_minus_vDOT, gigadot2_pool)
}

omnipool_baseline, gigadot_baseline = scenario_dict[baseline_option]
custom_omnipool, custom_pool = create_custom_scenario(omnipool_baseline, gigadot_baseline, op_multiplier, pool_multiplier)
scenario_dict["Custom Scenario"] = (custom_omnipool, custom_pool)

st.sidebar.markdown("### Custom Scenario Pie Charts")
# For the custom scenario, if there is a custom pool (gigaDOT scenario), graph both the omnipool part and the pool part.
if op_multiplier != 1 or pool_multiplier != 1:
    if custom_pool is not None:
        display_op_and_ss(custom_omnipool.lrna, custom_pool.liquidity, usd_prices,
                          f"Custom Scenario: {baseline_option}", x_size, y_size)
    else:
        # Otherwise, just display the omnipool liquidity chart.
        fig_custom, ax_custom = plt.subplots(figsize=(10, 10))
        display_liquidity(ax_custom, custom_omnipool.lrna, f"Custom Scenario: {baseline_option}")
        st.sidebar.pyplot(fig_custom)
else:
    st.sidebar.write("*Change multipliers to something other than 1 to get a custom scenario.*")

st.sidebar.markdown("### Baseline Scenarios")

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
    agent.add(tkn_buy, quantity)
    agent.remove(tkn_sell, quantity)

# model

agent = Agent(enforce_holdings=False)
buy_sizes = [1, 10, 100, 1000, 10000]  # buying DOT with vDOT, DOT with USDT, vDOT with USDT
buy_sizes.sort()
sell_amts_omnipool = []

routes = {
    "Current Omnipool": {
        ('DOT', 'vDOT'): [{'tkn_sell': 'DOT', 'tkn_buy': 'vDOT', 'pool': "omnipool"}],
        ('2-Pool', 'DOT'): [{'tkn_sell': '2-Pool', 'tkn_buy': 'DOT', 'pool': "omnipool"}],
        ('2-Pool', 'vDOT'): [{'tkn_sell': '2-Pool', 'tkn_buy': 'vDOT', 'pool': "omnipool"}],
    },
    "gigaDOT with 3 assets": {
        ('DOT', 'vDOT'): [{'tkn_sell': 'DOT', 'tkn_buy': 'vDOT', 'pool': "gigaDOT"}],
        ('2-Pool', 'DOT'): [{'tkn_sell': '2-Pool', 'tkn_buy': 'DOT', 'pool': "omnipool"}],
        ('2-Pool', 'vDOT'): [
            {'tkn_sell': '2-Pool', 'tkn_buy': 'DOT', 'pool': "omnipool"},
            {'tkn_sell': 'DOT', 'tkn_buy': 'vDOT', 'pool': "gigaDOT"}
        ]
    },
    "gigaDOT with 2 assets": {
        ('DOT', 'vDOT'): [
            {'tkn_sell': 'DOT', 'tkn_buy': 'aDOT', 'pool': "money market"},
            {'tkn_sell': 'aDOT', 'tkn_buy': 'vDOT', 'pool': "gigaDOT"}
        ],
        ('2-Pool', 'DOT'): [{'tkn_sell': '2-Pool', 'tkn_buy': 'DOT', 'pool': "omnipool"}],
        ('2-Pool', 'vDOT'): [
            {'tkn_sell': '2-Pool', 'tkn_buy': 'DOT', 'pool': "omnipool"},
            {'tkn_sell': 'DOT', 'tkn_buy': 'aDOT', 'pool': "money market"},
            {'tkn_sell': 'aDOT', 'tkn_buy': 'vDOT', 'pool': "gigaDOT"}
        ]
    }
}

if op_multiplier != 1 or pool_multiplier != 1:
    routes["Custom Scenario"] = routes[baseline_option]

sell_amts_dicts = {}
for scenario in routes:
    sell_amts_dicts[scenario] = {}
    omnipool, gigaDOT = scenario_dict[scenario]
    for (tkn_sell, tkn_buy) in [('DOT', 'vDOT'), ('2-Pool', 'DOT'), ('2-Pool', 'vDOT')]:
        sell_amts = []
        for buy_size in buy_sizes:
            route = routes[scenario][(tkn_sell, tkn_buy)]
            assert tkn_sell == route[0]['tkn_sell']
            assert tkn_buy == route[-1]['tkn_buy']
            new_agent = agent.copy()
            trade_amt = buy_size
            for trade in reversed(route):
                if trade['pool'] == "omnipool":
                    new_state, new_agent = simulate_omnipool_swap(
                        omnipool, agent, tkn_buy=trade['tkn_buy'], tkn_sell=trade['tkn_sell'], buy_quantity=trade_amt
                    )
                    assert not new_state.fail
                elif trade['pool'] == "gigaDOT":
                    new_state, new_agent = simulate_stableswap_swap(
                        gigaDOT, agent, tkn_buy=trade['tkn_buy'], tkn_sell=trade['tkn_sell'], buy_quantity=trade_amt
                    )
                    assert not new_state.fail
                elif trade['pool'] == "money market":
                    money_market_swap(new_agent, trade['tkn_buy'], trade['tkn_sell'], trade_amt)
                else:
                    raise ValueError(f"Unknown pool type: {trade['pool']}")
                trade_amt = -new_agent.get_holdings(trade['tkn_sell'])
            sell_amts.append(trade_amt)
        sell_amts_dicts[scenario][(tkn_sell, tkn_buy)] = sell_amts


# graph slippage

slippage = {}
for scenario in routes:
    slippage[scenario] = {}
    for tkn_pair in [('DOT', 'vDOT'), ('2-Pool', 'DOT'), ('2-Pool', 'vDOT')]:
        sell_amts = sell_amts_dicts[scenario][tkn_pair]
        prices = [sell_amts[i] / buy_sizes[i] for i in range(len(buy_sizes))]
        lowest_price = prices[0]
        slippage[scenario][tkn_pair] = [(prices[i] - lowest_price) / lowest_price for i in range(len(buy_sizes))]

# Define markers for better visibility
markers = ['o', 's', '^', 'D', 'x']  # Circle, square, triangle, diamond, cross

# Mapping token pairs to marker indices
for idx, tkn_pair in enumerate([('DOT', 'vDOT'), ('2-Pool', 'vDOT'), ('2-Pool', 'DOT')]):
    st.subheader(f"Options for {tkn_pair} Slippage")

    options = ["Current", "GigaDOT as 3-pool", "GigaDOT as 2-pool", "Custom Scenario"]
    # The multiselect returns a list of selected series; by default, all are selected.
    selected_series = st.multiselect(f"Select series to display for {tkn_pair}", options, default=options, key=f"multiselect_{tkn_pair}")

    # Create the figure and axis for this graph
    fig, ax = plt.subplots(figsize=(10, 6))

    # Conditionally plot each series based on checkbox values
    if "Current" in selected_series:
        ax.plot(buy_sizes, slippage["Current Omnipool"][tkn_pair],
                marker=markers[0], linestyle='-', label='Current', linewidth=2)
    if "GigaDOT as 3-pool" in selected_series:
        ax.plot(buy_sizes, slippage["gigaDOT with 3 assets"][tkn_pair],
                marker=markers[1], linestyle='--', label='GigaDOT as 3-pool', linewidth=2)
    if "GigaDOT as 2-pool" in selected_series:
        ax.plot(buy_sizes, slippage["gigaDOT with 2 assets"][tkn_pair],
                marker=markers[4], linestyle='-', label='GigaDOT as 2-pool', linewidth=2)
    if "Custom Scenario" in selected_series and "Custom Scenario" in slippage:
        ax.plot(buy_sizes, slippage["Custom Scenario"][tkn_pair],
                marker=markers[2], linestyle='--', label='Custom', linewidth=2)

    # Add grid, legend, title, and axis labels
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    ax.set_title(f"{tkn_pair} Slippage")
    ax.set_xlabel('Buy size')
    ax.set_ylabel('Slippage')

    # Display the plot in Streamlit
    st.pyplot(fig)
