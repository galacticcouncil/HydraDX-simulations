import copy
import math

from matplotlib import pyplot as plt
import sys, os
import streamlit as st

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.agents import Agent
from hydradx.apps.gigadot_modeling.utils import get_omnipool_minus_vDOT, set_up_gigaDOT_3pool, set_up_gigaDOT_2pool, \
    create_custom_scenario, simulate_route, get_slippage_dict
from hydradx.apps.gigadot_modeling.display_utils import display_liquidity, display_op_and_ss

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
usd_prices['LRNA'] = lrna_price

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

# Create two columns in the sidebar
col1, col2 = st.sidebar.columns(2)

# Multiplier for the Omnipool DOT/vDOT amounts:
with col1:
    op_multiplier = st.number_input(
        "DOT/vDOT in Omnipool multiplier",
        min_value=0.1, value=1.0, step=0.1, key="op_mult"
    )

# Multiplier for the size of the gigaDOT pool:
with col2:
    pool_multiplier = st.number_input(
        "gigaDOT pool size multiplier",
        min_value=0.1, value=1.0, step=0.1, key="pool_mult"
    )

with col1:
    amp_2pool = st.number_input(
        "A for 2-Pool gigaDOT",
        min_value=5, value=150, step=5, key="amp_2pool"
    )

with col2:
    amp_3pool = st.number_input(
        "A for 3-Pool gigaDOT",
        min_value=5, value=320, step=5, key="amp_3pool"
    )

# current pools

current_omnipool = OmnipoolState(tokens=tokens, lrna_fee=lrna_fee, asset_fee=asset_fee)

omnipool_minus_vDOT = get_omnipool_minus_vDOT(current_omnipool)
gigadot_pool = set_up_gigaDOT_3pool(current_omnipool, amp_3pool)
gigadot2_pool = set_up_gigaDOT_2pool(current_omnipool, amp_2pool)
scenario_dict = {
    "Current Omnipool": (current_omnipool, None),
    "gigaDOT with 3 assets": (omnipool_minus_vDOT, gigadot_pool),
    "gigaDOT with 2 assets": (omnipool_minus_vDOT, gigadot2_pool)
}

st.sidebar.subheader("Liquidity distribution")

x_size, y_size = 10, 10

omnipool_baseline, gigadot_baseline = scenario_dict[baseline_option]
custom_omnipool, custom_pool = create_custom_scenario(omnipool_baseline, gigadot_baseline, op_multiplier, pool_multiplier)
scenario_dict["Custom Scenario"] = (custom_omnipool, custom_pool)

st.sidebar.markdown("### Custom Scenario Pie Charts")
# For the custom scenario, if there is a custom pool (gigaDOT scenario), graph both the omnipool part and the pool part.
if op_multiplier != 1 or pool_multiplier != 1:
    if custom_pool is not None:
        fig = display_op_and_ss(custom_omnipool.lrna, custom_pool.liquidity, usd_prices,
                          f"Custom Scenario: {baseline_option}", x_size, y_size)
        st.sidebar.pyplot(fig)
    else:
        # Otherwise, just display the omnipool liquidity chart.
        fig_custom, ax_custom = plt.subplots(figsize=(10, 10))
        display_liquidity(ax_custom, custom_omnipool.lrna, lrna_price, f"Custom Scenario: {baseline_option}")
        st.sidebar.pyplot(fig_custom)
else:
    st.sidebar.write("*Change multipliers to something other than 1 to get a custom scenario.*")

st.sidebar.markdown("### Baseline Scenarios")

fig = display_op_and_ss(omnipool_minus_vDOT.lrna, gigadot_pool.liquidity, usd_prices, "gigaDOT with 3 assets", x_size, y_size)
st.sidebar.pyplot(fig)
fig = display_op_and_ss(omnipool_minus_vDOT.lrna, gigadot2_pool.liquidity, usd_prices, "gigaDOT with 2 assets", x_size, y_size)
st.sidebar.pyplot(fig)
fig, ax = plt.subplots(figsize=(x_size, y_size))
display_liquidity(ax, lrna_amounts, lrna_price, "Current Omnipool")
st.sidebar.pyplot(fig)

# model

# Multiplier for the Omnipool DOT/vDOT amounts:
max_trade_size = st.number_input(
    "Max trade size, in bought token (DOT or vDOT)",
    min_value=2, value=10000, step=1, key="max_trade_size"
)
n = 10
min_trade_size = 1
step = (max_trade_size - min_trade_size) / (n - 1) if n > 1 else 0
buy_sizes = [min_trade_size + i * step for i in range(n)]

agent = Agent(enforce_holdings=False)
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
            try:
                _, _, new_agent = simulate_route(omnipool, gigaDOT, agent, buy_size, route)
            except AssertionError:
                break
            else:
                trade_amt = -new_agent.get_holdings(tkn_sell)
                sell_amts.append(trade_amt)
        sell_amts_dicts[scenario][(tkn_sell, tkn_buy)] = sell_amts


# graph slippage
slippage = get_slippage_dict(sell_amts_dicts, buy_sizes)

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
        n = len(slippage["Current Omnipool"][tkn_pair])
        ax.plot(buy_sizes[0:n], slippage["Current Omnipool"][tkn_pair],
                marker=markers[0], linestyle='-', label='Current', linewidth=2)
    if "GigaDOT as 3-pool" in selected_series:
        n = len(slippage["gigaDOT with 3 assets"][tkn_pair])
        ax.plot(buy_sizes[0:n], slippage["gigaDOT with 3 assets"][tkn_pair],
                marker=markers[1], linestyle='--', label='GigaDOT as 3-pool', linewidth=2)
    if "GigaDOT as 2-pool" in selected_series:
        n = len(slippage["gigaDOT with 2 assets"][tkn_pair])
        ax.plot(buy_sizes[0:n], slippage["gigaDOT with 2 assets"][tkn_pair],
                marker=markers[4], linestyle='-', label='GigaDOT as 2-pool', linewidth=2)
    if "Custom Scenario" in selected_series and "Custom Scenario" in slippage:
        n = len(slippage["Custom Scenario"][tkn_pair])
        ax.plot(buy_sizes[0:n], slippage["Custom Scenario"][tkn_pair],
                marker=markers[2], linestyle='--', label='Custom', linewidth=2)

    # Add grid, legend, title, and axis labels
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    ax.set_title(f"{tkn_pair} Slippage")
    ax.set_xlabel('Buy size')
    ax.set_ylabel('Slippage')

    # Display the plot in Streamlit
    st.pyplot(fig)
