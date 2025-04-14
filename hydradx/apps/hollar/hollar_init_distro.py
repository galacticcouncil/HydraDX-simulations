import copy
import math

from matplotlib import pyplot as plt
import sys, os
import streamlit as st

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.model.amm.agents import Agent
from hydradx.model.hollar import StabilityModule
from hydradx.apps.gigadot_modeling.utils import get_omnipool_minus_vDOT, set_up_gigaDOT_3pool, set_up_gigaDOT_2pool, \
    create_custom_scenario, simulate_route, get_slippage_dict
from hydradx.apps.gigadot_modeling.display_utils import display_liquidity, display_op_and_ss_multiple

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


# Create two columns in the sidebar
col1, col2, col3 = st.sidebar.columns(3)

# TVL of 2-pool in Omnipool
with col1:
    tvl_2pool = st.number_input(
        "TVL of 2-pool in Omnipool",
        min_value=1_000_000, value=15_000_000, step=1, key="tvl_2pool"
    )

# Multiplier for the size of the gigaDOT pool:
with col2:
    init_hollar_liq = st.number_input(
        "Initial Hollar liquidity",
        min_value=100_000, value=1_000_000, step=1, key="init_hollar_liq"
    )

with col3:
    amp = st.number_input(
        "A for stableswap pools",
        min_value=5, value=100, step=1, key="amp"
    )

# scale down omnipool tokens
mult_2pool = tvl_2pool / current_omnipool_liquidity['2-Pool']
tokens['2-Pool'] = {'liquidity': tvl_2pool, 'LRNA': lrna_amounts['2-Pool'] * mult_2pool}

# current pools

omnipool = OmnipoolState(tokens=tokens, lrna_fee=lrna_fee, asset_fee=asset_fee)
pegs = {'aUSDT': 1, 'aUSDC': 1, 'sUSDS': 1.05, 'sUSDE': 1.16}
init_stablepools = {}
for tkn in ['aUSDT', 'aUSDC', 'sUSDS', 'sUSDE']:
    stable_tokens = {tkn: init_hollar_liq / 4, 'HOLLAR': init_hollar_liq / 4}
    init_stablepools[tkn] = StableSwapPoolState(
        stable_tokens, amp, trade_fee=0.0002, unique_id=tkn, peg=pegs[tkn]
    )

ss_liquidity = [ss.liquidity for ss in init_stablepools.values()]

st.sidebar.subheader("Liquidity distribution")

x_size, y_size = 10, 10

fig = display_op_and_ss_multiple(omnipool.lrna, ss_liquidity, usd_prices, "Hollar distribution", x_size, y_size)
st.sidebar.pyplot(fig)

hsm_liquidity = {'aUSDT': 1_000_000, 'aUSDC': 1_000_000, 'sUSDS': 500_000 / pegs['sUSDS'], 'sUSDE': 500_000 / pegs['sUSDE']}
init_pools_list = [init_stablepools[tkn] for tkn in hsm_liquidity]

col1, col2 = st.columns(2)
with col1:
    buyback_speed = st.number_input(
        "Buyback speed",
        min_value=0.000001, value=0.0002, step=0.000001, key="buyback_speed", format="%.6f",
        help=("This number will be multiplied by the imbalance in the associated stableswap pool to determine how"
              " much Hollar can be bought back in one block. If a pool has 1,000,000 Hollar and 100_000 aUSDT, the"
              " imbalance is 450_000. That number would be multiplied by the buyback speed to determine how much Hollar"
              " can be bought back in one block.")
    )

with col2:
    num_blocks = st.number_input(
        "Number of blocks to simulate",
        min_value=1000, value=40_000, step=1, key="num_blocks",
        help="Simulation will end early if HSM has brought Hollar price back to peg"
    )

st.subheader("Scenarios")

if "num_pairs" not in st.session_state:
    st.session_state.num_pairs = 2  # default starting value

col1, col2 = st.columns(2)
if st.session_state.num_pairs < 10:
    with col1:
        scenario_added = st.button("Add scenario")
        if scenario_added:
            st.session_state.num_pairs += 1
if st.session_state.num_pairs > 1:
    with col2:
        scenario_removed = st.button("Remove last scenario")
        if scenario_removed:
            st.session_state.num_pairs -= 1

default_hollar_amounts = [2000000, 3000000]
default_dump_blocks = [10000, 10000]
n = st.session_state.num_pairs
default_hollar_amounts = default_hollar_amounts[:n] + [0] * max(0, n - len(default_hollar_amounts))
default_dump_blocks = default_dump_blocks[:n] + [0] * max(0, n - len(default_dump_blocks))
hollar_amounts_inputs = []
hollar_dump_blocks_inputs = []

with col1:
    st.text("Hollar Dumped")
with col2:
    st.text("Duration of Hollar dump (in blocks)")

for i in range(st.session_state.num_pairs):
    with col1:
        hollar_amounts_inputs.append(st.number_input(
            f"sell_amt_{i}", min_value=0, value=default_hollar_amounts[i], step=1, key=f"sell_amt_{i}",
            label_visibility='collapsed'
        ))
    with col2:
        hollar_dump_blocks_inputs.append(st.number_input(
            f"sell_blocks_{i}", min_value=0, value=default_dump_blocks[i], step=1, key=f"sell_blocks_{i}",
            label_visibility='collapsed'
        ))

hollar_amounts = []
hollar_dump_blocks = []
for i in range(len(hollar_amounts_inputs)):
    if hollar_amounts_inputs[i] > 0 and hollar_dump_blocks_inputs[i] > 0:
        hollar_amounts.append(hollar_amounts_inputs[i])
        hollar_dump_blocks.append(hollar_dump_blocks_inputs[i])


def model_hollar_dump(
        sell_amts,
        hollar_dump_blocks,
        init_stablepools,
        hsm_liquidity,
        buyback_speed,
        num_blocks
):
    hsm_vals_dict = {}
    spot_prices_dict = {}
    hollar_sold_dict = {}
    for i in range(len(sell_amts)):
        sell_amt = sell_amts[i]
        num_blocks_dump = hollar_dump_blocks[i]
        stablepools = copy.deepcopy(init_stablepools)
        pools_list = [stablepools[tkn] for tkn in hsm_liquidity]
        agent = Agent(enforce_holdings=False)
        arb_agent = Agent(enforce_holdings=False)
        hsm = StabilityModule(hsm_liquidity, buyback_speed, pools_list, max_buy_price_coef=0.999)
        spot_prices = [stablepools['aUSDT'].price('HOLLAR', 'aUSDT')]
        init_hsm_value = sum([pegs[tkn] * hsm.liquidity[tkn] for tkn in pegs])
        hsm_values = [init_hsm_value]
        hollar_sell_amts = []
        for i in range(num_blocks):
            if len(hsm_values) > num_blocks_dump and hsm_values[-1] == hsm_values[-2]:
                len_extend = min(num_blocks - i, 1000)
                hsm_values.extend([hsm_values[-1]] * len_extend)
                spot_prices.extend([spot_prices[-1]] * len_extend)
                assert hollar_sell_amts[-1] == 0
                hollar_sell_amts.extend([0] * len_extend)
                break
            else:
                before_hollar_amt = hsm.pools['aUSDT'].liquidity['HOLLAR']
                for tkn, ss in stablepools.items():
                    hsm.arb(arb_agent, tkn)
                    if tkn == 'aUSDT':
                        hollar_sold = before_hollar_amt - hsm.pools[tkn].liquidity['HOLLAR']
                    if i < num_blocks_dump:
                        ss.swap(agent, 'HOLLAR', tkn,  sell_quantity=sell_amt / num_blocks_dump / 4)
                    ss.update()
                    hsm.update()
                spot_prices.append(stablepools['aUSDT'].price('HOLLAR', 'aUSDT'))
                hsm_values.append(sum([pegs[tkn] * hsm.liquidity[tkn] for tkn in pegs]))
                hollar_sell_amts.append(hollar_sold)
        hsm_vals_dict[(sell_amt, num_blocks_dump)] = hsm_values
        spot_prices_dict[(sell_amt, num_blocks_dump)] = spot_prices
        hollar_sold_dict[(sell_amt, num_blocks_dump)] = hollar_sell_amts
    return hsm_vals_dict, spot_prices_dict, hollar_sold_dict

if not scenario_added:
    hsm_vals_dict, spot_prices_dict, hollar_sold_dict = model_hollar_dump(
        hollar_amounts, hollar_dump_blocks, init_stablepools, hsm_liquidity, buyback_speed, num_blocks
    )

    fig, ax = plt.subplots()
    for (sell_amt, num_blocks_dump), spot_prices in spot_prices_dict.items():
        ax.plot(spot_prices, label=f"{sell_amt} Hollar, {num_blocks_dump} blocks")
    ax.set_title("Spot price of Hollar (in aUSDT)")
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots()
    for (sell_amt, num_blocks_dump), hsm_values in hsm_vals_dict.items():
        ax.plot(hsm_values, label=f"{sell_amt} Hollar, {num_blocks_dump} blocks")
    ax.set_title("Value remaining in Stability Module")
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots()
    for (sell_amt, num_blocks_dump), hollar_sell_amts in hollar_sold_dict.items():
        ax.plot(hollar_sell_amts, label=f"{sell_amt} Hollar, {num_blocks_dump} blocks")
    ax.set_title("Amount of Hollar sold per block, for aUSDT")
    ax.legend()
    st.pyplot(fig)
