import copy

from matplotlib import pyplot as plt
import sys, os
import streamlit as st

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.model.amm.agents import Agent
from hydradx.model.hollar import StabilityModule
from hydradx.apps.gigadot_modeling.display_utils import display_op_and_ss_multiple

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
        min_value=100_000, value=2_000_000, step=1, key="init_hollar_liq"
    )

with col3:
    amp = st.number_input(
        "A for stableswap pools",
        min_value=5, value=20, step=1, key="amp"
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

hsm_liquidity = {'aUSDT': 500_000, 'aUSDC': 500_000, 'sUSDS': 500_000 / pegs['sUSDS'], 'sUSDE': 500_000 / pegs['sUSDE']}
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
    st.session_state.num_pairs = 5  # default starting value

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

default_hollar_amounts = [1000000, 1000000, 1000000, 1000000, 1000000]
default_dump_blocks = [100, 500, 1000, 3000, 5000]
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

price_plot_n = 100


def model_hollar_dump(
        sell_amts,
        hollar_dump_blocks,
        init_stablepools,
        hsm_liquidity,
        buyback_speed,
        num_blocks,
        tkn_chart
):
    hsm_vals_dict = {}
    spot_prices_dict = {}
    hollar_sold_dict = {}
    for i in range(len(sell_amts)):
        sell_amt = sell_amts[i]
        num_blocks_dump = hollar_dump_blocks[i]
        stablepools = {tkn_chart: init_stablepools[tkn_chart].copy()}
        pools_list = [stablepools[tkn_chart]]
        # agent = Agent(enforce_holdings=False)
        arb_agent = Agent(enforce_holdings=False)
        reduced_liquidity = {tkn_chart: hsm_liquidity[tkn_chart]}
        hsm = StabilityModule(reduced_liquidity, buyback_speed, pools_list, max_buy_price_coef=0.999)
        spot_prices = [stablepools[tkn_chart].price('HOLLAR', tkn_chart)]
        init_hsm_value = sum([pegs[tkn] * hsm_liquidity[tkn] for tkn in pegs])
        hsm_values = [init_hsm_value]
        hollar_sell_amts = []
        for j in range(num_blocks):
            if (len(hsm_values) > num_blocks_dump and (hsm_values[-1] == hsm_values[-2]
                    or spot_prices[-1] >= hsm.max_buy_price_coef[tkn_chart])):  # note this only works because aUSDT peg is 1
                len_extend = min(num_blocks - j, 1000)
                hsm_values.extend([hsm_values[-1]] * len_extend)
                spot_prices.extend([spot_prices[-1]] * (len_extend // price_plot_n))
                # assert hollar_sell_amts[-1] == 0
                hollar_sell_amts.extend([0] * len_extend)
                break
            else:
                before_hsm_liq = hsm.liquidity[tkn_chart]
                for tkn, ss in stablepools.items():
                    max_buy_amt = hsm._get_max_buy_amount(tkn)  # note this ignores self.max_buy_price_coef
                    if tkn == tkn_chart:
                        hollar_sold = max_buy_amt  # track max_buy_amt for aUSDT
                    if j < num_blocks_dump:  # add in Hollar dumping to net swap
                        hollar_buy_amt = max_buy_amt - sell_amt / num_blocks_dump / 4
                    else:
                        hollar_buy_amt = max_buy_amt

                    # hsm.arb(arb_agent, tkn)

                    arb_agent.add(hsm.native_stable, max_buy_amt)  # flash mint Hollar for arb
                    hsm.swap(arb_agent, tkn_buy=tkn, tkn_sell=hsm.native_stable, sell_quantity=max_buy_amt)
                    if hollar_buy_amt > 0:
                        ss.swap(arb_agent, tkn_buy=hsm.native_stable, tkn_sell=tkn, buy_quantity=hollar_buy_amt)
                    elif hollar_buy_amt < 0:
                        ss.swap(arb_agent, tkn_buy=tkn, tkn_sell=hsm.native_stable, sell_quantity=-hollar_buy_amt)
                    arb_agent.remove(hsm.native_stable, max_buy_amt)  # burn Hollar that was minted

                    # if i < num_blocks_dump:
                    #     ss.swap(agent, 'HOLLAR', tkn,  sell_quantity=sell_amt / num_blocks_dump / 4)
                    ss.update()
                    hsm.update()
                after_hsm_liq = hsm.liquidity[tkn_chart]
                hsm_delta = after_hsm_liq - before_hsm_liq
                hsm_loss_total = hsm_delta * len(init_stablepools)
                if (j+1) % price_plot_n == 0:
                    spot_prices.append(stablepools[tkn_chart].price('HOLLAR', tkn_chart))
                hsm_values.append(hsm_values[-1] + hsm_loss_total)
                hollar_sell_amts.append(hollar_sold)
        hsm_vals_dict[(sell_amt, num_blocks_dump)] = hsm_values
        spot_prices_dict[(sell_amt, num_blocks_dump)] = spot_prices
        hollar_sold_dict[(sell_amt, num_blocks_dump)] = hollar_sell_amts
    return hsm_vals_dict, spot_prices_dict, hollar_sold_dict

if not scenario_added:
    hsm_vals_dict, spot_prices_dict, hollar_sold_dict = model_hollar_dump(
        hollar_amounts, hollar_dump_blocks, init_stablepools, hsm_liquidity, buyback_speed, num_blocks, 'aUSDT'
    )

    fig, ax = plt.subplots()
    for (sell_amt, num_blocks_dump), spot_prices in spot_prices_dict.items():
        # interpolate price
        interp_spot_prices = []
        for i in range(len(spot_prices) - 1):
            p = spot_prices[i]
            next_p = spot_prices[i+1]
            interped_prices = [p + (next_p - p) * j / price_plot_n for j in range(price_plot_n)]
            interp_spot_prices.extend(interped_prices)

        ax.plot(interp_spot_prices, label=f"{sell_amt} Hollar, {num_blocks_dump} blocks")
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
