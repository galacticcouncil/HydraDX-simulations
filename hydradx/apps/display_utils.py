from matplotlib import pyplot as plt

# Custom function to display actual values instead of percentages
def actual_value_labels(pct, all_values):
    absolute = pct / 100.0 * sum(all_values)  # Convert % to actual value
    if absolute >= 1_000_000:  # If value is 1 million or more
        return f"${absolute / 1_000_000:.3g}m"
    elif absolute >= 1_000:  # If value is 1 thousand or more
        return f"${absolute / 1_000:.3g}k"
    else:  # If value is below 1,000
        return f"${absolute:.3g}"

def display_liquidity_usd(ax, usd_dict, title = None):
    labels = list(usd_dict.keys())
    sizes = list(usd_dict.values())

    ax.pie(sizes, labels=labels, autopct=lambda pct: actual_value_labels(pct, sizes), startangle=140)
    if title is not None:
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

    return fig


def display_ss_multiple(ss_liquidity):

    ss_liquidity_d = {}
    if isinstance(ss_liquidity, list):
        for ss in ss_liquidity:
            tkn = next(key for key in ss if key != "HOLLAR")
            ss_liquidity_d[tkn] = {ss_tkn: ss[ss_tkn] for ss_tkn in ss}
            hollar_per_ss = ss['HOLLAR']
    else:
        tkn = next(key for key in ss_liquidity if key != "HOLLAR")
        ss_liquidity_d[tkn] = {ss_tkn: ss_liquidity[ss_tkn] for ss_tkn in ss_liquidity}
        hollar_per_ss = ss_liquidity['HOLLAR']

    num_pools = len(ss_liquidity_d)
    fig, axs = plt.subplots(num_pools, 1)
    fig.subplots_adjust(hspace=-0.1)  # Adjust the spacing (lower values reduce the gap)
    i = 0
    for tkn, ss in ss_liquidity_d.items():
        stableswap_usd = {tkn: hollar_per_ss, 'HOLLAR': hollar_per_ss}
        display_liquidity_usd(axs[i], stableswap_usd)
        i += 1

    return fig
