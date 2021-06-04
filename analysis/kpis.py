# Create KPI metrics for:
# Slippage
# Price elasticity of outgoing transactions size
# Impermanent Loss
# Metrics are running time series

import pandas as pd
import numpy as np
import statsmodels.api as sm

def slippage(liquidity_information, assets, rdf, market):
    """Compute the slippage every timestep for a single experiment run summarized in
    the 'rdf' dataframe, with trading asset pair in 'assets', for the AMM specified by
    'market'.
    """
    
    market_name = market['name']
    market_fee = market['fee']

    time_liquidity_in = liquidity_information['time_entered'] 
    time_liquidity_out = liquidity_information['time_exited']
    
    res_in = []
    res_out = []
    trans_in = []
    trans_out = []
    p_out_for_in = []
    
    # If `asset_random_choice` and `trade_random_size` have 'np.nan' as a value anywhere in the timestep
    # then there were no trades/swaps in the experiment; return empty values
    if not rdf.asset_random_choice[1:].any() or not rdf.trade_random_size[1:].any():
        return {
            'error' : 'No trades/swaps in this experiment, no associated slippage calculations.'
        }
    
    # Assign ids for risk assets IN and OUT for Hydra
    asset_types = np.append(rdf.asset_random_choice.unique(), 'all')
    asset_in_ids = rdf.asset_random_choice
    asset_out_ids = pd.Series([x for y in asset_in_ids for x in assets if x != y  ])
    
    slippage = { asset : [] for asset in asset_types }
    elasticity = { asset : [] for asset in asset_types }

    if market_name == 'hydra': 
        
        for t in range(0,len(asset_in_ids)-1):
            
            reserve_asset_in_prev = rdf.pool[t].pool[asset_in_ids[t]]['R']
            reserve_asset_in = rdf.pool[t+1].pool[asset_in_ids[t]]['R']
            reserve_asset_out = rdf.pool[t+1].pool[asset_out_ids[t]]['R']
            reserve_asset_out_prev = rdf.pool[t].pool[asset_out_ids[t]]['R']
            price_asset_in = rdf.pool[t+1].pool[asset_in_ids[t]]['P']
            price_asset_in_prev = rdf.pool[t].pool[asset_in_ids[t]]['P']
            price_asset_out = rdf.pool[t+1].pool[asset_out_ids[t]]['P']
            price_asset_out_prev = rdf.pool[t].pool[asset_out_ids[t]]['P']
            
            weight_asset_in_prev = rdf.pool[t].pool[asset_in_ids[t]]['W']
            weight_asset_out_prev = rdf.pool[t].pool[asset_out_ids[t]]['W']
            
            # note that transactions must be net of fee paid, as fee is lost to
            # trader
            transactions_in = rdf.trade_random_size[t] * ( 1 + market_fee)
            transactions_out = -(reserve_asset_out - reserve_asset_out_prev)
            
            price_out_for_in = price_asset_out / price_asset_in
            price_out_for_in_prev = price_asset_out_prev / price_asset_in_prev
            
            
            
            # If transactions_out or transactions_in is zero, was not a swap/trade event; return n.a.
            if transactions_out == 0 or transactions_in == 0 or t == time_liquidity_in-1 or t == time_liquidity_out-1:
                for asset in asset_types:
                    if asset_in_ids[t] == asset or asset == 'all':
                        elasticity[asset].append(np.nan) 
                        slippage[asset].append(np.nan)
            else:
                # Compute percent change in reserve
                reserve_in_pct_change = transactions_in / reserve_asset_in_prev
                reserve_out_pct_change = transactions_out / reserve_asset_out_prev

                # Compute percent change in price (OUT per IN) from swap/trade
                price_pct_change = (price_out_for_in - price_out_for_in_prev) / price_out_for_in_prev

                # Slippage calculation #1: elasticity of price with respect to transactions size
                
                # Slippage calculation #2: percentage difference between effective and spot price
                
                for asset in asset_types:
                    # analytical slippage
                    sl = ((reserve_in_pct_change / reserve_out_pct_change) *
                             (weight_asset_in_prev/weight_asset_out_prev) - 1)
                    #sl = ((transactions_in/transactions_out) - price_out_for_in_prev) / price_out_for_in_prev
                    if asset_in_ids[t] == asset or asset == 'all':
                        slippage[asset].append(sl)
                        elasticity[asset].append(price_pct_change / reserve_in_pct_change)
            
            res_in.append(reserve_asset_in)
            res_out.append(reserve_asset_out)
            trans_in.append(transactions_in)
            trans_out.append(transactions_out)
            p_out_for_in.append(price_out_for_in)

    elif market_name == 'uni':
        
        for t in range(1,len(asset_in_ids)-1):
            asset_in_id = 'UNI_' + str(asset_in_ids[t]) + str(asset_out_ids[t])
            asset_out_id = 'UNI_' + str(asset_out_ids[t]) + str(asset_in_ids[t])
            price_id = 'UNI_P_ij'

            reserve_asset_in = rdf[asset_in_id][t+1]
            reserve_asset_in_prev = rdf[asset_in_id][t]
            reserve_asset_out = rdf[asset_out_id][t+1]
            reserve_asset_out_prev = rdf[asset_out_id][t]
            
            # compute prices directly from reserve amounts
            
            transactions_in = rdf.trade_random_size[t]
            transactions_out = -(reserve_asset_out - reserve_asset_out_prev)
            
            if asset_in_ids[t] == 'i':
                price_out_for_in = rdf[price_id][t+1]
                price_out_for_in_prev = rdf[price_id][t]
            elif asset_in_ids[t] == 'j':
                price_out_for_in = 1 / rdf[price_id][t+1]
                price_out_for_in_prev = 1 / rdf[price_id][t]

            

            # If transactions_out or transactions_in is zero, was not a swap/trade event; return n.a.
            if transactions_out == 0 or transactions_in == 0 or t == time_liquidity_in-1 or t == time_liquidity_out-1:
                for asset in asset_types:
                    if asset_in_ids[t] == asset or asset == 'all':
                        elasticity[asset].append(np.nan) 
                        slippage[asset].append(np.nan)
            else:
                # Compute percent change in reserve
                reserve_in_pct_change = transactions_in / reserve_asset_in_prev
                reserve_out_pct_change = transactions_out / reserve_asset_out_prev
                
                # Compute percent change in price (OUT per IN)
                price_pct_change = (price_out_for_in - price_out_for_in_prev) / price_out_for_in_prev
                
                for asset in asset_types:
                    # analytical version of Uniswap slippage:
                    sl = reserve_in_pct_change + market_fee / (1 - market_fee)
                    #sl = ((transactions_in/transactions_out) - price_out_for_in_prev) / price_out_for_in_prev
                    if asset_in_ids[t] == asset or asset == 'all':
                        slippage[asset].append(sl)
                        elasticity[asset].append(price_pct_change / reserve_in_pct_change)

            
            res_in.append(reserve_asset_in)
            res_out.append(reserve_asset_out)
            trans_in.append(transactions_in)
            trans_out.append(transactions_out)
            p_out_for_in.append(price_out_for_in)
    
    return {
        'reserve_asset_in' : pd.Series(res_in),
        'reserve_asset_out' : pd.Series(res_out),
        'transactions_in' : pd.Series(trans_in), 
        'transactions_out' : pd.Series(trans_out),
        'price_out_for_in' : pd.Series(p_out_for_in),
        'elasticity' : { asset : pd.Series(elasticity[asset]) for asset in asset_types }, 
        'slippage' : { asset : pd.Series(slippage[asset]) for asset in asset_types }
    }
    
def impermanent_loss(liquidity_information, rdf, market):
    
    """Compute the impermanent loss every timestep for a single experiment run summarized in
    the 'rdf' dataframe, for the AMM specified by 'market'.
    """
    
    market_name = market['name']
    market_fee = market['fee']

    agent_id = liquidity_information['lp_agent_number']
    asset_id = liquidity_information['asset']
    liquidity_added = liquidity_information['liquidity_added']
    time_liquidity_in = liquidity_information['time_entered'] 
    time_liquidity_out = liquidity_information['time_exited']
    
    # Set ids for asset that the LP has added
    agent_asset_id = 'r_' + str(asset_id) + '_out'
    share_id = 's_' + str(asset_id)
    
    # Build cumulative transactions IN amount (assumes fee eventually assessed to IN asset)
    transactions_in_cumulative = rdf.trade_random_size[time_liquidity_in:time_liquidity_out+1].reset_index(drop = True).cumsum()
    transactions_in_cumulative.name = "transactions_in_cumulative"

    if market_name == 'hydra':
        
        # Get share awarded to LP from addLiquidity event
        share_rewarded = ( rdf['hydra_agents'][time_liquidity_in].iloc[agent_id][share_id] - 
                             rdf['hydra_agents'][time_liquidity_in - 1].iloc[agent_id][share_id] )
        # Compute impermanent loss over entire time series
        Ri_over_Wi = []
        price_of_liquidity = []
        # Build loop computing IL for every timestep
        for t in rdf['hydra_agents'].keys()[time_liquidity_in:]:
            if t <= time_liquidity_out:
                Ri_over_Wi.append(rdf.pool[t].pool[asset_id]['R'] / rdf.pool[t].pool[asset_id]['W'])
                price_of_liquidity.append(rdf.pool[t].pool[asset_id]['P'])
        price_of_liquidity = pd.Series(price_of_liquidity)
        Ri_over_Wi = pd.Series(Ri_over_Wi)
        Wq_over_Sq = rdf['Wq'][time_liquidity_in:time_liquidity_out+1] / rdf['Sq'][time_liquidity_in:time_liquidity_out+1]
        liquidity_removed = Wq_over_Sq.reset_index(drop=True) * Ri_over_Wi.reset_index(drop = True) * share_rewarded
        hold_value = liquidity_added * price_of_liquidity
        pool_value = liquidity_removed * price_of_liquidity
        
        # Get reserve time series for asset the LP has added
        reserve_asset_in = []
        for t in rdf.pool.keys()[time_liquidity_in:]:
            if t<= time_liquidity_out:
                reserve_asset_in.append(rdf.pool[t].pool[str(asset_id)]['R'])
        reserve_asset_in = pd.Series(reserve_asset_in, name='reserve_asset_in')
        
    elif market_name == 'uni':
        
        asset_reserve_key = 'UNI_R' + str(asset_id)
        asset_share_key = 'UNI_S' + str(asset_id)
        #asset_price_key = 'UNI_P_RQ' + str(asset_id)
        asset_price_key = 'UNI_P_ij'
        asset_in_id = 'UNI_R' + str(asset_id)
        if asset_id == 'i':
            asset_out_id = 'j'
        elif asset_id == 'j':
            asset_out_id = 'i'
        asset_in_swap_id = 'UNI_' + str(asset_id) + str(asset_out_id)

        # Get share awarded to LP from addLiquidity event
        share_rewarded = ( rdf['uni_agents'][time_liquidity_in + 1].iloc[agent_id][share_id] - 
                             rdf['uni_agents'][time_liquidity_in].iloc[agent_id][share_id] )
        
        # Compute impermanent loss over entire time series
        
        # Get reserve time series for asset the LP has added
        # 8 May 2021: column 'UNI_Rx' keeps track of Q <-> asset and add/remove Liquidity
        # columns 'UNI_ij' & 'UNI_ji' keep track of asset i <-> asset j swap/trades
        # To get full reserve balance effects, need to diff and update 'UNI_Rx' columns
        #reserve_asset_out = pd.Series([rdf[asset_out_id][0]])
        reserve_asset_in = [rdf[asset_in_id][0]]
        for t in range(1,len(rdf[asset_in_id].diff()) - time_liquidity_in):
            if t <= time_liquidity_out:
                add_in = reserve_asset_in[t-1] + rdf[asset_in_id].diff()[t] + rdf[asset_in_swap_id].diff()[t]
            #add_out = pd.Series([reserve_asset_out[t-1] + rdf[asset_out_id].diff()[t] + 
            #                       rdf[asset_out_swap_id].diff()[t]])
                reserve_asset_in.append(add_in)
        reserve_asset_in = pd.Series(reserve_asset_in, name='reserve_asset_in')
            #reserve_asset_out = reserve_asset_out.append(add_out, ignore_index = True)

        liquidity_removed = (reserve_asset_in / 
                               rdf[asset_share_key][time_liquidity_in:time_liquidity_out+1]) * share_rewarded
        
        #liquidity_removed = (rdf[asset_reserve_key][time_liquidity_in:time_liquidity_out+1] / 
        #                       rdf[asset_share_key][time_liquidity_in:time_liquidity_out+1]) * share_rewarded
        
        #print('share after:', rdf['uni_agents'][time_liquidity_in+1].iloc[agent_id][share_id])
        #print('share before:', rdf['uni_agents'][time_liquidity_in].iloc[agent_id][share_id])
        
        if asset_id == 'i':
            price_of_liquidity = rdf[asset_price_key][time_liquidity_in:time_liquidity_out+1]
        elif asset_id == 'j':
            price_of_liquidity = 1 / rdf[asset_price_key][time_liquidity_in:time_liquidity_out+1]
        
        pool_value = liquidity_removed * price_of_liquidity
        hold_value = liquidity_added * price_of_liquidity

        # Get reserve time series for asset the LP has added
        # 8 May 2021: column 'UNI_Rx' keeps track of Q <-> asset and add/remove Liquidity
        # columns 'UNI_ij' & 'UNI_ji' keep track of asset i <-> asset j swap/trades
        # To get full reserve balance effects, need to diff and update 'UNI_Rx' columns
        #reserve_asset_out = pd.Series([rdf[asset_out_id][0]])
        reserve_asset_in = [rdf[asset_in_id][0]]
        for t in range(1,len(rdf[asset_in_id].diff()) - time_liquidity_in):
            if t <= time_liquidity_out:
                add_in = reserve_asset_in[t-1] + rdf[asset_in_id].diff()[t] + rdf[asset_in_swap_id].diff()[t]
            #add_out = pd.Series([reserve_asset_out[t-1] + rdf[asset_out_id].diff()[t] + 
            #                       rdf[asset_out_swap_id].diff()[t]])
                reserve_asset_in.append(add_in)
        reserve_asset_in = pd.Series(reserve_asset_in, name='reserve_asset_in')
            #reserve_asset_out = reserve_asset_out.append(add_out, ignore_index = True)
    
    IL = (pool_value / hold_value) - 1
    return {
        'pool_value' : pool_value,
        'hold_value' : hold_value,
        'price_liquidity_asset' : price_of_liquidity,
        'impermanent_loss' : IL,
        'liquidity_removed' : liquidity_removed,
        'transactions_in_cumulative' : transactions_in_cumulative,
        'reserve_asset_in' : reserve_asset_in
    }
    
def compute_slippage(liquidity_information, assets, market_information, experiments, verbose = True):
    subset_array = experiments['subset'].unique()
    MC_simulation_array = experiments['simulation'].unique()
    kpis = {}
    
    for subset in subset_array:
        kpi_subset = { market : {} for market in market_information.keys() }
        experiment_by_subset = experiments.sort_values(by=['subset']).reset_index(drop=True)
        sub_experiments = experiment_by_subset[experiment_by_subset['subset']==subset].copy()
        for simulation in MC_simulation_array:
            sub_monte_carlo = sub_experiments[sub_experiments['simulation'] == simulation]
            rdf = sub_monte_carlo.sort_values(by=['timestep']).reset_index(drop=True).copy()
            
            # the following snippet is for future reference if fees are analyzed
            # requires that the 'config_ids' variable be saved from experiments
            # *****
            #config_rdf = [x for x in config_ids if x['simulation_id'] == simulation and 
            #              x['subset_id'] == subset][0]
            #fee = 1 - config_rdf['M']['fee_numerator']/config_rdf['M']['fee_denominator']
            # *****
            if verbose:
                print("***Slippage calc for sim ", simulation, " of subset ", subset, "***")
            for market in market_information.keys():
                kpi_subset[market].update({simulation : slippage(liquidity_information, 
                    assets, rdf, market_information[market])})
        kpis.update({subset: kpi_subset})
        
    return kpis
        
def compute_impermanent_loss(liquidity_information, market_information, experiments, verbose = True):
    subset_array = experiments['subset'].unique()
    MC_simulation_array = experiments['simulation'].unique()
    kpis = {}
    
    for subset in subset_array:
        kpi_subset = { market : {} for market in market_information.keys() }
        experiment_by_subset = experiments.sort_values(by=['subset']).reset_index(drop=True)
        sub_experiments = experiment_by_subset[experiment_by_subset['subset']==subset].copy()
        for simulation in MC_simulation_array:
            sub_monte_carlo = sub_experiments[sub_experiments['simulation'] == simulation]
            rdf = sub_monte_carlo.sort_values(by=['timestep']).reset_index(drop=True).copy()
            if verbose:
                print("***IL calc for sim ", simulation, " of subset ", subset, "***")
            for market in market_information.keys():
                kpi_subset[market].update({simulation : impermanent_loss(liquidity_information,
                    rdf, market_information[market])})
        kpis.update({subset: kpi_subset})
        
    return kpis
    
def compute_regression(kpis, market_information, measures, experiments, verbose = True):
    subset_array = experiments['subset'].unique()
    MC_simulation_array = experiments['simulation'].unique()
    kpi_thresholds = {}
    
    sl_kpis = kpis['slippage']
    il_kpis = kpis['impermanent_loss'] 

    for subset in subset_array:

        kpi_threshold_values = { 
            market : {
            'slippage' : [],
            'elasticity' : [],
            'impermanent_loss' : []
            } for market in market_information.keys() 
        }
        
        for market in market_information.keys():
            for measure in measures:
                for simulation in MC_simulation_array:
                    if measure == 'elasticity':
                        depvar = sl_kpis[subset][market][simulation][measure]['all']
                        indepvar = sl_kpis[subset][market][simulation]['transactions_out']
                        xname = ['Transactions Size']
                    elif measure == 'slippage':
                        depvar = sl_kpis[subset][market][simulation][measure]['all']
                        indepvar = sl_kpis[subset][market][simulation]['transactions_out']
                        indepvar = sm.add_constant(indepvar)
                        xname = ['Constant', 'Transactions Size']
                    elif measure == 'impermanent_loss':
                        depvar = il_kpis[subset][market][simulation][measure]
                        indepvar = pd.concat([il_kpis[subset][market][simulation]['transactions_in_cumulative'],
                                            il_kpis[subset][market][simulation]['reserve_asset_in']], axis=1)
                        indepvar = sm.add_constant(indepvar)
                        xname = ['Constant', 'Transactions Size', 'Balance Size']
                    if verbose:
                        print('********************************************************************************')
                        print(f'Subset: {subset}; Market: {market}; Simulation: {simulation}; Measure: {measure}')
                        print('********************************************************************************')
                    try:
                        model = sm.OLS(depvar,indepvar, missing = 'drop')
                        results = model.fit()
                        if verbose:
                            print(results.summary(xname=xname))
                        kpi_threshold_values[market][measure].append((results.params, results.pvalues))
                    except ValueError as e:
                        print('Market: ', market, ' with measure: ', measure, '; error: ', e)

        kpi_thresholds.update({subset : kpi_threshold_values})
        
    return kpi_thresholds
                        
    
    

