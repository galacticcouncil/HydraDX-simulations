import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def hydra_pool_plot(experiments,test_title,T, asset_id):
    """
For any asset on the risk side of the Hydra Omnipool this function plots quantities of:
- its reserve
- its shares
- its price
- its coefficient
    """
    asset_R = []
    asset_S = []
    asset_P = []
    asset_C = []

    df = experiments
    df = df[df['substep'] == df.substep.max()]
    df.fillna(0,inplace=True)
    # print(df.index)
    df.reset_index()
    for i in range (df.substep.max(),T+df.substep.max(),df.substep.max()): 
        asset_R_list = []
        asset_R_list.append(df.pool[i].pool[asset_id]['R'])
        asset_R.append(asset_R_list)
        
        asset_S_list = []
        asset_S_list.append(df.pool[i].pool[asset_id]['S'])
        asset_S.append(asset_S_list)

        asset_P_list = []
        asset_P_list.append(df.pool[i].pool[asset_id]['P'])
        # agent_h.append(np.mean(agent_h_list))
        asset_P.append(asset_P_list)

        asset_C_list = []
        asset_C_list.append(df.pool[i].pool[asset_id]['C'])
        # agent_h.append(np.mean(agent_h_list))
        asset_C.append(asset_C_list)
        

    plt.figure(figsize=(20,6))
    plt.subplot(141)
    plt.plot(df.timestep,asset_R,label='Asset Reserve', marker='o')
    plt.xlabel('Timestep')
    plt.ylabel('Reserves')
    plt.legend()
    plt.title('Reserve' + ' for Asset ' + str(asset_id))

    plt.subplot(142)
     
    # plt.plot(range(df.substep.max(),T+df.substep.max(),df.substep.max()),asset_S,label='Asset Shares', marker='o')
    plt.plot(df.timestep,asset_S,label='Asset Shares', marker='o') # asset_S
    plt.xlabel('Timestep')
    plt.ylabel('Shares')
    plt.legend()
    plt.title('Shares' + ' for Asset ' + str(asset_id))

    plt.subplot(143)
    plt.plot(df.timestep,asset_P,label='Asset '+ asset_id + ' Price', marker='o')
    plt.xlabel('Timestep')
    plt.ylabel('Price')
    plt.legend()
    plt.title('Price' + ' for Asset ' + str(asset_id))

    plt.subplot(144)
    plt.plot(df.timestep,asset_C,label='Asset '+ asset_id + ' Price', marker='o')
    plt.xlabel('Timestep')
    plt.ylabel('Coefficient')
    plt.legend()
    plt.title('Coefficient' + ' for Asset ' + str(asset_id))
       
    plt.show()

def hydra_pool_price_plot(experiments,test_title,T, asset_id_list):
    """
For any selection of assets on the risk side of the Hydra omnipool this function plots their prices.
    """
    asset_P = {k:[] for k in asset_id_list}
    df = experiments
    df = df[df['substep'] == df.substep.max()]
    df.fillna(0,inplace=True)

    
    for asset_id in asset_P:
        asset_P_list = []
    
        for i in range(df.substep.max(),T, df.substep.max()): 
        # for asset_id in asset_id_list:
            # print(asset_id)
            asset_P_list = []
            asset_P_list.append(df.pool[i].pool[asset_id]['P'])
            # agent_h.append(np.mean(agent_h_list))
            # asset_P[asset_id].append(asset_P_list)
            asset_P[str(asset_id)].append(df.pool[i].pool[asset_id]['P'])

    # print(asset_P)
    plt.figure(figsize=(12, 8))
    for asset_id in asset_id_list:
        plt.plot(range(df.substep.max(),T,df.substep.max()),asset_P[asset_id],label='Asset '+ asset_id + ' Price in H', marker='o')
    # plt.plot(range(df.substep.max(),T,df.substep.max()),asset_P[asset_id_list[1]],label='Asset '+ asset_id_list[1] + ' Price', marker='o')
    # print(asset_P)
    plt.legend()
    plt.title(test_title + ' for Asset ' + str(asset_id_list))
    plt.xlabel('Timestep')
    plt.ylabel('Asset Base Price')
    plt.show()

def hydra_agent_value_plot_rev(experiments,test_title,T): #, agent_index, asset_id):
    """
This function plots agent values for each agent that went through the Hydra World.
Values are token holdings multiplied by prices.
In tokens since they are virtual should be valued as shares, should fix the seeming negative token gain
    """
    agent_h = []
    agent_r_i_out = []
    agent_s_i = []
    agent_r_j_out = []
    agent_s_j = []

    df = experiments
    df = df[df['substep'] == df.substep.max()]
    df.fillna(0,inplace=True)
    number_of_agents = 8
    for agent_index in range(number_of_agents):
        agent_h = []
        agent_r_i_out = []
        agent_s_i = []
        agent_r_j_out = []
        agent_s_j = []

        for i in range(df.substep.max(),T, df.substep.max()): 
            agent_h_list = []
            agent_h_list.append(df.hydra_agents.values[i]['h'][agent_index])
            # agent_h.append(np.mean(agent_h_list))
            agent_h.append(agent_h_list)
    
            asset_id = 'i'
            agent_r_i_out_list= []
            agent_r_i_out_list.append(df.hydra_agents.values[i]['r_' + asset_id + '_out'][agent_index])
            p_rq_list = []
            p_rq_list.append(float(df.pool[i].pool[asset_id]['P']))
            agent_r_i_out.append(np.divide(agent_r_i_out_list,p_rq_list))
    
            agent_s_i_list= []
            s_i_pool = []
            q_reserve = []
            agent_s_i_list.append(int(df.hydra_agents.values[i]['s_' + asset_id][agent_index]))
            s_i_pool.append(int(df.Sq.values[i]))
            q_reserve.append(int(df.Q.values[i]))        
            agent_s_i.append(np.multiply(np.divide(agent_s_i_list,s_i_pool),q_reserve))



        sub_total_i = np.add(agent_r_i_out,agent_s_i)

        agent_total = np.add(sub_total_i,agent_h)
    
        plt.figure(figsize=(10, 5))
        plt.plot(range(df.substep.max(),T, df.substep.max()),agent_h,label='agent_h', marker='o')
        asset_id = 'i'
        plt.plot(range(df.substep.max(),T, df.substep.max()),agent_r_i_out,label='agent_r_' + asset_id + '_out',marker='o')
        plt.plot(range(df.substep.max(),T, df.substep.max()),agent_s_i,label='agent_s_' + asset_id,marker='o')

        plt.plot(range(df.substep.max(),T, df.substep.max()),agent_total,label='agent_total',marker='o')

        plt.legend()
        plt.title(test_title + ' for Agent ' + str(agent_index))
        plt.xlabel('Timestep')
        plt.ylabel('Agent Holdings Value')
    plt.show()


def hydra_agent_value_plot(experiments,test_title,T): #, agent_index, asset_id):
    """
This function plots agent values for each agent that went through the Hydra World.
Values are token holdings multiplied by prices.
    """
    agent_h = []
    agent_r_i_out = []
    agent_s_i = []
    agent_r_j_out = []
    agent_s_j = []

    df = experiments
    df = df[df['substep'] == df.substep.max()]
    df.fillna(0,inplace=True)
    number_of_agents = 8
    for agent_index in range(number_of_agents):
        agent_h = []
        agent_r_i_out = []
        agent_s_i = []
        agent_r_j_out = []
        agent_s_j = []

        for i in range(df.substep.max(),T, df.substep.max()): 
            agent_h_list = []
            agent_h_list.append(df.hydra_agents.values[i]['h'][agent_index])
            # agent_h.append(np.mean(agent_h_list))
            agent_h.append(agent_h_list)
    
            asset_id = 'i'
            agent_r_i_out_list= []
            agent_r_i_out_list.append(df.hydra_agents.values[i]['r_' + asset_id + '_out'][agent_index])
            p_rq_list = []
            p_rq_list.append(float(df.pool[i].pool[asset_id]['P']))
            agent_r_i_out.append(np.divide(agent_r_i_out_list,p_rq_list))
    
            agent_s_i_list= []
            s_i_pool = []
            q_reserve = []
            agent_s_i_list.append(int(df.hydra_agents.values[i]['s_' + asset_id][agent_index]))
            s_i_pool.append(int(df.Sq.values[i]))
            q_reserve.append(int(df.Q.values[i]))        
            agent_s_i.append(np.multiply(np.divide(agent_s_i_list,s_i_pool),q_reserve))

            asset_id = 'j'
            agent_r_j_out_list= []
            agent_r_j_out_list.append(df.hydra_agents.values[i]['r_' + asset_id + '_out'][agent_index])
            p_rq_list = []
            p_rq_list.append(float(df.pool[i].pool[asset_id]['P']))
            agent_r_j_out.append(np.divide(agent_r_j_out_list,p_rq_list))
    
            agent_s_j_list= []
            s_j_pool = []
            q_reserve = []
            agent_s_j_list.append(int(df.hydra_agents.values[i]['s_' + asset_id][agent_index]))
            s_j_pool.append(int(df.Sq.values[i]))
            q_reserve.append(int(df.Q.values[i]))        
            agent_s_j.append(np.multiply(np.divide(agent_s_j_list,s_j_pool),q_reserve))

        sub_total_i = np.add(agent_r_i_out,agent_s_i)
        sub_total_j = np.add(agent_r_j_out,agent_s_j)

        agent_total = np.add(np.add(sub_total_i,sub_total_j),agent_h)
    
        plt.figure(figsize=(10, 5))
        plt.plot(range(df.substep.max(),T, df.substep.max()),agent_h,label='agent_h', marker='o')
        asset_id = 'i'
        plt.plot(range(df.substep.max(),T, df.substep.max()),agent_r_i_out,label='agent_r_' + asset_id + '_out',marker='o')
        plt.plot(range(df.substep.max(),T, df.substep.max()),agent_s_i,label='agent_s_' + asset_id,marker='o')
        asset_id = 'j'
        plt.plot(range(df.substep.max(),T, df.substep.max()),agent_r_j_out,label='agent_r_' + asset_id + '_out',marker='o')
        plt.plot(range(df.substep.max(),T, df.substep.max()),agent_s_j,label='agent_s_' + asset_id,marker='o')
 
        plt.plot(range(df.substep.max(),T, df.substep.max()),agent_total,label='agent_total',marker='o')

        plt.legend()
        plt.title(test_title + ' for Agent ' + str(agent_index))
        plt.xlabel('Timestep')
        plt.ylabel('Agent Holdings Value')
    plt.show()

def agent_value_plot(experiments,test_title,T): #, agent_index, asset_id):
    """
This function plots agent values for each agent that went through the Uniswap World.
Values are token holdings multiplied by prices.
    """
    agent_h = []
    agent_r_i_out = []
    agent_s_i = []
    agent_r_j_out = []
    agent_s_j = []
    
    df = experiments
    df = df[df['substep'] == df.substep.max()]
    df.fillna(0,inplace=True)

    number_of_agents = 8
    for agent_index in range(number_of_agents):
        agent_h = []
        agent_r_i_out = []
        agent_s_i = []
        agent_r_j_out = []
        agent_s_j = []
    
        for i in range (0,T): 
            agent_h_list = []
            agent_h_list.append(df.uni_agents.values[i]['h'][agent_index])
            # agent_h.append(np.mean(agent_h_list))
            agent_h.append(agent_h_list)

            asset_id = 'i'
            agent_r_i_out_list= []
            agent_r_i_out_list.append(df.uni_agents.values[i]['r_' + asset_id + '_out'][agent_index])
            p_rq_list = []
            p_rq_list.append(df['UNI_P_RQ' + asset_id].values[i])
            agent_r_i_out.append(np.divide(agent_r_i_out_list,p_rq_list))
    
            agent_s_i_list= []
            s_i_pool = []
            q_reserve = []
            agent_s_i_list.append(df.uni_agents.values[i]['s_' + asset_id][agent_index])
            s_i_pool.append(df['UNI_S' + asset_id].values[i])
            q_reserve.append(df['UNI_S' + asset_id].values[i])        
            agent_s_i.append(np.multiply(np.divide(agent_s_i_list,s_i_pool),q_reserve))

            asset_id = 'j'
            agent_r_j_out_list= []
            agent_r_j_out_list.append(df.uni_agents.values[i]['r_' + asset_id + '_out'][agent_index])
            p_rq_list = []
            p_rq_list.append(df['UNI_P_RQ' + asset_id].values[i])
            agent_r_j_out.append(np.divide(agent_r_j_out_list,p_rq_list))
    
            agent_s_j_list= []
            s_j_pool = []
            q_reserve = []
            agent_s_j_list.append(df.uni_agents.values[i]['s_' + asset_id][agent_index])
            s_j_pool.append(df['UNI_S' + asset_id].values[i])
            q_reserve.append(df['UNI_S' + asset_id].values[i])        
            agent_s_j.append(np.multiply(np.divide(agent_s_j_list,s_j_pool),q_reserve))

        sub_total_i = np.add(agent_r_i_out,agent_s_i)
        sub_total_j = np.add(agent_r_j_out,agent_s_j)

        agent_total = np.add(np.add(sub_total_i,sub_total_j),agent_h)
        # print(agent_s_i)
        fig = plt.figure(figsize=(10, 5))
        plt.plot(range(0,T),agent_h,label='agent_h', marker='o')
        asset_id = 'i'
        plt.plot(range(0,T),agent_r_i_out,label='agent_r_' + asset_id + '_out',marker='o')
        plt.plot(range(0,T),agent_s_i,label='agent_s_' + asset_id,marker='o')
        asset_id = 'j'
        plt.plot(range(0,T),agent_r_j_out,label='agent_r_' + asset_id + '_out',marker='o')
        plt.plot(range(0,T),agent_s_j,label='agent_s_' + asset_id,marker='o')
        plt.plot(range(0,T),agent_total,label='agent_total',marker='o')

        plt.legend()
        plt.title(test_title + ' for Agent ' + str(agent_index))
        plt.xlabel('Timestep')
        plt.ylabel('Agent Holdings Value')
    plt.show()

def hydra_agent_plot(experiments,test_title,T): #, agent_index):
    """
This function plots asset holdings for each agent that went through the Hydra World.
Asset holdings are token quantities held by the agent.
    """
    agent_h = []
    agent_r_i_out = []
    agent_r_i_in = []
    agent_r_j_out = []
    agent_r_j_in = []

    df = experiments
    df = df[df['substep'] == df.substep.max()]
    df.fillna(0,inplace=True)
    number_of_agents = 8
    for agent_index in range(number_of_agents):
        agent_h = []
        agent_r_i_out = []
        agent_r_i_in = []
        agent_r_j_out = []
        agent_r_j_in = []
        for i in range (0,T): 
            agent_h_list = []
            agent_h_list.append(df.hydra_agents.values[i]['h'][agent_index])
            agent_h.append(np.mean(agent_h_list))
    
            agent_r_i_out_list= []
            agent_r_i_out_list.append(df.hydra_agents.values[i]['r_i_out'][agent_index])
            agent_r_i_out.append(np.mean(agent_r_i_out_list))
    
            agent_r_i_in_list= []
            agent_r_i_in_list.append(df.hydra_agents.values[i]['r_i_in'][agent_index])
            agent_r_i_in.append(np.mean(agent_r_i_in_list))

            agent_r_j_out_list= []
            agent_r_j_out_list.append(df.hydra_agents.values[i]['r_j_out'][agent_index])
            agent_r_j_out.append(np.mean(agent_r_j_out_list))
    
            agent_r_j_in_list= []
            agent_r_j_in_list.append(df.hydra_agents.values[i]['r_j_in'][agent_index])
            agent_r_j_in.append(np.mean(agent_r_j_in_list))

        plt.figure(figsize=(10, 5))
        plt.plot(range(0,T),agent_h,label='agent_h', marker='o')
        plt.plot(range(0,T),agent_r_i_out,label='agent_r_i_out',marker='o')
        plt.plot(range(0,T),agent_r_i_in,label='agent_r_i_in',marker='o')
        plt.plot(range(0,T),agent_r_j_out,label='agent_r_j_out',marker='o')
        plt.plot(range(0,T),agent_r_j_in,label='agent_r_j_in',marker='o')

        plt.legend()
        plt.title(test_title + str(agent_index))
        plt.xlabel('Timestep')
        plt.ylabel('Tokens')
    plt.show()

def agent_plot(experiments,test_title,T):   #, agent_index, asset_id):
    """
This function plots asset holdings for each agent that went through the Uniswap World.
Asset holdings are token quantities held by the agent.
    """
    agent_h = []
    agent_r_i_out = []
    agent_r_i_in = []
    agent_s_i = []
    # asset_P = {k:[] for k in asset_id_list}

    df = experiments
    df = df[df['substep'] == df.substep.max()]
    df.fillna(0,inplace=True)

    number_of_agents = 8
    for agent_index in range(number_of_agents):
        agent_h = []
        agent_r_i_out = []
        agent_r_i_in = []
        agent_r_j_out = []
        agent_r_j_in = []
        for i in range (0,T): 
            agent_h_list = []
            agent_h_list.append(df.uni_agents.values[i]['h'][agent_index])
            agent_h.append(np.mean(agent_h_list))
            asset_id = 'i'
            agent_r_i_out_list= []
            agent_r_i_out_list.append(df.uni_agents.values[i]['r_' + asset_id + '_out'][agent_index])
            agent_r_i_out.append(np.mean(agent_r_i_out_list))
    
            agent_r_i_in_list= []
            agent_r_i_in_list.append(df.uni_agents.values[i]['r_' + asset_id + '_in'][agent_index])
            agent_r_i_in.append(np.mean(agent_r_i_in_list))
            
            asset_id = 'j'
            agent_r_j_out_list= []
            agent_r_j_out_list.append(df.uni_agents.values[i]['r_' + asset_id + '_out'][agent_index])
            agent_r_j_out.append(np.mean(agent_r_j_out_list))
    
            agent_r_j_in_list= []
            agent_r_j_in_list.append(df.uni_agents.values[i]['r_' + asset_id + '_in'][agent_index])
            agent_r_j_in.append(np.mean(agent_r_j_in_list))

        plt.figure(figsize=(10, 5))
        # plt.subplot(121)
        plt.plot(range(0,T),agent_h,label='agent_h', marker='o')
        asset_id = 'i'
        plt.plot(range(0,T),agent_r_i_out,label='agent_r_' + asset_id + '_out',marker='o')
        plt.plot(range(0,T),agent_r_i_in,label='agent_r_' + asset_id + '_in',marker='o')
        asset_id = 'j'
        plt.plot(range(0,T),agent_r_j_out,label='agent_r_' + asset_id + '_out',marker='o')
        plt.plot(range(0,T),agent_r_j_in,label='agent_r_' + asset_id + '_in',marker='o')
        plt.legend()
        plt.title(test_title + str(agent_index))
        plt.xlabel('Timestep')
        plt.ylabel('Tokens')
    plt.show()


def mean_agent_plot(experiments,test_title,T):
    """
This function shows mean agent holdings in the Uniswap World.
    """
    agent_h = []
    agent_r_i_out = []

    
    df = experiments
    df = df[df['substep'] == df.substep.max()]
    df.fillna(0,inplace=True)

    for i in range(df.substep.max(),T, df.substep.max()): 
        agent_h_list = []
        agent_h_list.append(df.uni_agents.values[i]['h'])
        agent_h.append(np.mean(agent_h_list))
        agent_r_i_out_list= []
        agent_r_i_out_list.append(df.uni_agents.values[i]['r_i_out'])
        agent_r_i_out.append(np.mean(agent_r_i_out_list))
  
    fig = plt.figure(figsize=(15, 10))
    plt.plot(range(df.substep.max(),T, df.substep.max()),agent_h,label='agent_h', marker='o')
    plt.plot(range(df.substep.max(),T, df.substep.max()),agent_r_i_out,label='agent_r_i_out',marker='o')
    plt.legend()
    plt.title(test_title)
    plt.xlabel('Timestep')
    plt.ylabel('Tokens')
    plt.show()

def price_plot(experiments,test_title, price_swap, numerator, denominator):
    """
This function shows two plots of swap prices of two assets in the Uniswap World.
Once where fees are included and once without fees.
    """
      
    df = experiments
    df = df[df['substep'] == df.substep.max()]
    df.fillna(0,inplace=True)
 
    plt.figure(figsize=(12, 8))
    
    token_ratio =  df[denominator]/ df[numerator]  
    plt.plot(df[price_swap],label='Swap Price', marker='o')
    plt.plot(token_ratio,label='Pool Ratio Price',marker='o')
    plt.legend()
    plt.title(test_title)
    plt.xlabel('Timestep')
    plt.ylabel('Price')
    plt.show()

def IL_plot(experiments,test_title, periods):
      
    df = experiments
    df = df[df['substep'] == df.substep.max()]
    df.fillna(0,inplace=True)
 
    plt.figure(figsize=(12, 8))

    UNI_IL_i = 2* np.sqrt(df.UNI_P_RQi.pct_change(periods)) / (1 + df.UNI_P_RQi.pct_change(periods)) - 1
    UNI_IL_j = 2* np.sqrt(df.UNI_P_RQj.pct_change(periods)) / (1 + df.UNI_P_RQj.pct_change(periods)) - 1
    
    plt.plot(UNI_IL_i,label='Asset i', marker='o')
    plt.plot(UNI_IL_j,label='Asset j',marker='o')
    plt.legend()
    plt.title(test_title + str(periods))
    plt.xlabel('Timestep')
    plt.ylabel('Price')
    plt.show()

def trade_liq_plot(experiments,test_title,T, asset_id_list):
    """
    Plot share to reserve ratio - S/R for each asset
    """
    asset_R = {k:[] for k in asset_id_list}
    df = experiments
    df = df[df['substep'] == df.substep.max()]
    df.fillna(0,inplace=True)

    plt.figure(figsize=(12, 8))
    # plt.subplot(121)
    for asset_id in asset_R:
        asset_R_list = []
        # asset_S_list = []

        for i in range(df.substep.max(),T, df.substep.max()): 
            asset_R_list = []
            # asset_S_list = []
            R = df.pool[i].pool[asset_id]['R']
            S =  df.pool[i].pool[asset_id]['S']
            # asset_R_list.append(S/R)
            # asset_R[str(asset_id)].append(asset_R_list)

            # asset_R_list.append(df.pool[i].pool[asset_id]['R'])
            asset_R[str(asset_id)].append(S/R)

    for asset_id in asset_id_list:
        plt.plot(range(df.substep.max(),T,df.substep.max()),asset_R[asset_id],label='Asset '+ asset_id, marker='o')
   
    plt.legend()
    plt.title(test_title)
    plt.xlabel('Timestep')
    plt.ylabel('Share to Reserve Ratio')
    plt.show()

def rel_price_plot(experiments,test_title,T, asset_id_list):
    """
    asset_id_list is an asset pair only to view relative prices
    """
    asset_R = {k:[] for k in asset_id_list}
    df = experiments
    df = df[df['substep'] == df.substep.max()]
    df.fillna(0,inplace=True)

    plt.figure(figsize=(12, 8))
    # plt.subplot(121)
    for asset_id in asset_R:
        asset_R_list = []
        # asset_S_list = []

        for i in range(df.substep.max(),T, df.substep.max()): 
            asset_R_list = []
            # asset_S_list = []
            R = df.pool[i].pool[asset_id]['R']
            S =  df.pool[i].pool[asset_id]['S']
            # asset_R_list.append(S/R)
            # asset_R[str(asset_id)].append(asset_R_list)

            # asset_R_list.append(df.pool[i].pool[asset_id]['R'])
            asset_R[str(asset_id)].append(R/S)

    i_in_j = [j / i for i,j in zip(*asset_R.values())]
    j_in_i = [i / j for i,j in zip(*asset_R.values())]

    # print(res)

    # for asset_id in asset_id_list:

    #     if asset_id=='i':
    #     # plt.plot(range(df.substep.max(),T,df.substep.max()),asset_R[asset_id],label='Asset '+ asset_id, marker='o')
    #         plt.plot(range(df.substep.max(),T,df.substep.max()),i_in_j,label='Asset Price '+ asset_id + ', j', marker='o')
    #     elif asset_id=='j':
    #     # plt.plot(range(df.substep.max(),T,df.substep.max()),asset_R[asset_id],label='Asset '+ asset_id, marker='o')
    #         plt.plot(range(df.substep.max(),T,df.substep.max()),j_in_i,label='Asset Price '+ asset_id + ', i', marker='o')
   
    for count, asset_id in enumerate(asset_id_list):
        # print(count, asset_id_list[count], asset_id_list[count-1])
        # if asset_id=='i':
        # plt.plot(range(df.substep.max(),T,df.substep.max()),asset_R[asset_id],label='Asset '+ asset_id, marker='o')
        if count == 0:
            plt.plot(range(df.substep.max(),T,df.substep.max()),i_in_j,label='Asset Price '+ asset_id + ',' +asset_id_list[count-1], marker='o')
        else:
            plt.plot(range(df.substep.max(),T,df.substep.max()),j_in_i,label='Asset Price '+ asset_id + ',' +asset_id_list[count-1], marker='o')

    plt.legend()
    plt.title(test_title)
    plt.xlabel('Timestep')
    plt.ylabel('Price')
    plt.show()

def relative_value_plot(experiments,test_title, T, asset_id_list):
    """
    Plot relative value change- delta R*P for each asset
    """
    asset_R = {k:[] for k in asset_id_list}
    df = experiments
    df = df[df['substep'] == df.substep.max()]
    df.fillna(0,inplace=True)

    for asset_id in asset_R:
        asset_R_list = []
        asset_P_list = []

        for i in range(df.substep.max(),T, df.substep.max()): 
            asset_R_list = []
            # asset_S_list = []
            R = df.pool[i].pool[asset_id]['R']
            P =  df.pool[i].pool[asset_id]['P']
            asset_R[str(asset_id)].append(R*P)

    # print(asset_P)
    plt.figure(figsize=(12, 8))
    for asset_id in asset_id_list:

        value_df = pd.DataFrame(asset_R[asset_id])
        value_df = value_df.pct_change()
        value_df.iloc[0] = 0
        # print(value_df)
        plt.plot(value_df,label='Asset '+ asset_id, marker='o')
  
    plt.legend()
    plt.title(test_title + ' for Asset ' + str(asset_id_list))
    plt.xlabel('Timestep')
    plt.ylabel('Asset Relative Value Change')

    plt.show()

def relative_liq_plot(experiments,test_title, T, asset_id_list):
    """
    Plot relative liquidity change- delta S for each asset
    """
    asset_S = {k:[] for k in asset_id_list}
    df = experiments
    df = df[df['substep'] == df.substep.max()]
    df.fillna(0,inplace=True)

    for asset_id in asset_S:
        for i in range(df.substep.max(),T, df.substep.max()): 
            S = df.pool[i].pool[asset_id]['S']
            asset_S[str(asset_id)].append(S)

    # print(asset_P)
    plt.figure(figsize=(12, 8))
    for asset_id in asset_id_list:

        value_df = pd.DataFrame(asset_S[asset_id])
        value_df = value_df.pct_change()
        value_df.iloc[0] = 0
        # print(value_df)
        plt.plot(value_df,label='Asset '+ asset_id, marker='o')
  
    plt.legend()
    plt.title(test_title + ' for Asset ' + str(asset_id_list))
    plt.xlabel('Timestep')
    plt.ylabel('Asset Liquidity Change')

    plt.show()

def slippage_plot(experiments,test_title, T, asset_id_list):
    """
    Plot relative liquidity change- delta S for each asset
    """
    asset_S = {k:[] for k in asset_id_list}
    df = experiments
    df = df[df['substep'] == df.substep.max()]
    df.fillna(0,inplace=True)
    dR = 0
    for asset_id in asset_S:
        for i in range(df.substep.max(),T, df.substep.max()): 
            S = df.pool[i].pool[asset_id]['S']
            R = df.pool[i].pool[asset_id]['R']
            # need to get diff here
            # dR = R - df.pool[i].pool[asset_id]['R']

            # asset_S[str(asset_id)].append(S*dR/R)
            asset_S[str(asset_id)].append(S/R)



    i_in_j = [j / i - 1 for i,j in zip(*asset_S.values())]

    # print(asset_P)
    plt.figure(figsize=(12, 8))
    # for asset_id in asset_id_list:

    #     value_df = pd.DataFrame(asset_S[asset_id])
    #     value_df = value_df.pct_change()
    #     value_df.iloc[0] = 0
    #     # print(value_df)
    #     plt.plot(value_df,label='Asset '+ asset_id, marker='o')
    plt.plot(i_in_j,label='Slippage '+ str(asset_id_list), marker='o')
  
    plt.legend()
    plt.title(test_title + ' for Asset ' + str(asset_id_list))
    plt.xlabel('Timestep')
    plt.ylabel('Asset Liquidity Change')

    plt.show()

def param_test_plot(experiments, config_ids, swept_variable, y_variable, *args):
    """
    experiments is the simulation result dataframe.
    config_ids is the list configs executed upon in the simulation.
    swept_variable is the key (string) in config_ids that was being tested against.
    y_variable is the state_variable (string) to be plotted against default timestep.

    *args for plotting more state_variables (string).
    """
    experiments = experiments.sort_values(by =['subset']).reset_index(drop=True)
    cols = 1
    rows = 1
    cc_idx = 0
    while cc_idx<len(experiments):
        cc = experiments.iloc[cc_idx]['subset']

        cc_label = experiments.iloc[cc_idx]['subset']

        secondary_label = [item['M'][swept_variable] for item in config_ids if  item["subset_id"]== cc_label]
        sub_experiments = experiments[experiments['subset']==cc]
        cc_idx += len(sub_experiments)
        fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(15*cols,7*rows))

        df = sub_experiments.copy()
        colors = ['orange', 'g', 'magenta', 'r', 'k' ]

        ax = axs
        title = swept_variable + ' Effect on ' + y_variable + '\n' + 'Scenario: ' + str(secondary_label[0]) + ' ' + swept_variable
        # + 'Scenario: ' + str(cc_label)  + ' rules_price'
        ax.set_title(title)
        ax.set_ylabel('Funds')

        df.plot(x='timestep', y=y_variable, label=y_variable, ax=ax, legend=True, kind ='scatter')

        for count, arg in enumerate(args):
            df.plot(x='timestep', y=arg, label=arg, ax=ax, legend=True, color = colors[count], kind ='scatter')

        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        ax.set_xlabel('Timesteps')
        ax.grid(color='0.9', linestyle='-', linewidth=1)

        plt.tight_layout()
            
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.patch.set_alpha(1)
    plt.close()
    return display(fig)

def param_fan_plot3(experiments, config_ids, swept_variable, y_variable, *args):
    """
    experiments is the simulation result dataframe.
    config_ids is the list configs executed upon in the simulation.
    swept_variable is the key (string) in config_ids that was being tested against.
    y_variable is the state_variable (string) to be plotted against default timestep.

    *args for plotting more state_variables (string).
    """
    experiments = experiments.sort_values(by =['subset']).reset_index(drop=True)
    cols = 1
    rows = 1
    cc_idx = 0
    while cc_idx<len(experiments):
        cc = experiments.iloc[cc_idx]['subset']

        cc_label = experiments.iloc[cc_idx]['subset']

        secondary_label = [item['M'][swept_variable] for item in config_ids if  item["subset_id"]== cc_label]
        sub_experiments = experiments[experiments['subset']==cc]
        cc_idx += len(sub_experiments)
        fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(15*cols,7*rows))

        df = sub_experiments.copy()
        df = df.groupby('timestep').agg({y_variable: ['min', 'mean', 'max']}).reset_index()
        colors = ['orange', 'g', 'magenta', 'r', 'k' ]

        ax = axs
        title = swept_variable + ' Effect on ' + y_variable + '\n' + 'Scenario: ' + str(secondary_label[0]) + ' ' + swept_variable
        # + 'Scenario: ' + str(cc_label)  + ' rules_price'
        ax.set_title(title)
        ax.set_ylabel('Funds')

        df.plot(x='timestep', y=(y_variable,'mean'),label = y_variable, ax=ax, legend=True)

        ax.fill_between(df.timestep, df[(y_variable,'min')], df[(y_variable,'max')], alpha=0.5)        
        ax.set_xlabel('Blocks')
        ax.grid(color='0.9', linestyle='-', linewidth=1)

        plt.tight_layout()
            
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.patch.set_alpha(1)
    plt.close()
    return display(fig)

def param_fan_plot(experiments, config_ids, swept_variable, y_variable, x_var, *args):
    """
    experiments is the simulation result dataframe.
    config_ids is the list configs executed upon in the simulation.
    swept_variable is the key (string) in config_ids that was being tested against.
    y_variable is the state_variable (string) to be plotted against default timestep.
    *args for plotting more state_variables (string).
    """
    experiments = experiments.sort_values(by =['subset']).reset_index(drop=True)
    cols = 1
    rows = 1
    cc_idx = 0
    while cc_idx<len(experiments):
        cc = experiments.iloc[cc_idx]['subset']

        cc_label = experiments.iloc[cc_idx]['subset']

        secondary_label = [item['M'][swept_variable] for item in config_ids if  item["subset_id"]== cc_label]
        sub_experiments = experiments[experiments['subset']==cc]
        cc_idx += len(sub_experiments)
        fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(15*cols,7*rows))

        df = sub_experiments.copy()
        df = df.groupby('timestep').agg({x_var: ['min', 'mean', 'max']}).reset_index()
        colors = ['orange', 'g', 'magenta', 'r', 'k' ]

        ax = axs
        title = swept_variable + ' Effect on ' + y_variable + '\n' + 'Scenario: ' + str(secondary_label[0]) + ' ' + swept_variable
        # + 'Scenario: ' + str(cc_label)  + ' rules_price'
        ax.set_title(title)
        ax.set_ylabel('Funds')

        
        df.plot(x='timestep', y=(x_var,'mean'),label = y_variable, ax=ax, legend=True)

        ax.fill_between(df.timestep, df[(x_var,'min')], df[(x_var,'max')], alpha=0.3)        
        ax.set_xlabel('Blocks')
        ax.grid(color='0.9', linestyle='-', linewidth=1)

        plt.tight_layout()
            
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.patch.set_alpha(1)
    plt.close()
    return display(fig)

def param_fan_plot2(experiments, config_ids, swept_variable, y_variable, *args):
    """
    experiments is the simulation result dataframe.
    config_ids is the list configs executed upon in the simulation.
    swept_variable is the key (string) in config_ids that was being tested against.
    y_variable is the state_variable (string) to be plotted against default timestep.
    *args for plotting more state_variables (string).
    """
    experiments = experiments.sort_values(by =['subset']).reset_index(drop=True)
    cols = 1
    rows = 1
    cc_idx = 0
    while cc_idx<len(experiments):
        cc = experiments.iloc[cc_idx]['subset']

        cc_label = experiments.iloc[cc_idx]['subset']

        secondary_label = [item['M'][swept_variable] for item in config_ids if  item["subset_id"]== cc_label]
        sub_experiments = experiments[experiments['subset']==cc]
        cc_idx += len(sub_experiments)
        fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(15*cols,7*rows))

        df = sub_experiments.copy()
        df = df.groupby('timestep').agg({'UNI_Ri': ['min', 'mean', 'max']}).reset_index()
        colors = ['orange', 'g', 'magenta', 'r', 'k' ]

        ax = axs
        title = swept_variable + ' Effect on ' + y_variable + '\n' + 'Scenario: ' + str(secondary_label[0]) + ' ' + swept_variable
        # + 'Scenario: ' + str(cc_label)  + ' rules_price'
        ax.set_title(title)
        ax.set_ylabel('Funds')

        
        df.plot(x='timestep', y=('UNI_Ri','mean'),label = y_variable, ax=ax, legend=True)

        ax.fill_between(df.timestep, df[('UNI_Ri','min')], df[('UNI_Ri','max')], alpha=0.3)        
        ax.set_xlabel('Blocks')
        ax.grid(color='0.9', linestyle='-', linewidth=1)

        plt.tight_layout()
            
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.patch.set_alpha(1)
    plt.close()
    return display(fig)

def param_pool_plot(experiments, config_ids, swept_variable, asset_id, y_variable, *args):
    """
    experiments is the simulation result dataframe.
    config_ids is the list configs executed upon in the simulation.
    swept_variable is the key (string) in config_ids that was being tested against.
    asset_id is the asset identifier in the pool (string) e.g i,j,k 
    y_variable is the state_variable (string) to be plotted against default timestep.

    *args for plotting more state_variables (string).
    """
    experiments = experiments.sort_values(by =['subset']).reset_index(drop=True)
    cols = 1
    rows = 1
    cc_idx = 0
    while cc_idx<len(experiments):
        cc = experiments.iloc[cc_idx]['subset']

        cc_label = experiments.iloc[cc_idx]['subset']

        secondary_label = [item['M'][swept_variable] for item in config_ids if  item["subset_id"]== cc_label]
        sub_experiments = experiments[experiments['subset']==cc]
        cc_idx += len(sub_experiments)
        fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(15*cols,7*rows))

        df = sub_experiments.copy()

        df_label = y_variable + asset_id
        df[df_label] = df.pool.apply(lambda x: np.array(x.pool[asset_id][y_variable]))
        colors = ['orange', 'g', 'magenta', 'r', 'k' ]
        
        ax = axs
        title = swept_variable + ' Effect on Pool Asset ' + asset_id + '\n' + 'Scenario: ' + str(secondary_label[0]) + ' ' + swept_variable
        # + 'Scenario: ' + str(cc_label)  + ' rules_price'
        ax.set_title(title)
        ax.set_ylabel('Funds')

        df.plot(x='timestep', y=df_label, label=df_label, ax=ax, legend=True, kind ='scatter')

        for count, arg in enumerate(args):
            df_arg_label = arg + asset_id
            df[df_arg_label] = df.pool.apply(lambda x: np.array(x.pool[asset_id][arg]))

            df.plot(x='timestep', y=df_arg_label, label=df_arg_label, ax=ax, legend=True, color = colors[count], kind ='scatter')

        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        ax.set_xlabel('Timesteps')
        ax.grid(color='0.9', linestyle='-', linewidth=1)

        plt.tight_layout()
            
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.patch.set_alpha(1)
    plt.close()
    return display(fig)

def param_pool_fan_plot(experiments, config_ids, swept_variable, asset_id, y_variable, *args):
    """
    experiments is the simulation result dataframe.
    config_ids is the list configs executed upon in the simulation.
    swept_variable is the key (string) in config_ids that was being tested against.
    asset_id is the asset identifier in the pool (string) e.g i,j,k 
    y_variable is the state_variable (string) to be plotted against default timestep.

    *args for plotting more state_variables (string).
    """
    experiments = experiments.sort_values(by =['subset']).reset_index(drop=True)
    cols = 1
    rows = 1
    cc_idx = 0
    while cc_idx<len(experiments):
        cc = experiments.iloc[cc_idx]['subset']

        cc_label = experiments.iloc[cc_idx]['subset']

        secondary_label = [item['M'][swept_variable] for item in config_ids if  item["subset_id"]== cc_label]
        sub_experiments = experiments[experiments['subset']==cc]
        cc_idx += len(sub_experiments)
        fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(15*cols,7*rows))

        df = sub_experiments.copy()
        df = df.pool.groupby('timestep').agg({y_variable: ['min', 'mean', 'max']}).reset_index()

        df_label = y_variable + asset_id
        df[df_label] = df.pool.apply(lambda x: np.array(x.pool[asset_id][y_variable]))
        colors = ['orange', 'g', 'magenta', 'r', 'k' ]
        
        ax = axs
        title = swept_variable + ' Effect on Pool Asset ' + asset_id + '\n' + 'Scenario: ' + str(secondary_label[0]) + ' ' + swept_variable
        # + 'Scenario: ' + str(cc_label)  + ' rules_price'
        ax.set_title(title)
        ax.set_ylabel('Funds')

        df.plot(x='timestep', y=df_label, label=df_label, ax=ax, legend=True, kind ='scatter')

        for count, arg in enumerate(args):
            df_arg_label = arg + asset_id
            df[df_arg_label] = df.pool.apply(lambda x: np.array(x.pool[asset_id][arg]))

            df.plot(x='timestep', y=df_arg_label, label=df_arg_label, ax=ax, legend=True, color = colors[count], kind ='scatter')
            ax.fill_between(df.timestep, x.pool[(y_variable,'min')], x.pool[(y_variable,'max')], alpha=0.5) 

        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        ax.set_xlabel('Timesteps')
        ax.grid(color='0.9', linestyle='-', linewidth=1)

        plt.tight_layout()
            
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.patch.set_alpha(1)
    plt.close()
    return display(fig)
    
def slippage_fan_plot(swept_var, sweep_dict, sl_kpis, market_information):
    colors = ['green', 'blue']
    subset_array = list(sl_kpis.keys())
    MC_simulation_array = list(sl_kpis[subset_array[0]][list(market_information)[0]].keys())
    markets = market_information.keys()
    ncols = len(markets)
    for measure in ['slippage', 'elasticity']:
        for subset in subset_array:
            for asset in ['i', 'j', 'all']:
                fig, axs = plt.subplots(ncols=ncols, nrows=1, figsize=(15,7))
                for i, market in enumerate(markets):
                    title_fig = f"{market} {measure}, sweep value {sweep_dict[subset]} of parameter '{swept_var}' for asset '{asset}'"
                    axs[i].set_title(title_fig)
                    axs[i].set_xlabel('Trade Sequence')
                    p = pd.concat([sl_kpis[subset][market][x][measure][asset] for x in MC_simulation_array], axis = 1)
                    axs[i].plot(p.mean(axis=1))
                    p10 = p.quantile(0.10, axis = 1)
                    p25 = p.quantile(0.25, axis = 1)
                    p75 = p.quantile(0.75, axis = 1)
                    p90 = p.quantile(0.90, axis = 1)
                    axs[i].fill_between(p.index, p25, p75, alpha = 0.5)
                    axs[i].fill_between(p.index, p10, p90, alpha = 0.25, color=colors[i])
                plt.close()
                display(fig)
                
def impermanent_loss_fan_plot(swept_var, sweep_dict, il_kpis, market_information):
    colors = ['green', 'blue']
    subset_array = list(il_kpis.keys())
    MC_simulation_array = list(il_kpis[subset_array[0]][list(market_information)[0]].keys())
    markets = market_information.keys()
    ncols = len(markets)

    for subset in subset_array:
        fig, axs = plt.subplots(ncols=ncols, nrows=1, figsize=(15,7))
        for i, market in enumerate(markets):
            title_fig = f"{market} IL, sweep value {sweep_dict[subset]} of parameter '{swept_var}'"
            axs[i].set_title(title_fig)
            axs[i].set_xlabel('Trade Sequence')
            p = pd.concat([il_kpis[subset][market][x]['impermanent_loss'] for x in MC_simulation_array], axis = 1)
            axs[i].plot(p.mean(axis=1))
            p10 = p.quantile(0.10, axis = 1)
            p25 = p.quantile(0.25, axis = 1)
            p75 = p.quantile(0.75, axis = 1)
            p90 = p.quantile(0.90, axis = 1)
            axs[i].fill_between(p.index, p25, p75, alpha = 0.5)
            axs[i].fill_between(p.index, p10, p90, alpha = 0.25, color=colors[i])
        plt.close()
        display(fig)

def param_pool_simulation_plot(experiments, config_ids, swept_variable, asset_id, y_variable, *args):
    """
    experiments is the simulation result dataframe.
    config_ids is the list configs executed upon in the simulation.
    swept_variable is the key (string) in config_ids that was being tested against.
    asset_id is the asset identifier in the pool (string) e.g i,j,k 
    y_variable is the state_variable (string) to be plotted against default timestep.

    *args for plotting more state_variables (string).
    """
    experiments = experiments.sort_values(by =['subset']).reset_index(drop=True)
    cols = 1
    rows = 1
    cc_idx = 0
    while cc_idx<len(experiments):
        cc = experiments.iloc[cc_idx]['subset']

        cc_label = experiments.iloc[cc_idx]['subset']

        secondary_label = [item['M'][swept_variable] for item in config_ids if  item["subset_id"]== cc_label]
        sub_experiments = experiments[experiments['subset']==cc]
        cc_idx += len(sub_experiments)
        fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(15*cols,7*rows))

        df = sub_experiments.copy()

        df_label = y_variable + asset_id
        df[df_label] = df.pool.apply(lambda x: np.array(x.pool[asset_id][y_variable]))
        colors = ['orange', 'g', 'magenta', 'r', 'k' ]
        df = df.groupby('timestep').agg({df_label: ['min', 'mean', 'max']}).reset_index()
        ax = axs
        title = swept_variable + ' Effect on Pool Asset ' + asset_id + '\n' + 'Scenario: ' + str(secondary_label[0]) + ' ' + swept_variable
        # + 'Scenario: ' + str(cc_label)  + ' rules_price'
        ax.set_title(title)
        ax.set_ylabel('Funds')

        df.plot(x='timestep', y=(df_label,'mean'), label=df_label, ax=ax, legend=True, kind ='scatter')
        ax.fill_between(df.timestep, df[(df_label,'min')], df[(df_label,'max')], alpha=0.3)    
        for count, arg in enumerate(args):
            df = sub_experiments.copy()
            
            df_arg_label = arg + asset_id
            df[df_arg_label] = df.pool.apply(lambda x: np.array(x.pool[asset_id][arg]))
            df = df.groupby('timestep').agg({df_arg_label: ['min', 'mean', 'max']}).reset_index()

            df.plot(x='timestep', y=(df_arg_label,'mean'), label=df_arg_label, ax=ax, legend=True, color = colors[count], kind ='scatter')
            ax.fill_between(df.timestep, df[(df_arg_label,'min')], df[(df_arg_label,'max')], alpha=0.3)    

        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        ax.set_xlabel('Timesteps')
        ax.grid(color='0.9', linestyle='-', linewidth=1)

        plt.tight_layout()
            
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.patch.set_alpha(1)
    plt.close()
    return display(fig)


