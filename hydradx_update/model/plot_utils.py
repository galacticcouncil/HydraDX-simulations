import matplotlib.pyplot as plt

def plot_state(df, var_list: list, sim_labels: list = ['0', '1', '2', '3']) -> None:
    simulations = df.simulation.unique()
    # print(simulations)
    plot_figs = 101 + 10*len(sim_labels)
    for var in var_list:
        plt.figure(figsize=(15, 5))
        if var in df.columns:
            init = plot_figs # make this scale to number of assets
            ax = plt.subplot(init, title=var)
            for i in simulations:
                df[[var, 'timestep']][df['simulation'] == i].astype(float).plot(ax=ax, y=[var], x='Timestep',
                                                                                label=[sim_labels[i]])
        elif var + '-0' in df.columns:
            max_i = 0
            while var + '-' + str(max_i + 1) in df.columns:
                max_i += 1
            for i in range(max_i + 1):
                init = plot_figs + i # make this scale to number of assets
                var_i = var + '-' + str(i)
                ax = plt.subplot(init, title=var_i)
                for j in simulations:
                    df[[var_i, 'timestep']][df['simulation'] == j].astype(float).plot(ax=ax, y=[var_i], x='Timestep',
                                                                                      label=[sim_labels[j]])
    plt.show()

def plot_vars(df, var_list: list, sim_labels: list = ['0', '1', '2', '3']) -> None:
    simulations = df.simulation.unique()
    print(simulations)
    for var in var_list:
        plt.figure(figsize=(15, 5))
        if var in df.columns:
            init = 131
            ax = plt.subplot(init, title=var)
            for i in simulations:
                df[[var, 'timestep']][df['simulation'] == i].astype(float).plot(ax=ax, y=[var], x='timestep',
                                                                                label=[sim_labels[i]])
        elif var + '-0' in df.columns:
            max_i = 0
            while var + '-' + str(max_i + 1) in df.columns:
                max_i += 1
            for i in range(max_i + 1):
                init = 141 + i # make this scale to number of assets
                var_i = var + '-' + str(i)
                ax = plt.subplot(init, title=var_i)
                for j in simulations:
                    df[[var_i, 'timestep']][df['simulation'] == j].astype(float).plot(ax=ax, y=[var_i], x='timestep',
                                                                                      label=[sim_labels[j]])
    plt.show()


'''

def plot_var_list(var_list, labels, rdf, run=1):
    plt.figure(figsize=(15,5))
    init = 131
    for var in var_list:
        ax = plt.subplot(init, title=var)
        if isinstance(var, list):
            for v in var:                   # Graphing one simulation but multiple vars on one plot
                rdf[(rdf['simulation'] == 0) & (rdf['run'] == run)].astype(float).plot(ax=ax, y=[v], x='timestep')
        elif 'run' in rdf:
            for i in range(len(labels)):    # Graphing multiple simulations on one plot
                rdf[(rdf['simulation'] == i) & (rdf['run'] == run)].astype(float).plot(ax=ax, y=[var], x='timestep',
                                                               label=[labels[i]])
        else:
            for i in range(len(labels)):    # Graphing multiple simulations on one plot
                rdf[rdf['simulation'] == i].astype(float).plot(ax=ax, y=[var], x='timestep',
                                                               label=[labels[i]])
        init += 1


# This is taken from a notebook from Matt Barlin and Kris Paruch
def param_pool_simulation_plot(experiments, y_variable):
    """
    experiments is the simulation result dataframe.
    y_variable is the state_variable (string) to be plotted against default timestep.
    """
    experiments = experiments.sort_values(by=['subset']).reset_index(drop=True)
    cols = 1
    rows = 1
    cc_idx = 0

    while cc_idx < len(experiments):
        cc = experiments.iloc[cc_idx]['subset']

        cc_label = experiments.iloc[cc_idx]['subset']

        sub_experiments = experiments[experiments['subset'] == cc]
        cc_idx += len(sub_experiments)

        fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(15 * cols, 7 * rows))

        df = sub_experiments.copy()

        #df_label = y_variable

        df[y_variable] = df.apply(lambda x: np.array(x[y_variable]), axis = 1)
        colors = ['orange', 'g', 'magenta', 'r', 'k']
        df = df.astype(float).groupby('timestep').agg({y_variable: ['min', 'mean', 'max']}).reset_index()
        ax = axs
        title = y_variable
        # + 'Scenario: ' + str(cc_label)  + ' rules_price'
        ax.set_title(title)
        ax.set_ylabel('Funds')

        df.plot(x='timestep', y=(y_variable, 'mean'), label=y_variable, ax=ax, legend=True, kind='scatter')
        ax.fill_between(df.timestep, df[(y_variable, 'min')], df[(y_variable, 'max')], alpha=0.3)

        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        ax.set_xlabel('Timesteps')
        ax.grid(color='0.9', linestyle='-', linewidth=1)

        plt.tight_layout()

    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.patch.set_alpha(1)
    plt.close()
    return fig


def agent_value_plot(experiments,test_title,T, agent_index):
    agent_h = []
    agent_r_i_out = []
    agent_s_i = []

    df = experiments
    df = df[df['substep'] == df.substep.max()]
    df.fillna(0,inplace=True)

    for i in range (0,T): 
        agent_h_list = []
        agent_h_list.append(df.uni_agents.values[i]['h'][agent_index])
        # agent_h.append(np.mean(agent_h_list))
        agent_h.append(agent_h_list)
   
        agent_r_i_out_list= []
        agent_r_i_out_list.append(df.uni_agents.values[i]['r_i_out'][agent_index])
        p_rq_list = []
        p_rq_list.append(df.UNI_P_RQ.values[i])
        agent_r_i_out.append(np.divide(agent_r_i_out_list,p_rq_list))
  
        agent_s_i_list= []
        s_i_pool = []
        q_reserve = []
        agent_s_i_list.append(df.uni_agents.values[i]['s_i'][agent_index])
        s_i_pool.append(df.UNI_Si.values[i])
        q_reserve.append(df.UNI_Q.values[i])        
        agent_s_i.append(np.multiply(np.divide(agent_s_i_list,s_i_pool),q_reserve))

    agent_total = np.add(np.add(agent_r_i_out,agent_s_i),agent_h)

    fig = plt.figure(figsize=(15, 10))
    plt.plot(range(0,T),agent_h,label='agent_h', marker='o')
    plt.plot(range(0,T),agent_r_i_out,label='agent_r_i_out',marker='o')
    plt.plot(range(0,T),agent_s_i,label='agent_s_i',marker='o')
    plt.plot(range(0,T),agent_total,label='agent_total',marker='o')

    plt.legend()
    plt.title(test_title + ' for Agent ' + str(agent_index))
    plt.xlabel('Timestep')
    plt.ylabel('Agent Holdings Value')
    plt.show()

def agent_plot(experiments,test_title,T, agent_index):
    agent_h = []
    agent_r_i_out = []
    agent_r_i_in = []
    agent_s_i = []

    df = experiments
    df = df[df['substep'] == df.substep.max()]
    df.fillna(0,inplace=True)

    for i in range (0,T): 
        agent_h_list = []
        agent_h_list.append(df.uni_agents.values[i]['h'][agent_index])
        agent_h.append(np.mean(agent_h_list))
   
        agent_r_i_out_list= []
        agent_r_i_out_list.append(df.uni_agents.values[i]['r_i_out'][agent_index])
        agent_r_i_out.append(np.mean(agent_r_i_out_list))
  
        agent_r_i_in_list= []
        agent_r_i_in_list.append(df.uni_agents.values[i]['r_i_in'][agent_index])
        agent_r_i_in.append(np.mean(agent_r_i_in_list))

    fig = plt.figure(figsize=(15, 10))
    plt.plot(range(0,T),agent_h,label='agent_h', marker='o')
    plt.plot(range(0,T),agent_r_i_out,label='agent_r_i_out',marker='o')
    plt.plot(range(0,T),agent_r_i_in,label='agent_r_i_in',marker='o')

 

    plt.legend()
    plt.title(test_title)
    plt.xlabel('Timestep')
    plt.ylabel('Tokens')
    plt.show()


def mean_agent_plot(experiments,test_title,T):
    agent_h = []
    agent_r_i_out = []

    
    df = experiments
    df = df[df['substep'] == df.substep.max()]
    df.fillna(0,inplace=True)

    for i in range (0,T): 
        agent_h_list = []
        agent_h_list.append(df.uni_agents.values[i]['h'])
        agent_h.append(np.mean(agent_h_list))
        agent_r_i_out_list= []
        agent_r_i_out_list.append(df.uni_agents.values[i]['r_i_out'])
        agent_r_i_out.append(np.mean(agent_r_i_out_list))
  
    fig = plt.figure(figsize=(15, 10))
    plt.plot(range(0,T),agent_h,label='agent_h', marker='o')
    plt.plot(range(0,T),agent_r_i_out,label='agent_r_i_out',marker='o')
    plt.legend()
    plt.title(test_title)
    plt.xlabel('Timestep')
    plt.ylabel('Tokens')
    plt.show()

def price_plot(experiments,test_title, price_swap, numerator, denominator):
      
    df = experiments
    df = df[df['substep'] == df.substep.max()]
    df.fillna(0,inplace=True)
 
    fig = plt.figure(figsize=(15, 10))

    token_ratio = df[numerator] / df[denominator]
    plt.plot(df[price_swap],label='Swap Price', marker='o')
    plt.plot(token_ratio,label='Pool Ratio Price',marker='o')
    plt.legend()
    plt.title(test_title)
    plt.xlabel('Timestep')
    plt.ylabel('Price')
    plt.show()

'''
