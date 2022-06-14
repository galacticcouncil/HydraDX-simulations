import matplotlib.pyplot as plt
import re


def plot_vars(df, var_list: list, sim_labels: list = ('0', '1')) -> None:
    simulations = df.simulation.unique()
    print(simulations)
    for var in var_list:
        fig = plt.figure(figsize=(20, 5))
        if var in df.columns:
            # rows, columns, index
            bounds = (1, 3, 1)
            ax = plt.subplot(*bounds, title=var)
            for i in simulations:
                df[[var, 'timestep']][df['simulation'] == i].astype(float).plot(ax=ax, y=[var], x='timestep',
                                                                                label=[sim_labels[i]])
        else:
            matches = list(filter(lambda s: re.match(f'{var}-.+', s), df.columns))

            for i, label in enumerate(matches):
                # rows, columns, index
                bounds = (1, len(matches), i+1)
                ax = plt.subplot(*bounds, title=label)
                for j in simulations:
                    df[[label, 'timestep']][df['simulation'] == j]\
                        .astype(float).plot(ax=ax, y=[label], x='timestep', label=[sim_labels[j]])
    plt.show()
