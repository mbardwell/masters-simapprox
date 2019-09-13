##################################################################
# code adapted from PyPSA Scigrid-DE example. Accessed June, 2018

# modified by Michael Bardwell, University of Alberta
##################################################################


#make the code as Python 3 compatible as possible
from __future__ import print_function, division, absolute_import

import pypsa

import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import cartopy.crs as ccrs


# In[253]:

def run(savefig=""):
    """
    parameters
    ----------
    savefig: list. Only "real-loading", ""reactive-loading", "generation", "line-loading", "reactive-feedin" are recognised
    """
    
    if savefig is not "":
        print("figures will be saved to directory that run() was called from")
    
    csv_folder_name = "/home/mike/Documents/pypsa/examples/ieee-13/ieee-13-with-load-gen/"

    network = pypsa.Network(import_name=csv_folder_name)


    # In[254]:


    fig,ax = plt.subplots(1,1, subplot_kw={"projection":ccrs.PlateCarree()})

    fig.set_size_inches(6,6)

    load_distribution = network.loads_t.p_set.loc[network.snapshots[0]].groupby(network.loads.bus).sum()

    network.plot(bus_sizes=100*load_distribution, ax=ax, title="Real Load Distribution")
    
    if "real-loading" in savefig:
        plt.savefig("ieee13-realloading.pdf")
        
    
    fig,ax = plt.subplots(1,1, subplot_kw={"projection":ccrs.PlateCarree()})

    fig.set_size_inches(6,6)

    load_distribution = network.loads_t.q_set.loc[network.snapshots[0]].groupby(network.loads.bus).sum()

    network.plot(bus_sizes=100*load_distribution, ax=ax, title="Reactive Load Distribution")
    if "reactive-loading" in savefig:
        plt.savefig("ieee13-reactiveloading.pdf")
    

    # In[255]:


    techs = ["Substation", "Capacitor"]

    n_graphs = len(techs)

    n_cols = 2

    if n_graphs % n_cols == 0:
        n_rows = n_graphs // n_cols
    else:
        n_rows = n_graphs // n_cols + 1

        
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, subplot_kw={"projection":ccrs.PlateCarree()})

    size = 5

    fig.set_size_inches(size*n_cols,size*n_rows)

    for i,tech in enumerate(techs):
        i_row = i // n_cols
        i_col = i % n_cols
        
        ax = axes[i_col]

        gens = network.generators[network.generators.carrier == tech]
        
        if tech == "Substation":
            gen_distribution = gens.groupby("bus").sum()["p_nom"].reindex(network.buses.index,fill_value=0.)
            network.plot(ax=ax,bus_sizes=5*gen_distribution)
        elif tech == "Capacitor":
            gen_distribution = gens.groupby("bus").sum()["q_set"].reindex(network.buses.index,fill_value=0.)
            network.plot(ax=ax,bus_sizes=500*gen_distribution)
        
        ax.set_title(tech)
        
    if "generation" in savefig:
        plt.savefig("ieee13-generation.pdf")


    # In[256]:


    network.pf()
    now = network.snapshots[0]


    # In[257]:


    print("With the linear load flow, there is the following per unit loading:")
    loading = network.buses_t.p0.loc[now]/network.lines.s_nom
    print(loading.describe())


    # In[259]:

    fig,ax = plt.subplots(1,1, subplot_kw={"projection":ccrs.PlateCarree()})
    fig.set_size_inches(6,6)

    network.plot(ax=ax,line_colors=abs(loading),line_cmap=plt.cm.jet,title="Line Loading")
    if "line-loading" in savefig:
        plt.savefig("ieee13-lineloading.pdf")

    # In[261]:

    #plot the reactive power

    fig,ax = plt.subplots(1,1, subplot_kw={"projection":ccrs.PlateCarree()})

    fig.set_size_inches(6,6)

    q = network.buses_t.q.loc[now]

    bus_colors = pd.Series("r",network.buses.index)
    bus_colors[q< 0.] = "b"


    network.plot(bus_sizes=abs(q)*100,ax=ax,bus_colors=bus_colors,title="Reactive Power Feed-in (red=+ve, blue=-ve)")
    if "reactive-feedin" in savefig:
        plt.savefig("ieee13-reactivefeedin.pdf")


    # In[263]:


    # exporting data
    save_path = csv_folder_name + "results/"
    network.buses_t.v_mag_pu.to_csv(save_path + "vmags.csv")
    network.buses_t.v_ang.to_csv(save_path + "vangs.csv")
    network.buses_t.q.to_csv(save_path + "qmags.csv")
    network.lines_t.p0.to_csv(save_path + "linemags.csv")

    return network
