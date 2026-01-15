import os
import numpy as np 
import pandas as pd 
import geopandas as gpd

import math
import json
import joblib
from random_forest.RF_train_test_save import get_rf_features
from sklearn.inspection import permutation_importance

import seaborn as sns
import matplotlib
from matplotlib import cm
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
from matplotlib_map_utils.core.north_arrow import NorthArrow, north_arrow
from matplotlib_map_utils.core.scale_bar import ScaleBar, scale_bar
import matplotlib.pyplot as plt
import matplotlib as mpl

from glob import glob

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '24'


INIT_PATH = 'D:/Research/data'

SEASON_LST = ['SPRING', 'SUMMER', 'FALL', 'WINTER']

YEAR_LST = ['2017', '2019']

##################################################################################################
########### from https://stackoverflow.com/questions/31908982/multi-color-legend-entry ###########
##################################################################################################

# define an object that will be used by the legend
class MulticolorPatch(object):
    def __init__(self, colors, hatch='', alpha=1):
        self.colors = colors
        self.hatch = hatch
        self.alpha = alpha
        
# define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            handlebox.add_artist(
                plt.Rectangle(
                    [width/len(orig_handle.colors) * i - handlebox.xdescent, -handlebox.ydescent],
                    width / len(orig_handle.colors),
                    height, 
                    facecolor=c, 
                    edgecolor='black',
                    alpha=orig_handle.alpha,
                    hatch=orig_handle.hatch
            ))

        # patch = PatchCollection(patches,match_original=True)

        # handlebox.add_artist(patch)
        return handlebox
##################################################################################################
##################################################################################################


def plot_legend():

    bounds = [-1, -0.05, 0, 0.05, 1, 5, 10]

    # colors = ['#f77495', '#ffbac0', '#fffce5', '#a7d1ef', '#6e9fe8', '#456dd4', '#283cb1', '#1a0a7b']     
    # ['#053061', '#186189', '#2497b2', '#3fcfdc', '#f2f2f2', '#fda6b6', '#e85d7c', '#b0264f', '#650b3b']

    # colors = [ '#ce5573', '#f0889a', '#ffc3cb', '#9bdde3', '#6db0c2', '#4983a1', '#295880', '#053061']
    colors = ['#dd5573', '#f5a6b3', '#ffdee8', '#e6ffff', '#93c5d3', '#5a88a6', '#2e4e79', '#001a4e']

    cmap = matplotlib.colors.ListedColormap(colors)
    norm = matplotlib.colors.BoundaryNorm(bounds, len(colors), extend='both')

    plt.rcParams['font.size'] = '40'

    fig = plt.figure(figsize=(25, 1))
    gs = fig.add_gridspec(1,1, wspace=2,hspace=2)

    gs.update(wspace=1, hspace=3)

    ax1 = gs.subplots()

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=ax1, orientation='horizontal',
                label='% PlanetBasemap Surface Water Area \u2014 % DSWE C.3 Surface Water Area')
    
    fig.savefig('D:/Research/figures/huc12_diff_dswe50/c3/legend.jpeg', dpi=300, bbox_inches='tight')

    return

def get_study_area_pswa():
    
    plbm_percent_lst_2017 = []
    plbm_percent_lst_2019 = []

    plbm_area_lst_2017 = []
    plbm_area_lst_2019 = []

    dswe_percent_lst_2017 = []
    dswe_percent_lst_2019 = []

    dswe_area_lst_2017 = []
    dswe_area_lst_2019 = []

    for SZN in SEASON_LST:
        szn_2017 = pd.read_csv(f"{INIT_PATH}/huc_stats_p3/2017_{SZN}_conf1_comp.csv", index_col=0)
        szn_2019 = pd.read_csv(f"{INIT_PATH}/huc_stats_p3/2019_{SZN}_conf1_comp.csv", index_col=0)

        szn_2017_dict = dict(szn_2017[['total_water_plbm', 'total_pixels_plbm', 'SWA_sqkm_plbm',
                                       'total_water_dswe', 'total_pixels_dswe', 'SWA_sqkm_dswe']].sum())
        plbm_percent_2017 = szn_2017_dict['total_water_plbm'] / szn_2017_dict['total_pixels_plbm'] * 100
        dswe_percent_2017 = szn_2017_dict['total_water_dswe'] / szn_2017_dict['total_pixels_dswe'] * 100


        szn_2019_dict = dict(szn_2019[['total_water_plbm', 'total_pixels_plbm', 'SWA_sqkm_plbm',
                                       'total_water_dswe', 'total_pixels_dswe', 'SWA_sqkm_dswe']].sum())
        plbm_percent_2019 = szn_2019_dict['total_water_plbm'] / szn_2019_dict['total_pixels_plbm'] * 100
        dswe_percent_2019 = szn_2019_dict['total_water_dswe'] / szn_2019_dict['total_pixels_dswe'] * 100


        plbm_percent_lst_2017.append(plbm_percent_2017)
        plbm_percent_lst_2019.append(plbm_percent_2019)

        plbm_area_lst_2017.append(szn_2017_dict['SWA_sqkm_plbm'])
        plbm_area_lst_2019.append(szn_2019_dict['SWA_sqkm_plbm'])

        dswe_percent_lst_2017.append(dswe_percent_2017)
        dswe_percent_lst_2019.append(dswe_percent_2019)

        dswe_area_lst_2017.append(szn_2017_dict['SWA_sqkm_dswe'])
        dswe_area_lst_2019.append(szn_2019_dict['SWA_sqkm_dswe'])

    return plbm_percent_lst_2017, plbm_percent_lst_2019, \
            dswe_percent_lst_2017, dswe_percent_lst_2019, \
            plbm_area_lst_2017, plbm_area_lst_2019, \
            dswe_area_lst_2017, dswe_area_lst_2019


def plot_study_area_pswa(pdsi:bool=False):

    plbm_percent_lst_2017, plbm_percent_lst_2019, \
        dswe_percent_lst_2017, dswe_percent_lst_2019, \
        plbm_area_lst_2017, plbm_area_lst_2019, \
        dswe_area_lst_2017, dswe_area_lst_2019 = get_study_area_pswa()

    x_axis = [1, 2, 3, 4]

    fig = plt.figure(figsize=(15, 7), layout='constrained')
    gs = fig.add_gridspec(1,2, hspace=.05, wspace=.03)
    ((ax1, ax2)) = gs.subplots(sharey=True)

    ax1.plot(x_axis, plbm_percent_lst_2017, color='#41E8E8', linewidth=5, 
             linestyle='dashed', marker="o", markersize=20)
    ax1.plot(x_axis, dswe_percent_lst_2017, color='#E849D3', linewidth=5,
             linestyle=(0,(3,1,1,1,1,1)), marker="s", markersize=20)
    ax1.set_ybound(1.8, 8.2)
    ax1.set_title('2017', y=1.0, pad=-65, color='white', size=45)
    ax1.set_xticks([1, 2, 3, 4])
    ax1.set_xticklabels(['Spring', 'Summer', 'Fall', 'Winter'])
    ax1.set_facecolor('#403f3f')
    ax1.xaxis.grid(True, which='major', linestyle = (0, (5, 10)))
    ax1.yaxis.grid(True, which='major', linestyle = (0, (5, 10)))
    ax1.set_ylabel("Percent Surface Water Area", size=30)
  
    ax2.plot(x_axis, plbm_percent_lst_2019, color='#41E8E8', linewidth=5, 
             linestyle='dashed', marker="o", markersize=20)
    ax2.plot(x_axis, dswe_percent_lst_2019, color='#E849D3', linewidth=5,
             linestyle=(0,(3,1,1,1,1,1)), marker="s", markersize=20)
    ax2.set_ybound(1.8, 8.2)
    ax2.set_title('2019', y=1.0, pad=-65, color='white', size=45)
    ax2.set_xticks([1, 2, 3, 4])
    ax2.set_xticklabels(['Spring', 'Summer', 'Fall', 'Winter'])
    ax2.set_facecolor('#403f3f')
    ax2.xaxis.grid(True, which='major', linestyle = (0, (5, 10)))
    ax2.yaxis.grid(True, which='major', linestyle = (0, (5, 10)))

    if pdsi:
        ax1b = ax1.twinx()
        ax1b.plot(x_axis, 
                [-4.705940025799912, -2.4294385782503665, -0.60439063274147, -1.0356908408049983], 
                color='#ff9d00', alpha=0.5, linewidth=2, marker='P', markersize=10)
        ax1b.set_ybound(-6.4, 6.4)
        ax1b.set_yticklabels([])
        
        ax2b = ax2.twinx()
        ax2b.plot(x_axis, 
                [4.9656755308461245, 3.8600105427765197, 1.7894875035476723, 1.4767297220687015], 
                color='#ff9d00', alpha=0.5, linewidth=2, marker='P', markersize=10)
        ax2b.set_ybound(-6.4, 6.4)
        ax2b.set_ylabel("PDSI")
        
    return


def get_class_info_dicts(conf:str):
    
    with open(f"{INIT_PATH}/Planet_DSWE_GSW/50thresh/class_counts_50thresh.json") as f:
        json_info = json.load(f)

    all_lst_2017 = []
    Planet_DSWE_lst_2017 = []
    Planet_GSW_lst_2017 = []
    DSWE_GSW_lst_2017 = []
    Planet_lst_2017 = []
    DSWE_lst_2017 = []
    GSW_lst_2017 = []

    all_lst_2019 = []
    Planet_DSWE_lst_2019 = []
    Planet_GSW_lst_2019 = []
    DSWE_GSW_lst_2019 = []
    Planet_lst_2019 = []
    DSWE_lst_2019 = []
    GSW_lst_2019 = []

    for szn in SEASON_LST:
        szn_2017_dict = json_info[f'2017_{szn}_{conf}']
        all_lst_2017.append(szn_2017_dict['all'][1])
        Planet_DSWE_lst_2017.append(szn_2017_dict['Planet_DSWE'][1])
        Planet_GSW_lst_2017.append(szn_2017_dict['Planet_GSW'][1])
        DSWE_GSW_lst_2017.append(szn_2017_dict['DSWE_GSW'][1])
        Planet_lst_2017.append(szn_2017_dict['PlanetBasemap'][1])
        DSWE_lst_2017.append(szn_2017_dict['DSWE'][1])
        GSW_lst_2017.append(szn_2017_dict['GSW'][1])

        szn_2019_dict = json_info[f'2019_{szn}_{conf}']
        all_lst_2019.append(szn_2019_dict['all'][1])
        Planet_DSWE_lst_2019.append(szn_2019_dict['Planet_DSWE'][1])
        Planet_GSW_lst_2019.append(szn_2019_dict['Planet_GSW'][1])
        DSWE_GSW_lst_2019.append(szn_2019_dict['DSWE_GSW'][1])
        Planet_lst_2019.append(szn_2019_dict['PlanetBasemap'][1])
        DSWE_lst_2019.append(szn_2019_dict['DSWE'][1])
        GSW_lst_2019.append(szn_2019_dict['GSW'][1])

    class_dict_2017 = {'all': all_lst_2017, 
                       'Planet_DSWE': Planet_DSWE_lst_2017,
                       'Planet_GSW': Planet_GSW_lst_2017,
                       'DSWE_GSW': DSWE_GSW_lst_2017,
                       'PlanetBasemap': Planet_lst_2017,
                       'DSWE': DSWE_lst_2017,
                       'GSW': GSW_lst_2017}
    class_dict_2019 = {'all': all_lst_2019, 
                       'Planet_DSWE': Planet_DSWE_lst_2019,
                       'Planet_GSW': Planet_GSW_lst_2019,
                       'DSWE_GSW': DSWE_GSW_lst_2019,
                       'PlanetBasemap': Planet_lst_2019,
                       'DSWE': DSWE_lst_2019,
                       'GSW': GSW_lst_2019}
    
    return class_dict_2017, class_dict_2019


def _add_conf_labels_above(ax, width):
    # conf 1
    ax.annotate('C. 1', xy=(0, 196), xytext=(0,211), 
            ha='center', va='bottom', color='#f2f2f2',
            arrowprops=dict(arrowstyle='-[, widthB=1.15, lengthB=0.35', lw=2.0, color="#f2f2f2"))
    ax.annotate('C. 1', xy=(1, 176), xytext=(1,191), 
            ha='center', va='bottom', color='#f2f2f2',
            arrowprops=dict(arrowstyle='-[, widthB=1.15, lengthB=0.35', lw=2.0, color='#f2f2f2'))
    ax.annotate('C. 1', xy=(2, 187), xytext=(2,202), 
            ha='center', va='bottom', color='#f2f2f2',
            arrowprops=dict(arrowstyle='-[, widthB=1.15, lengthB=0.35', lw=2.0, color='#f2f2f2'))
    ax.annotate('C. 1', xy=(3, 322), xytext=(3,337), 
            ha='center', va='bottom', color='#f2f2f2',
            arrowprops=dict(arrowstyle='-[, widthB=1.15, lengthB=0.35', lw=2.0, color='#f2f2f2'))

    # conf 2
    ax.annotate('C. 2', xy=(0+width*2+0.055, 196), xytext=(0+width*2+0.055,211), 
            ha='center', va='bottom', color='#f2f2f2',
            arrowprops=dict(arrowstyle='-[, widthB=1.15, lengthB=0.35', lw=2.0, color='#f2f2f2'))
    ax.annotate('C. 2', xy=(1+width*2+0.055, 178), xytext=(1+width*2+0.055,193), 
            ha='center', va='bottom', color='#f2f2f2',
            arrowprops=dict(arrowstyle='-[, widthB=1.15, lengthB=0.35', lw=2.0, color='#f2f2f2'))
    ax.annotate('C. 2', xy=(2+width*2+0.055, 189), xytext=(2+width*2+0.055,204), 
            ha='center', va='bottom', color='#f2f2f2',
            arrowprops=dict(arrowstyle='-[, widthB=1.15, lengthB=0.35', lw=2.0, color='#f2f2f2'))
    ax.annotate('C. 2', xy=(3+width*2+0.055, 323), xytext=(3+width*2+0.055,338), 
            ha='center', va='bottom', color='#f2f2f2',
            arrowprops=dict(arrowstyle='-[, widthB=1.15, lengthB=0.35', lw=2.0, color='#f2f2f2'))

    # conf 3
    ax.annotate('C. 3', xy=(0+width*4+0.105, 209), xytext=(0+width*4+0.105,224), 
            ha='center', va='bottom', color='#f2f2f2',
            arrowprops=dict(arrowstyle='-[, widthB=1.15, lengthB=0.35', lw=2.0, color='#f2f2f2'))
    ax.annotate('C. 3', xy=(1+width*4+0.105, 185), xytext=(1+width*4+0.105,200), 
            ha='center', va='bottom', color='#f2f2f2',
            arrowprops=dict(arrowstyle='-[, widthB=1.15, lengthB=0.35', lw=2.0, color='#f2f2f2'))
    ax.annotate('C. 3', xy=(2+width*4+0.105, 201), xytext=(2+width*4+0.105,216), 
            ha='center', va='bottom', color='#f2f2f2',
            arrowprops=dict(arrowstyle='-[, widthB=1.15, lengthB=0.35', lw=2.0, color='#f2f2f2'))
    ax.annotate('C. 3', xy=(3+width*4+0.105, 351), xytext=(3+width*4+0.105,366), 
            ha='center', va='bottom', color='#f2f2f2',
            arrowprops=dict(arrowstyle='-[, widthB=1.15, lengthB=0.35', lw=2.0, color='#f2f2f2'))

    return ax


def _add_conf_labels_below(ax, width):
    # conf 1
    ax.annotate('C. 1', xy=(0.074, -0.005), xycoords='axes fraction', xytext=(0.074, -0.065), 
            ha='center', va='bottom', color='black',
            arrowprops=dict(arrowstyle='-[, widthB=1, lengthB=0.2', lw=2.0, color='black'))
    ax.annotate('C. 1', xy=(0.312, -0.005), xycoords='axes fraction', xytext=(0.312, -0.065), 
            ha='center', va='bottom', color='black',
            arrowprops=dict(arrowstyle='-[, widthB=1, lengthB=0.2', lw=2.0, color='black'))
    ax.annotate('C. 1', xy=(0.55, -0.005), xycoords='axes fraction', xytext=(0.55, -0.065), 
            ha='center', va='bottom', color='black',
            arrowprops=dict(arrowstyle='-[, widthB=1, lengthB=0.2', lw=2.0, color='black'))
    ax.annotate('C. 1', xy=(0.788, -0.005), xycoords='axes fraction', xytext=(0.788, -0.065), 
            ha='center', va='bottom', color='black',
            arrowprops=dict(arrowstyle='-[, widthB=1, lengthB=0.2', lw=2.0, color='black'))

    # conf 2
    ax.annotate('C. 2', xy=(0.143, -0.005), xycoords='axes fraction', xytext=(0.143, -0.065), 
            ha='center', va='bottom', color='black',
            arrowprops=dict(arrowstyle='-[, widthB=1, lengthB=0.2', lw=2.0, color='black'))
    ax.annotate('C. 2', xy=(0.3815, -0.005), xycoords='axes fraction', xytext=(0.3815, -0.065), 
            ha='center', va='bottom', color='black',
            arrowprops=dict(arrowstyle='-[, widthB=1, lengthB=0.2', lw=2.0, color='black'))
    ax.annotate('C. 2', xy=(0.6191, -0.005), xycoords='axes fraction', xytext=(0.6191, -0.065), 
            ha='center', va='bottom', color='black',
            arrowprops=dict(arrowstyle='-[, widthB=1, lengthB=0.2', lw=2.0, color='black'))
    ax.annotate('C. 2', xy=(0.8571, -0.005), xycoords='axes fraction', xytext=(0.8571, -0.065), 
            ha='center', va='bottom', color='black',
            arrowprops=dict(arrowstyle='-[, widthB=1, lengthB=0.2', lw=2.0, color='black'))

    # conf 3
    ax.annotate('C. 3', xy=(0.212, -0.005), xycoords='axes fraction', xytext=(0.212, -0.065), 
            ha='center', va='bottom', color='black',
            arrowprops=dict(arrowstyle='-[, widthB=1, lengthB=0.2', lw=2.0, color='black'))
    ax.annotate('C. 3', xy=(0.45, -0.005), xycoords='axes fraction', xytext=(0.45, -0.065), 
            ha='center', va='bottom', color='black',
            arrowprops=dict(arrowstyle='-[, widthB=1, lengthB=0.2', lw=2.0, color='black'))
    ax.annotate('C. 3', xy=(0.688, -0.005), xycoords='axes fraction', xytext=(0.688, -0.065), 
            ha='center', va='bottom', color='black',
            arrowprops=dict(arrowstyle='-[, widthB=1, lengthB=0.2', lw=2.0, color='black'))
    ax.annotate('C. 3', xy=(0.926, -0.005), xycoords='axes fraction', xytext=(0.926, -0.065), 
            ha='center', va='bottom', color='black',
            arrowprops=dict(arrowstyle='-[, widthB=1, lengthB=0.2', lw=2.0, color='black'))

    return ax

def plot_stacked_bar(conf_txt='above'):

    class_dict_conf1_2017, class_dict_conf1_2019 = get_class_info_dicts('conf1')
    class_dict_conf2_2017, class_dict_conf2_2019 = get_class_info_dicts('conf2')
    class_dict_conf3_2017, class_dict_conf3_2019 = get_class_info_dicts('conf3')

    szn_lst = ['Spring', 'Summer', 'Fall', 'Winter']

    n=4
    x = np.arange(n) 
    width=0.12

    fig, ax = plt.subplots(figsize=(15,10))
    plt.rcParams['font.size'] = '24'
    ax.yaxis.grid(True, which='major', linestyle = (0, (5, 10)))
    bottom_17_conf1, bottom_17_conf2, bottom_17_conf3 = np.zeros(4), np.zeros(4), np.zeros(4)
    bottom_19_conf1, bottom_19_conf2, bottom_19_conf3 = np.zeros(4), np.zeros(4), np.zeros(4)
    color_dict = {'all': '#4149D3', 
                  'Planet_DSWE': '#4149D3',
                  'Planet_GSW': '#41E8E8',
                  'DSWE_GSW': '#E849D3',
                  'PlanetBasemap': '#41E8E8',
                  'DSWE': '#E849D3',
                  'GSW': '#000003'}
    # {'all':'#4149D3',
    # 'PlanetBasemap':'#41E8E8',
    # 'DSWE':'#E849D3'}

    for data_name in ['all', 
                  'Planet_DSWE',
                  'Planet_GSW',
                  'PlanetBasemap',
                  'DSWE_GSW',
                  'DSWE']:
        # conf1
        p1 = ax.bar(x-0.05, class_dict_conf1_2017[data_name], width, 
                   label=data_name, bottom=bottom_17_conf1, color=color_dict[data_name], alpha=0.65, hatch='//')
        p1 = ax.bar(x+width-0.05, class_dict_conf1_2019[data_name], 
                   width, label=data_name, bottom=bottom_19_conf1, color=color_dict[data_name], hatch='\\')
        
        bottom_17_conf1 += class_dict_conf1_2017[data_name]
        bottom_19_conf1 += class_dict_conf1_2019[data_name]

        # conf2
        p2 = ax.bar(x+2*width, class_dict_conf2_2017[data_name], width, 
                   label=data_name, bottom=bottom_17_conf2, color=color_dict[data_name], alpha=0.65, hatch='//')
        p2 = ax.bar(x+3*width, class_dict_conf2_2019[data_name], 
                   width, label=data_name, bottom=bottom_19_conf2, color=color_dict[data_name], hatch='\\')

        bottom_17_conf2 += class_dict_conf2_2017[data_name]
        bottom_19_conf2 += class_dict_conf2_2019[data_name]

        # conf3
        p3 = ax.bar(x+4*width+0.05, class_dict_conf3_2017[data_name], width, 
                   label=data_name, bottom=bottom_17_conf3, color=color_dict[data_name], alpha=0.65, hatch='//')
        p3 = ax.bar(x+5*width+0.05, class_dict_conf3_2019[data_name], 
                   width, label=data_name, bottom=bottom_19_conf3, color=color_dict[data_name], hatch='\\')

        bottom_17_conf3 += class_dict_conf3_2017[data_name]
        bottom_19_conf3 += class_dict_conf3_2019[data_name]


    ax.set_facecolor('#000003')
    ax.set_ybound(0, 450)
    ax.set_yticks([0, 100, 200, 300, 400])
    ax.set_ylabel('Surface Water Area (km$^2$)', size=34)

    if conf_txt == 'above':
        plt.xticks(x + width * 2.5, szn_lst, size=30) 
        ax = _add_conf_labels_above(ax, width)

    if conf_txt == 'below':
        plt.xticks(x + width * 2.5, ['\nSpring', '\nSummer', '\nFall', '\nWinter'], size=30) 
        ax = _add_conf_labels_below(ax, width)

    # reordering the labels
    # specify order
    h = [MulticolorPatch(['#E849D3']),
         MulticolorPatch(['#41E8E8']),
         MulticolorPatch(['#4149D3']),
         MulticolorPatch(['#4149D3', '#41E8E8', '#E849D3'], alpha=0.65, hatch='//'),
         MulticolorPatch(['#4149D3', '#41E8E8', '#E849D3'], hatch='\\')]
    l = ['DSWE Only', 'PlanetBasemap Only', 'PlanetBasemap & DSWE', '2017', '2019']

    # # pass handle & labels lists along with order as below
    fig.legend(h, l,\
                ncol=2, fontsize=28, facecolor='#000003', labelcolor='#f2f2f2',\
                handler_map={MulticolorPatch: MulticolorPatchHandler()},
                loc='upper right', bbox_to_anchor=(0.7, 0.865))

    plt.savefig(f"D:/Research/figures/stacked_swArea_50thresh_{conf_txt}.jpeg", dpi=300,\
        facecolor='w', edgecolor='w', transparent=False, pad_inches=0)

    return


def get_feature_importances():
    # Subset bands to those selected in Feature Selection
    bands_full = ['Blue', 'Green', 'Red', 'NIR', 
            'NDWI', 'NDVI', 
            'Blue\n3x3 mean', 'Green\n3x3 mean', 'Red\n3x3 mean', 'NIR\n3x3 mean', 
            'NDWI\n3x3 mean', 'NDVI\n3x3 mean', 
            'Elevation', 'Slope', 'Hillshade',
            'alpha']
    index_bands = [0, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14] # testing from RFECV ROC AUC

    bands = [bands_full[i] for i in index_bands]

    rf = joblib.load('D:/Research/data/RF_SRTM_AUC_compressed.joblib')
    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    
    forest_importances = pd.DataFrame({'Band':bands,
                                       'RF_Importance':importances,
                                       'St-Dev':std}).sort_values(
                                           by=['RF_Importance'],ascending=False)

    # Get the permutation feature importances
    # Testing dataset
    X_test, Y_test = get_rf_features('D:/Research/data/Shapefiles/test_classified_points.shp', train=False)
    perm_feature_importances = permutation_importance(rf, X_test, Y_test, 
                                                    n_repeats=10, random_state=42)
    perm_forest_importances = pd.DataFrame({'Band':bands,
                                            'RF_Perm_Importance':perm_feature_importances['importances_mean'],
                                            'RF_Perm_StDev':perm_feature_importances['importances_std'],
                                            })\
                        .sort_values(by=['RF_Perm_Importance'],ascending=False)
    # perm_forest_importances

    return forest_importances, perm_forest_importances


def plot_featureImportances():

    forest_importances, perm_forest_importances = get_feature_importances()


    fig = plt.figure(figsize=(13, 15), layout='constrained')
    gs = fig.add_gridspec(2,1, hspace=.05, wspace=.03)
    ((ax1, ax2)) = gs.subplots()

    # ax1.set_facecolor('#f2f2f2')
    # ax2.set_facecolor('#f2f2f2')

    forest_importances['RF_Importance'].plot.bar(ax=ax1, color='#494949')
    # forest_importances['RF_Importance'].plot.bar(yerr=forest_importances['St-Dev'], ax=ax1)
    ax1.set_xticklabels(forest_importances['Band'])
    ax1.set_ylabel("Feature Importance\n(Mean decrease in impurity)", size=28)
    ax1.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
    ax1.set_ybound(-0.02,0.225)

    perm_forest_importances['RF_Perm_Importance'].plot.bar(ax=ax2, color='#494949')
    # perm_forest_importances['RF_Perm_Importance'].plot.bar(yerr=perm_forest_importances['RF_Perm_StDev'], ax=ax2)
    ax2.set_xticklabels(perm_forest_importances['Band'])
    ax2.set_ylabel("Permutation Feature Importance\n(Mean accuracy decrease)", size=28)
    ax2.set_yticks([0, 0.005, 0.01, 0.015, 0.02])
    ax2.set_ybound(-0.002,0.0225)

    ax1.yaxis.grid(True, which='major', linestyle = (0, (5, 10)))
    ax2.yaxis.grid(True, which='major', linestyle = (0, (5, 10)))
    
    ax1.text(9.6,0.202,'(a)', size=42)
    ax2.text(9.6,0.0202,'(b)', size=42)

    plt.savefig(f"D:/Research/figures/featureImportance.jpeg", dpi=300,\
        facecolor='w', edgecolor='w', transparent=False, pad_inches=0)

    return


def get_huc12_pswa_diff(thresh=True):

    i=0
    for SZN in SEASON_LST:
        if thresh:
            col_name = 'plbm_dswe50_pSW_diff'
            col_name_17 = 'plbm_dswe50_pSW_diff17'
            col_name_19 = 'plbm_dswe50_pSW_diff19'

        else:
            col_name = 'plbm_dswe_pSW_diff'
            col_name_17 = 'plbm_dswe_pSW_diff17'
            col_name_19 = 'plbm_dswe_pSW_diff19'

        szn_2017 = pd.read_csv(f"{INIT_PATH}/huc_stats_p3/2017_{SZN}_conf3_comp.csv", index_col=0, 
                               usecols=['huc12', col_name, 'pSW_plbm'])
        szn_2019 = pd.read_csv(f"{INIT_PATH}/huc_stats_p3/2019_{SZN}_conf3_comp.csv", index_col=0, 
                               usecols=['huc12', col_name, 'pSW_plbm'])

        temp_df = szn_2017.merge(szn_2019, on='huc12', suffixes=['17', '19'])
        temp_df = temp_df.rename(columns={col_name_17 : f'{SZN}_2017',
                                          col_name_19 : f'{SZN}_2019',
                                          'pSW_plbm17' : f'plbm_{SZN}_2017',
                                          'pSW_plbm19' : f'plbm_{SZN}_2019'})
        
        if i==0:
            full_df = temp_df
            i+=1
        else:
            full_df = full_df.merge(temp_df, on='huc12')

    return full_df

def add_huc12_colors(val):
    if val <= -1:
        return '#dd5573'
    if -1 < val <= -0.05:
        return '#f5a6b3'
    elif -0.05 < val <= 0:
        return '#ffdee8'
    elif 0 < val <= 0.05:
        return '#e6ffff'
    elif 0.05 < val <= 1:
        return '#93c5d3'
    elif 1 < val <= 5:
        return '#5a88a6'
    elif 5 < val <= 10:
        return '#2e4e79'
    elif val > 10:
        return '#001a4e'

def plot_huc12_diff():

    huc12_shp = gpd.read_file('D:/Research/data/NWI/HU8_03130001_Watershed/HUC12_EPSG-5070/huc12_epsg5070.shp', 
                              columns=['huc12', 'areasqkm', 'geometry'])
    huc12_shp[['huc12']] = huc12_shp[['huc12']].apply(pd.to_numeric)

    huc12_diff_df = get_huc12_pswa_diff(thresh=True)

    huc12_shp = huc12_shp.merge(huc12_diff_df, on='huc12')

    yr_szn_lst = ['SPRING_2017', 'SUMMER_2017', 'FALL_2017', 'WINTER_2017',
                  'SPRING_2019', 'SUMMER_2019', 'FALL_2019', 'WINTER_2019']

    for yr_szn in yr_szn_lst:
        huc12_shp[f'{yr_szn}_colors'] = huc12_shp[f'{yr_szn}'].map(add_huc12_colors)

        fig, ax = plt.subplots(figsize=(8,12))
        huc12_shp.plot(
            color=huc12_shp[f'{yr_szn}_colors'], edgecolor='black', linewidth=0.4, ax = ax) 
        huc12_shp.loc[huc12_shp[f'plbm_{yr_szn}'] == 0].plot(
            edgecolor='black',
            hatch='//',
            color="#dfdfdf",
            linewidth=0.4,
            ax=ax)

        if yr_szn == 'WINTER_2019':
            scale_bar(ax, location="lower right", style="ticks", 
                    bar={"projection": huc12_shp.crs,
                        "length": 0.5,
                        "tickwidth": 3,
                        "height":0.2},
                    units={"loc":"opposite"},
                    text={"fontfamily":"serif",
                            #   "fontweight":"bold",
                            "fontsize":24})
            north_arrow(ax=ax, location="lower right", rotation={"degrees":0},
                        label={"fontfamily":"serif",
                            #   "fontweight":"bold",
                            "fontsize":24},
                        aob={"bbox_to_anchor":(0.98, 0.13), 
                            "bbox_transform":ax.transAxes},
                        scale=0.75)
        ax.set_axis_off()
        plt.tight_layout()

        fig.savefig(f'D:/Research/figures/huc12_diff_dswe50/c3/{yr_szn}.jpeg', dpi=300, bbox_inches='tight')
        plt.close()

    return


plot_study_area_pswa()

plot_stacked_bar()

plot_featureImportances()

plot_huc12_diff()
plot_legend()

##### PDSI Info to add to plot #####
# 2017
# 	spring:	-4.705940025799912
# 	summer:	-2.4294385782503665
# 	fall:	-0.60439063274147
# 	winter:	-1.0356908408049983
# 2019
# 	spring:	4.9656755308461245
# 	summer:	3.8600105427765197
# 	fall:	1.7894875035476723
# 	winter:	1.4767297220687015

