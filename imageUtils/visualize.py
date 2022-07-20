import os
import sys
import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

from surfplot import Plot

import SUITPy.flatmap as flatmap
from nilearn.plotting import view_surf

from imageUtils.helper_functions import get_gifti_colors, get_gifti_columns, get_gifti_labels

_base_dir = os.path.dirname(os.path.abspath(__file__))
_surf_dir = os.path.join(_base_dir, 'surfaces')

def view_cerebellum(
    gifti, 
    cscale=None, 
    colorbar=True, 
    title=True,
    new_figure=True,
    cmap='coolwarm',
    outpath=None,
    labels=None
    ):
    """Visualize (optionally saves) data on suit flatmap, plots either *.func.gii or *.label.gii data

    Args: 
        gifti (str): full path to gifti image
        cscale (list or None): default is None
        colorbar (bool): default is False.
        title (bool): default is True
        new_figure (bool): default is True. If false, appends to current axis. 
        cmap (str or matplotlib colormap): default is 'jet'
        labels (list): list of labels for *.label.gii. default is None
    """


    # figure out if 3D or 4D
    img = nib.load(gifti)

    # determine overlay and get metadata
    # get column names
    if '.func.' in gifti:
        overlay_type = 'func'
    elif '.label.' in gifti:
        overlay_type = 'label'
        _, _, cmap = get_gifti_colors(img)
        labels = get_gifti_labels(img)

    for (data, col) in zip(img.darrays, get_gifti_columns(img)):

        view = flatmap.plot(data.data, 
        overlay_type=overlay_type, 
        cscale=cscale, 
        cmap=cmap, 
        label_names=labels, 
        colorbar=colorbar, 
        new_figure=new_figure
        )

        # print title
        fname = Path(gifti).name.split('.')[0]
        if title:
            view.set_title(f'{fname}-{col}')

        # save to disk
        if outpath is not None:
            plt.savefig(os.path.join(outpath, f'{fname}-{col}.png'), dpi=300, format='png', bbox_inches='tight', pad_inches=0)

        plt.show()

def view_cortex(
    gifti, 
    surf_mesh=None,
    threshold=None,
    title=True,
    vmin=None,
    vmax=None,
    outpath=None,
    symmetric_cmap=True,
    column=None,
    colorbar=True
    ):
    """Visualize (optionally saves) data on inflated cortex, plots either *.func.gii or *.label.gii data

    Args: 
        gifti (str): fullpath to file: *.func.R.gii or *.label.R.gii
        surf_mesh (str or None): fullpath to surface mesh file *.inflated.surf.gii. If None, takes group `inflated` mesh from `surfaces`
        vmin (int or None):
        title (bool): default is True
        cmap (str or matplotlib colormap): 'default is "jet"' 
    """

    # figure out if 3D or 4D
    img = nib.load(gifti)

    if '.R.' in gifti:
        hemisphere = 'R'
    elif '.L.' in gifti:
        hemisphere = 'L'

    # determine overlay and get metadata
    # get column names
    if '.func.' in gifti:
        overlay_type = 'func'
    elif '.label.' in gifti:
        overlay_type = 'label'
        _, _, cmap = get_gifti_colors(img)
        labels = get_gifti_labels(img)

    if surf_mesh is None:
        surf_mesh = os.path.join(_surf_dir, f'fs_LR.32k.{hemisphere}.inflated.surf.gii')
    
    data_all = img.darrays
    cols = get_gifti_columns(img)

    if column is not None:
        data_all = [data_all[column]]
        cols = [cols[column]]

    for (data, col) in zip(data_all, cols):
        
        if hemisphere=='L':
            orientation = 'lateral'

        # print title
        fname = Path(gifti).name.split('.')[0]
        title_name = None
        if title:
            title_name = f'{fname}-{col}-{orientation}'
        
        # plot to surface
        view = view_surf(surf_mesh=surf_mesh[0], 
                        surf_map=np.nan_to_num(data.data), 
                        vmin=vmin,
                        vmax=vmax,
                        threshold=threshold,
                        symmetric_cmap=symmetric_cmap,
                        title=title_name,
                        black_bg=False,
                        colorbar=colorbar
                        ) 
    
        view.open_in_browser()

        # save to disk
        if outpath is not None:
            plt.savefig(os.path.join(outpath, f'{fname}-{col}-{orientation}.png'), dpi=300, format='png', bbox_inches='tight', pad_inches=0)

def view_cortex_inflated(
    giftis,
    surf_mesh=None,
    colorbar=True, 
    borders=False,
    outpath=None,
    column=1
    ):
    """save cortical atlas to disk (and plot if plot=True)

    Args: 
        giftis (list of str or list of nib gifti obj): list has to be [left hemisphere, right hemisphere]. 
        surf_mesh (list of str or None): list of fullpaths to surface mesh file [<*.L.inflated*>, <*.R.inflaed*>]. If None, takes `inflated` mesh from `surfaces` for L and R hemispheres
        colorbar (bool): default is True
        borders (bool): default is False
        plot (bool): default is True
    """

    # get surface mesh
    if surf_mesh is None:
        lh = os.path.join(_surf_dir, f'fs_LR.32k.L.inflated.surf.gii')
        rh = os.path.join(_surf_dir, f'fs_LR.32k.R.inflated.surf.gii')

    gifti_dict = {}
    for hem, gifti in zip(['L', 'R'], giftis):
        if isinstance(gifti, str):
            gifti_dict.update({hem: nib.load(gifti)})
        else:
            gifti_dict.update({hem: gifti})

    data_lh_all = gifti_dict['L'].darrays
    data_rh_all = gifti_dict['R'].darrays
    cols = get_gifti_columns(giftis[0])

    if column is not None:
        data_lh_all = [data_lh_all[column]]
        data_rh_all = [data_rh_all[column]]
        cols = [cols[column]]

    for (data_lh, data_rh, col) in zip(data_lh_all, data_rh_all, cols):

        p = Plot(lh[0], rh[0], size=(400, 200), zoom=1.2, views='lateral') # views='lateral', zoom=1.2, 

        p.add_layer({'left': np.nan_to_num(data_lh.data), 'right': np.nan_to_num(data_rh.data)},  cbar_label=col, as_outline=borders, cbar=colorbar) # cmap='YlOrBr_r',

        kws = {'location': 'right', 'label_direction': 45, 'decimals': 3,
       'fontsize': 16, 'n_ticks': 2, 'shrink': .15, 'aspect': 8,
       'draw_border': False}
        fig = p.build(cbar_kws=kws)

        plt.show()
        
        if outpath is not None:
            fig.savefig(outpath, dpi=300, bbox_inches='tight')
    
    return fig

def view_atlas_cortex(
    giftis=None,
    surf_mesh=None,
    colorbar=True, 
    borders=False,
    ):
    """save cortical atlas to disk (and plot if plot=True)

    Args: 
        giftis (list of str or None): [<left hemisphere>, <right hemisphere>], if None, surface flatmaps are plotted
        surf_mesh (list of str or None): list of fullpaths to surface mesh file [<*.L.inflated*>, <*.R.inflaed*>]. If None, takes `inflated` mesh from `surfaces` for L and R hemispheres
        colorbar (bool): default is True
        borders (bool): default is False
        plot (bool): default is True
    """

    # get surface mesh
    if surf_mesh is None:
        lh = os.path.join(_surf_dir, f'fs_LR.32k.L.inflated.surf.gii')
        rh = os.path.join(_surf_dir, f'fs_LR.32k.R.inflated.surf.gii')

    _, _, cmap = get_gifti_colors(fpath=giftis[0])
    
    p = Plot(lh, rh)
    
    if giftis is not None:
        p.add_layer({'left': giftis[0], 'right': giftis[1]}, cmap=cmap, cbar_label='Cortical Networks', as_outline=borders, cbar=colorbar) # 
        fig = p.build()
    
    plt.show()
    
    return fig

def view_atlas_cerebellum(
    gifti=None, 
    colorbar=True,
    outpath=None,
    new_figure=True,
    labels=None
    ):
    """General purpose function for plotting (optionally saving) cerebellar atlas
    Args: 
        gifti (str or NOne): fullpath to gifti. If None, flatmap is plotted
        structure (str): default is 'cerebellum'. other options: 'cortex'
        colorbar (bool): default is False. If False, saves colorbar separately to disk.
        outpath (str or None): outpath to file. if None, not saved to disk.
        new_figure (bool): default is True
        lbaels (list of int or None): default is None. 
    Returns:
        viewing object to visualize parcellations
    """
    if gifti is not None:
        img = nib.load(gifti)

        _, _, cmap = get_gifti_colors(img, ignore_0=False)
        label_names = get_gifti_labels(img)
        data = img.darrays[0].data
    else:
        data = np.zeros((28935,1))
        cmap = None
        colorbar = False
        label_names = None

    if labels is not None:
        for idx, num in enumerate(data):
            if num not in labels: 
                data[idx] = 0

    view = flatmap.plot(data, 
        overlay_type='label', 
        cmap=cmap, 
        label_names=label_names, 
        colorbar=colorbar, 
        new_figure=new_figure
        )

    # save to disk
    if outpath is not None:
        plt.savefig(outpath, dpi=300, format='png', bbox_inches='tight', pad_inches=0)

    plt.show()
    
    return view

def view_colorbar(
    gifti,
    outpath=None,
    labels=None,
    orientation='vertical'
    ):
    """Makes colorbar for *.label.gii file
        
    Args:
        gifti (str): full path to *.label.gii
        outpath (str or None): default is None. file not saved to disk.
    """

    rotation = 90
    if orientation is 'horizontal':
        rotation = 45

    plt.figure()
    fig, ax = plt.subplots(figsize=(1,10)) # figsize=(1, 10)

    rgba, cpal, cmap = get_gifti_colors(gifti)

    if labels is None:
        labels = get_gifti_labels(gifti)

    bounds = np.arange(cmap.N + 1)

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb3 = mpl.colorbar.ColorbarBase(ax, cmap=cmap.reversed(cmap), 
                                    norm=norm,
                                    ticks=bounds,
                                    format='%s',
                                    orientation='vertical',
                                    )
    cb3.set_ticklabels(labels[::-1])  
    cb3.ax.tick_params(size=0)
    cb3.set_ticks(bounds+.5)
    cb3.ax.tick_params(axis='y', which='major', labelsize=30, labelrotation=rotation)

    if outpath:
        plt.savefig(outpath, bbox_inches='tight', dpi=150)

    return cb3
