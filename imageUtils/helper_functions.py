import os
import sys
import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from nilearn import image
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
from nilearn.input_data import NiftiMasker
from nilearn.surface import load_surf_data
import subprocess
from scipy.stats import mode

import SUITPy.flatmap as flatmap
from nilearn.surface import vol_to_surf

"""
includes functions for visualizations for cortex and cerebellum
​
@author: Maedbh King
"""

def make_label_gifti_cortex(
    data, 
    anatomical_struct='CortexLeft', 
    label_names=None,
    column_names=None, 
    label_RGBA=None
    ):
    """
    Generates a label GiftiImage from a numpy array
       @author joern.diedrichsen@googlemail.com, Feb 2019 (Python conversion: switt)

    INPUTS:
        data (np.array):
             numVert x numCol data
        anatomical_struct (string):
            Anatomical Structure for the Meta-data default= 'CortexLeft'
        label_names (list): 
            List of strings for label names
        column_names (list):
            List of strings for names for columns
        label_RGBA (list):
            List of rgba vectors
    OUTPUTS:
        gifti (label GiftiImage)

    """
    try:
        num_verts, num_cols = data.shape
    except: 
        data = np.reshape(data, (len(data),1))
        num_verts, num_cols  = data.shape

    num_labels = len(np.unique(data))

    # check for 0 labels
    zero_label = 0 in data

    # Create naming and coloring if not specified in varargin
    # Make columnNames if empty
    if column_names is None:
        column_names = []
        for i in range(num_cols):
            column_names.append("col_{:02d}".format(i+1))

    # Determine color scale if empty
    if label_RGBA is None:
        hsv = plt.cm.get_cmap('hsv',num_labels)
        color = hsv(np.linspace(0,1,num_labels))
        # Shuffle the order so that colors are more visible
        color = color[np.random.permutation(num_labels)]
        label_RGBA = np.zeros([num_labels,4])
        for i in range(num_labels):
            label_RGBA[i] = color[i]
        if zero_label:
            label_RGBA = np.vstack([[0,0,0,1], label_RGBA[1:,]])

    # Create label names
    if label_names is None:
        idx = 0
        if not zero_label:
            idx = 1
        for i in range(num_labels):
            label_names.append("label-{:02d}".format(i + idx))

    # Create label.gii structure
    C = nib.gifti.GiftiMetaData.from_dict({
        'AnatomicalStructurePrimary': anatomical_struct,
        'encoding': 'XML_BASE64_GZIP'})

    E_all = []
    for (label,rgba,name) in zip(np.arange(num_labels),label_RGBA,label_names):
        E = nib.gifti.gifti.GiftiLabel()
        E.key = label 
        E.label= name
        E.red = rgba[0]
        E.green = rgba[1]
        E.blue = rgba[2]
        E.alpha = rgba[3]
        E.rgba = rgba[:]
        E_all.append(E)

    D = list()
    for i in range(num_cols):
        d = nib.gifti.GiftiDataArray(
            data=np.float32(data[:, i]),
            intent='NIFTI_INTENT_LABEL', 
            datatype='NIFTI_TYPE_FLOAT32', # was NIFTI_TYPE_INT32
            meta=nib.gifti.GiftiMetaData.from_dict({'Name': column_names[i]})
        )
        D.append(d)

    # Make and return the gifti file
    gifti = nib.gifti.GiftiImage(meta=C, darrays=D)
    gifti.labeltable.labels.extend(E_all)
    return gifti

def make_func_gifti_cortex(
    data, 
    anatomical_struct='CortexLeft', 
    column_names=None
    ):
    """
    Generates a function GiftiImage from a numpy array
       @author joern.diedrichsen@googlemail.com, Feb 2019 (Python conversion: switt)

    Args:
        data (np array): shape (vertices x columns) 
        anatomical_struct (str): Anatomical Structure for the Meta-data default='CortexLeft'
        column_names (list or None): List of strings for column names, default is None
    Returns:
        gifti (functional GiftiImage)
    """
    try:
        num_verts, num_cols = data.shape
    except: 
        data = np.reshape(data, (len(data),1))
        num_verts, num_cols  = data.shape
  
    # Make columnNames if empty
    if column_names is None:
        column_names = []
        for i in range(num_cols):
            column_names.append("col_{:02d}".format(i+1))

    C = nib.gifti.GiftiMetaData.from_dict({
    'AnatomicalStructurePrimary': anatomical_struct,
    'encoding': 'XML_BASE64_GZIP'})

    E = nib.gifti.gifti.GiftiLabel()
    E.key = 0
    E.label= '???'
    E.red = 1.0
    E.green = 1.0
    E.blue = 1.0
    E.alpha = 0.0

    D = list()
    for i in range(num_cols):
        d = nib.gifti.GiftiDataArray(
            data=np.float32(data[:, i]),
            intent='NIFTI_INTENT_NONE',
            datatype='NIFTI_TYPE_FLOAT32',
            meta=nib.gifti.GiftiMetaData.from_dict({'Name': column_names[i]})
        )
        D.append(d)

    gifti = nib.gifti.GiftiImage(meta=C, darrays=D)
    gifti.labeltable.labels.append(E)

    return gifti

def get_gifti_colors(
    gifti,
    ignore_0=True
    ):
    """get gifti labels for fpath (should be *.label.gii)

    Args: 
        gifti (str or nib obj): full path to atlas
        ignore_0 (bool): default is True. ignores 0 index
    Returns: 
        rgba (np array): shape num_labels x num_rgba
        cpal (matplotlib color palette)
        cmap (matplotlib colormap)
    """
    if isinstance(gifti, str):
        img = nib.load(gifti)
    elif isinstance(gifti, nib.gifti.gifti.GiftiImage):
        img = gifti
    else:
        ValueError(print('gifti must be gifti object or full path to gifti image'))

    labels = img.labeltable.labels

    rgba = np.zeros((len(labels),4))
    for i,label in enumerate(labels):
        rgba[i,] = labels[i].rgba
    
    if ignore_0:
        rgba = rgba[1:]
        labels = labels[1:]

    cmap = LinearSegmentedColormap.from_list('mylist', rgba, N=len(rgba))
    mpl.cm.register_cmap("mycolormap", cmap)
    cpal = sns.color_palette("mycolormap", n_colors=len(rgba))

    return rgba, cpal, cmap

def get_gifti_labels(
    gifti,
    ignore_0=True
    ):
    """get gifti labels for fpath (should be *.label.gii)

    Args: 
        gifti (str or nib obj): full path to atlas (*.label.gii) or nib obj
    Returns: 
        labels (list): list of label names
    """
    if isinstance(gifti, str):
        img = nib.load(gifti)
    elif isinstance(gifti, nib.gifti.gifti.GiftiImage):
        img = gifti
    else:
        ValueError(print('gifti must be gifti object or full path to gifti image'))

    # labels = img.labeltable.get_labels_as_dict().values()
    label_dict = img.labeltable.get_labels_as_dict()

    if ignore_0:
        label_dict.pop(0)

    return list(label_dict.values())

def get_gifti_columns(
    gifti
    ):
    """get column names from gifti

    Args: 
        gifti (str or nib obj): full path to atlas (*.label.gii) or nib obj
    Returns: 
        column_names (list): list of column names
    """
    if isinstance(gifti, str):
        img = nib.load(gifti)
    elif isinstance(gifti, nib.gifti.gifti.GiftiImage):
        img = gifti
    else:
        ValueError(print('gifti must be gifti object or full path to gifti image'))

    column_names = []
    for col in img.darrays:
        col_name =  list(col.metadata.values())[0]
        column_names.append(col_name)

    return column_names

def make_gifti_cerebellum(
    data, 
    mask,
    outpath='/',
    stats='nanmean', 
    data_type='func',
    save_nifti=True, 
    save_gifti=True,
    column_names=[], 
    label_RGBA=[],
    label_names=[],
    ):
    """Takes data (np array) or 3D/4D nifti obj or str; optionally computes mean/mode along first dimension; optionally saves nifti and gifti map to disk

    If `data` is 3D/4D nifti obj or str, then nifti is masked using `mask` and np array (N x 6937) is returned

    Args: 
        data (np array or nib obj or str): np array of shape (subjs x N x 6937) or shape (N x 6937) or nib obj (3D or 4D) or str (fullpath to nifti)
        mask (nib obj or str): nib obj of mask or fullpath to mask
        outpath (str): save path for output file (must contain *.label.gii or *.func.gii)
        stats (str or None): 'nanmean', 'mode' or None. If not None, `stats` is done on axis=0
        data_type (str): 'func' or 'label'
        save_nifti (bool): default is False, saves nifti to fpath
        column_names (list):
        label_RGBA (list):
        label_names (list):
    Returns: 
        saves gifti and/or nifti image to disk, returns gifti
    """

    if isinstance(mask, str):
        mask = nib.load(mask)
    elif isinstance(mask, nib.nifti1.Nifti1Image):
        pass

    if isinstance(data, str):
        data = nib.load(data)
        data = mask_vol(mask, data, output='2D')
    elif isinstance(data, nib.nifti1.Nifti1Image):
        data = mask_vol(mask, data, output='2D')
    elif isinstance(data, np.ndarray):
        pass

    # get mean or mode of data along first dim
    if stats=='nanmean':
        data = np.nanmean(data, axis=0)
    elif stats=='mode':
        data = mode(data, axis=0)
        data = data.mode[0]
    elif stats is None:
        pass

    # convert cerebellum data array to nifti
    imgs = mask_vol(mask, data, output='3D')
    
    # save nifti(s) to disk
    if save_nifti:
        fname = Path(outpath).name
        if len(imgs)>1:
            img = image.concat_imgs(imgs)
        else:
            img = imgs[0]
        nib.save(img, str(Path(outpath).parent) + '/' + fname.rsplit('.')[0] + '.nii')

    # make and save gifti
    if data_type=='label':
        surf_data = flatmap.vol_to_surf(imgs, space="SUIT", stats='mode')
        gii_img = flatmap.make_label_gifti(data=surf_data, label_names=label_names, column_names=column_names, label_RGBA=label_RGBA)
    elif data_type=='func':
        surf_data = flatmap.vol_to_surf(imgs, space="SUIT", stats='nanmean')
        gii_img = flatmap.make_func_gifti(data=surf_data, column_names=column_names)
    
    if save_gifti:
        nib.save(gii_img, outpath)
        print(f'saving gifti to {outpath}')
        return gii_img

def get_gifti_structure(
    gifti
    ):
    """get gifti structure for `img`

    Args: 
        gifti (str or nib obj): full path to atlas (*.label.gii) or nib obj
    """

    if isinstance(gifti, str):
        img = nib.load(gifti)
    elif isinstance(gifti, nib.gifti.gifti.GiftiImage):
        img = gifti
    else:
        ValueError(print('gifti must be gifti object or full path to gifti image'))

    # get metadata
    meta = img.get_meta()

    structure = meta.metadata['AnatomicalStructurePrimary']

    return structure

def get_gifti_hemisphere(gifti):
    """get gifti hemisphere for `img`

    Args: 
        gifti (str or nib obj): full path to atlas (*.label.gii) or nib obj
    """

    if isinstance(gifti, str):
        img = nib.load(gifti)
    elif isinstance(gifti, nib.gifti.gifti.GiftiImage):
        img = gifti
    else:
        ValueError(print('gifti must be gifti object or full path to gifti image'))

    # get metadata
    meta = img.get_meta()

    structure = meta.metadata['AnatomicalStructurePrimary']

    # check if there is information about hemisphere
    if any(x in structure for x in ['right', 'Right']):
        hemisphere = 'R'
    elif any(x in structure for x in ['left', 'Left']):
        hemisphere = 'L'
    else:
        hemisphere = 'B' # bilateral
    
    return hemisphere

def get_gifti_id(
    gifti
    ):
    """Get id for `roi` for `hemisphere`
    
    Args: 
        gifti (str): gifti (str or nib obj): full path to atlas (*.label.gii) or nib obj
    Returns: 
        1D np array of labels
    """
    
    labels = load_surf_data(gifti)

    # get min, max labels for each hem
    min_label = np.nanmin(labels[labels!=0])
    max_label = np.nanmax(labels[labels!=0])

    # get labels per hemisphere
    labels_hem = np.arange(min_label-1, max_label)

    if '.R.' in gifti:
        labels_hem += labels_hem[-1]+1
    
    return labels_hem

def mask_vol(
    mask, 
    vol, 
    output='2D'
    ):
    """ mask volume using NiftiMasker, input can be volume (3D/4D Nifti1Image) or np array (n_cols x n_voxels) or str (to 3D/4D volume)

    If output is '3D' inverse transform is computed (go from 2D np array to list of 3D nifti(s))
    If output is '2D' then transform is computed (mask 3D nifti(s) and return 2D np array)

    Args: 
        mask (str or nib obj): `../cerebellarGreySUIT3mm.nii`
        vol (str or nib obj or np array): can be str to nifti (4D or 3D) or nifti obj (4D or 3D) or 2d array (n_cols x n_voxels)
        output (str): '2D' or '3D'. default is '2D'
    Returns: 
        np array shape (n_cols x n_voxels) if output='2D'
        list of nifti obj(s) if output='3D' (multiple niftis are returned if `vol` is 4D nifti)
    """
    nifti_masker = NiftiMasker(standardize=False, mask_strategy='background', memory_level=2,
                            smoothing_fwhm=None, memory="nilearn_cache") 

    # load mask if it's a string
    if isinstance(mask, str):
        mask = nib.load(mask)
    elif isinstance(mask, nib.nifti1.Nifti1Image):
        pass

    # fit the mask
    nifti_masker.fit(mask)

    # masking is done in-place by nilearn..
    # so fullpath is never submitted directly to `nifti_masker`

    # check vol format
    if isinstance(vol, str):
        vol = nib.load(vol)
        fmri_masked = nifti_masker.transform(vol) #  (n_time_points x n_voxels)
    elif isinstance(vol, nib.nifti1.Nifti1Image):
        fmri_masked = nifti_masker.transform(vol) #  (n_time_points x n_voxels)
    elif isinstance(vol, np.ndarray): 
        try:
            num_vert, num_col = vol.shape
        except: 
            vol = np.reshape(vol, (1,len(vol)))
        nib_obj = nifti_masker.inverse_transform(vol)
        fmri_masked = nifti_masker.transform(nib_obj)

    # return masked vol
    if output=="2D":
        return fmri_masked
    elif output=="3D":
        nib_obj = nifti_masker.inverse_transform(fmri_masked)
        nib_objs = []
        for i in np.arange(nib_obj.shape[3]):
            nib_objs.append(image.index_img(nib_obj,i))
        return nib_objs

def coords_to_linvidxs(coords,vol_def,mask=False):
    """
    Maps coordinates to linear voxel indices
    INPUT:
        coords (3xN matrix or Qx3xN array):
            (x,y,z) coordinates
        vol_def (nibabel object):
            Nibabel object with attributes .affine (4x4 voxel to coordinate transformation matrix from the images to be sampled (1-based)) and shape (1x3 volume dimension in voxels)
        mask (bool):
            If true, uses the mask image to restrict voxels (all outside = -1)
    OUTPUT:
        linvidxs (N-array or QxN matrix):
            Linear voxel indices
        good (bool) boolean array that tells you whether the index was in the mask
    """
    mat = np.linalg.inv(vol_def.affine)

    # Check that coordinate transformation matrix is 4x4
    if (mat.shape != (4,4)):
        raise(NameError('Error: Matrix should be 4x4'))

    rs = coords.shape
    if (rs[-2] != 3):
        raise(NameError('Coords need to be a (Kx) 3xN matrix'))

    # map to 3xP matrix (P coordinates)
    ijk = mat[0:3,0:3] @ coords + mat[0:3,3:]
    ijk = np.rint(ijk).astype(int)
    
    if ijk.ndim<=2:
        i = ijk[0]
        j = ijk[1]
        k = ijk[2]
    elif ijk.ndim==3:
        i = ijk[:,0]
        j = ijk[:,1]
        k = ijk[:,2]

    # Now set the indices out of range to 
    good = (i>=0) & (i<vol_def.shape[0]) & (j>=0) & (j<vol_def.shape[1]) &  (k[2]>=0) & (k[2]<vol_def.shape[2])
    
    linindx = np.ravel_multi_index((i,j,k),vol_def.shape,mode='clip')

    if mask:
        M=vol_def.get_fdata().ravel()
        good = good & (M[linindx]>0)
    
    return linindx, good

def affine_transform(x1, x2, x3, M):
    """
    Returns affine transform of x
    Args:
        x1 (np-array):
            X-coordinate of original
        x2 (np-array):
            Y-coordinate of original
        x3 (np-array):
            Z-coordinate of original
        M (2d-array):
            4x4 transformation matrix
    Returns:
        x1 (np-array):
            X-coordinate of transform
        x2 (np-array):
            Y-coordinate of transform
        x3 (np-array):
            Z-coordinate of transform
    """
    y1 = M[0,0]*x1 + M[0,1]*x2 + M[0,2]*x3 + M[0,3]
    y2 = M[1,0]*x1 + M[1,1]*x2 + M[1,2]*x3 + M[1,3]
    y3 = M[2,0]*x1 + M[2,1]*x2 + M[2,2]*x3 + M[2,3]
    return (y1,y2,y3)

def coords_to_voxelidxs(coords,vol_def):
    """
    Maps coordinates to linear voxel indices
    Args:
        coords (3*N matrix or 3xPxQ array):
            (x,y,z) coordinates
        vol_def (nibabel object):
            Nibabel object with attributes .affine (4x4 voxel to coordinate transformation matrix from the images to be sampled (1-based)) and shape (1x3 volume dimension in voxels)
    Returns:
        linidxsrs (np.ndarray):
            N-array or PxQ matrix of Linear voxel indices
    """
    mat = np.array(vol_def.affine)

    # Check that coordinate transformation matrix is 4x4
    if (mat.shape != (4,4)):
        sys.exit('Error: Matrix should be 4x4')

    rs = coords.shape
    if (rs[0] != 3):
        sys.exit('Error: First dimension of coords should be 3')

    if (np.size(rs) == 2):
        nCoordsPerNode = 1
        nVerts = rs[1]
    elif (np.size(rs) == 3):
        nCoordsPerNode = rs[1]
        nVerts = rs[2]
    else:
        sys.exit('Error: Coordindates have %d dimensions, not supported'.format(np.size(rs)))

    # map to 3xP matrix (P coordinates)
    coords = np.reshape(coords,[3,-1])
    coords = np.vstack([coords,np.ones((1,rs[1]))])

    ijk = np.linalg.solve(mat,coords)
    ijk = np.rint(ijk)[0:3,:]
    # Now set the indices out of range to -1
    for i in range(3):
        ijk[i,ijk[i,:]>=vol_def.shape[i]]=-1
    return ijk

def sample_image(img,xm,ym,zm,interpolation):
    """
    Return values after resample image
    
    Args:
        img (Nifti image)
        xm (np-array)
            X-coordinate in world coordinates 
        ym (np-array)
            Y-coordinate in world coordinates
        zm (np-array)
            Z-coordinate in world coordinates 
        interpolation (int)
            0: Nearest neighbor
            1: Trilinear interpolation 
    Returns:
        value (np-array)
            Array contains all values in the image
    """
    im,jm,km = affine_transform(xm,ym,zm,np.linalg.inv(img.affine))

    if interpolation == 1:
        ir = np.floor(im).astype('int')
        jr = np.floor(jm).astype('int')
        kr = np.floor(km).astype('int')

        invalid = np.logical_not((im>=0) & (im<img.shape[0]-1) & (jm>=0) & (jm<img.shape[1]-1) & (km>=0) & (km<img.shape[2]-1))
        ir[invalid] = 0
        jr[invalid] = 0
        kr[invalid] = 0 
                
        id = im - ir
        jd = jm - jr
        kd = km - kr

        D = img.get_fdata()
        if D.ndim == 4:
            ns = id.shape + (1,)
        if D.ndim ==5: 
            ns = id.shape + (1,1)
        else:
            ns = id.shape
        
        id = id.reshape(ns)
        jd = jd.reshape(ns)
        kd = kd.reshape(ns)

        c000 = D[ir, jr, kr]
        c100 = D[ir+1, jr, kr]
        c110 = D[ir+1, jr+1, kr]
        c101 = D[ir+1, jr, kr+1]
        c111 = D[ir+1, jr+1, kr+1]
        c010 = D[ir, jr+1, kr]
        c011 = D[ir, jr+1, kr+1]
        c001 = D[ir, jr, kr+1]

        c00 = c000*(1-id)+c100*id
        c01 = c001*(1-id)+c101*id
        c10 = c010*(1-id)+c110*id
        c11 = c011*(1-id)+c111*id
        
        c0 = c00*(1-jd)+c10*jd
        c1 = c01*(1-jd)+c11*jd
        
        value = c0*(1-kd)+c1*kd
    elif interpolation == 0:
        ir = np.rint(im).astype('int')
        jr = np.rint(jm).astype('int')
        kr = np.rint(km).astype('int')

        ir, jr, kr, invalid = check_range(img, ir, jr, kr)
        value = img.get_fdata()[ir, jr, kr]
    
    # Kill the invalid elements
    if value.dtype is float:
        value[invalid]=np.nan
    else: 
        value[invalid]=0
    return value

def cortex_vol_to_surf(
    nifti, 
    pial_meshes,
    white_meshes, 
    hemispheres=['CortexLeft', 'CortexRight'],
    mask_img=None,
    column_names=None,
    zscore=False
    ):
    """Map vol to surf for cortex (left and right hemispheres)

    Uses subject-specific workbench connectome pial and white surfaces

    Args: 
        nifti (str or nib obj): full path to volume or nib obj
        pial_meshes (list of str): list of fullpath to pial surfaces (left and right hemis)
        white_meshes (list of str): list of fullpath to white surfaces (left and right hemis)
        hemispheres (list of str): default is ['CortexLeft', 'CortexRight']. Should be same length as `pial_meshes` and `white_meshes`
        mask_img (nib obj or str or None): 
        column_names (None or list): column names for gifti

    Returns: 
        giftis (list): gifti of shape (n_vertices x n_tasks)
    """
    # get mesh surfaces and textures
    if isinstance(nifti, str):
        img = nib.load(nifti)
    elif isinstance(nifti, nib.nifti1.Nifti1Image):
        img = nifti
    else:
        ValueError(print('nifti must be nifti object or full path to nifti image'))

    if mask_img is not None:
        if isinstance(mask_img, str):
            mask_img = nib.load(mask_img)
        elif isinstance(nifti, nib.nifti1.Nifti1Image):
            pass
    
    giis = []
    for (pial_mesh, white_mesh, hem) in zip(pial_meshes, white_meshes, hemispheres):
        # texture = vol_to_surf_sw(img, pial_fpath, white_fpath) # suzanne witt version
        texture = vol_to_surf(img, 
                            surf_mesh=pial_mesh, 
                            inner_mesh=white_mesh,
                            mask_img=mask_img,
                            ) # nilearn version

        if zscore:
            texture = stats.zscore(texture, axis=1, nan_policy='omit')

        # make gifti
        gii = make_func_gifti_cortex(data=texture, 
                                    anatomical_struct=hem, 
                                    column_names=column_names
                                    )
        
        giis.append(gii)

    return giis

def smooth_surface(
    func_fpath, 
    surf_fpath, 
    out_fpath=None, 
    kernel_sigma=2
    ):
    """Smooths functional data on cortical surfaces

    Args: 
        func_fpath (str): fpath to *.<hem>.func.gii
        surf_fpath (str): fpath to *.<hem>.inflated.surf.gii
        out_fpath (str or None): default is None, save in same directory as `func_fpath`
        kernel_sigma (int): default is 2
    """
    if not out_fpath: 
        func_dir = Path(func_fpath).parent
        func_name = Path(func_fpath).name
        out_fpath = os.path.join(func_dir, f'smooth_{func_name}')

    subprocess.run(['wb_command', '-metric-smoothing', surf_fpath, func_fpath, kernel_sigma, out_fpath])
    print(f'smoothing {func_fpath}')