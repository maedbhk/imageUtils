"""
Downloading Cortical neuroImaging datasets: atlas datasets
@author: maedbhking
A lot of the functionality was based on `nilearn.datasets.atlas`
https://github.com/nilearn/nilearn/blob/main/nilearn/datasets/atlas.py`
"""
import json
import requests

from imageUtils.utils import _get_dataset_dir, _fetch_files
from imageUtils._utils import fill_doc

@fill_doc
def fetch_yeo_2011(data_dir=None, base_url=None,
                    resume=True, verbose=1,
                    ):
    """Download and return file names for the Yeo et al. (2011) atlas
    The provided images are in gifti format
    Parameters
    ----------
    %(data_dir)s
    base_url : string, optional
        base_url of files to download (None results in default base_url).
    %(resume)s
    %(verbose)s
    Returns
    -------
    data : data dict
        Dictionary, contains keys:
            - data_dir: Absolute path of downloaded folder
            - files: list of string. Absolute paths of downloaded files on disk.
            - description: A short description of `data` and some references.
    Notes
    -----
    For more details, see
    https://github.com/DiedrichsenLab/fs_LR_32/tree/reformatting_mk/Yeo_2011
    """

    suffixes = ['.32k.R.label.gii', '.32k.L.label.gii']

    if base_url is None:
        base_url = ('https://github.com/DiedrichsenLab/fs_LR_32/raw/reformatting_mk/Yeo_2011')

    dataset_name = 'Yeo_2011'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)

    # get maps from `atlas_description.json`
    url = base_url + '/atlas_description.json'
    resp = requests.get(url)
    data_dict = json.loads(resp.text)

    # get map names and description
    maps = data_dict['Maps']
    fdescr = data_dict['LongDesc']

    # get filename for maps
    maps_full = []
    for map in maps:
        for suffix in suffixes:
            maps_full.append(f'{map}{suffix}')

    files = []
    for f in maps_full:
        files.append((f, base_url + '/' + f, {}))

    # get local fullpath(s) of downloaded file(s)
    fpaths = _fetch_files(data_dir, files, resume=resume, verbose=verbose)

    return dict({'data_dir': data_dir,
                'files': fpaths,
                'description': fdescr})
