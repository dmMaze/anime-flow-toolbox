import os.path as osp
import os
from pathlib import Path
from typing import Union, List, Dict
import json
from collections.abc import MutableMapping
import gzip
import pillow_jxl
import cv2

import numpy as np
from PIL import Image


if np.lib.NumpyVersion(np.__version__) >= '2.0.0b1':
    NP_BOOL_TYPES = (np.bool, np.bool_)
    NP_FLOAT_TYPES = (np.float16, np.float32, np.float64)
else:
    NP_BOOL_TYPES = (np.bool_, np.bool8)
    NP_FLOAT_TYPES = (np.float_, np.float16, np.float32, np.float64)

NP_INT_TYPES = (np.int_, np.int8, np.int16, np.int32, np.int64, np.uint, np.uint8, np.uint16, np.uint32, np.uint64)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.ScalarType):
            if isinstance(obj, NP_BOOL_TYPES):
                return bool(obj)
            elif isinstance(obj, NP_FLOAT_TYPES):
                return float(obj)
            elif isinstance(obj, NP_INT_TYPES):
                return int(obj)
        return json.JSONEncoder.default(self, obj)


def json2dict(json_path: str):

    if json_path.endswith('.gz'):
        with gzip.open(json_path, 'rt', encoding='utf8') as f:
            metadata = json.load(f)
        return metadata

    with open(json_path, 'r', encoding='utf8') as f:
        metadata = json.loads(f.read())
    return metadata


def dict2json(adict: dict, json_path: str, compress=None):
    if compress is None:
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(adict, ensure_ascii=False, cls=NumpyEncoder))
    elif compress == 'gzip':
        if not json_path.endswith('.gz'):
            json_path += '.gz'
        with gzip.open(json_path, 'wt', encoding="utf8") as zipfile:
            json.dump(adict, zipfile, ensure_ascii=False, cls=NumpyEncoder)
    else:
        raise Exception(f'Invalid compression: {compress}')


IMG_EXT = {'.bmp', '.jpg', '.png', '.jpeg', '.webp'}
def find_all_imgs(img_dir, abs_path=False, sort=False):
    imglist = []
    dir_list = os.listdir(img_dir)
    for filename in dir_list:
        if filename.startswith('.'):
            continue
        file_suffix = Path(filename).suffix
        if file_suffix.lower() not in IMG_EXT:
            continue
        if abs_path:
            imglist.append(osp.join(img_dir, filename))
        else:
            imglist.append(filename)
    if sort:
        imglist.sort()
    return imglist


def find_all_files_recursive(tgt_dir: Union[List, str], ext: Union[List, set], exclude_dirs=None):
    if isinstance(tgt_dir, str):
        tgt_dir = [tgt_dir]

    if exclude_dirs is None:
        exclude_dirs = set()

    filelst = []
    for d in tgt_dir:
        for root, _, files in os.walk(d):
            if osp.basename(root) in exclude_dirs:
                continue
            for f in files:
                if Path(f).suffix.lower() in ext:
                    filelst.append(osp.join(root, f))

    return filelst


def find_all_imgs_recursive(tgt_dir, exclude_dirs=None):
    return find_all_files_recursive(tgt_dir, IMG_EXT, exclude_dirs)


VIDEO_EXT = {'.mp4', '.gif', '.webm', '.avif', '.mkv'}
def find_all_videos_recursive(tgt_dir,exclude_dirs=None):
    return find_all_files_recursive(tgt_dir, VIDEO_EXT, exclude_dirs)


def load_exec_list(exec_list, rank=None, world_size=None, check_exist=False, to_imgs=False):
    '''
    split exec_list by rank and world_size if available
    '''

    if isinstance(exec_list, str):
        if exec_list.endswith('.json') or exec_list.endswith('.json.gz'):
            exec_list = json2dict(exec_list)
        else:
            with open(exec_list, 'r', encoding='utf8') as f:
                exec_list = f.read().split('\n')
    else:
        assert isinstance(exec_list, list)

    if rank is not None and world_size is not None:
        nexec = len(exec_list) // world_size
        nstart = nexec * rank
        if rank == world_size - 1:
            exec_list = exec_list[nstart:]
        else:
            exec_list = exec_list[nstart:nstart+nexec]

    if to_imgs:
        _exec_list = []
        for p in exec_list:
            if osp.isdir(p):
                _exec_list += find_all_imgs(p, sort=True, abs_path=True)
            else:
                _exec_list.append(p)
        exec_list = _exec_list

    if check_exist:
        nlist = []
        for p in exec_list:
            if osp.exists(p):
                nlist.append(p)
        return nlist
    else:
        return exec_list


def load_image(imgp: str, mode="RGB", output_type='numpy'):
    """
    return RGB image as output_type
    """
    img = Image.open(imgp).convert(mode)
    if output_type == 'numpy':
        img = np.array(img)
        if len(img.shape) == 2:
            img = img[..., None]
    return img


def flatten_dict(dictionary, parent_key='', separator='_'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def imglist2imgrid(imglist, cols=4):
    current_row = []
    grid = []
    grid.append(current_row)
    if len(imglist) < cols:
        cols = len(imglist)
    for ii, img in enumerate(imglist):
        current_row.append(img)
        if len(current_row) >= cols and ii != len(imglist) - 1:
            current_row = []
            grid.append(current_row)
    if len(grid) > 1 and len(grid[-1]) < cols:
        for ii in range(cols - len(grid[-1])):
            grid[-1].append(np.full_like(grid[-1][-1], fill_value=255))
    if len(grid) > 1:
        for ii, row in enumerate(grid):
            grid[ii] = np.concatenate(row, axis=1)
        grid = np.concatenate(grid, axis=0)
    else:
        grid = np.concatenate(grid[0], axis=1)
    return grid


def resize_keepasp(im, new_shape=640, scaleup=True, interpolation=cv2.INTER_LINEAR, stride=None):
    shape = im.shape[:2]  # current shape [height, width]

    if new_shape is not None:
        if not isinstance(new_shape, tuple):
            new_shape = (new_shape, new_shape)
    else:
        new_shape = shape

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    if stride is not None:
        h, w = new_unpad
        if h % stride != 0 :
            new_h = (stride - (h % stride)) + h
        else :
            new_h = h
        if w % stride != 0 :
            new_w = (stride - (w % stride)) + w
        else :
            new_w = w
        new_unpad = (new_h, new_w)
        
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=interpolation)
    return im