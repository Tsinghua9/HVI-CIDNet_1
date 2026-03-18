import glob
from tqdm import tqdm
from PIL import Image
import imquality.brisque as brisque
import numpy as np
import skimage.transform
import warnings
from loss.niqe_utils import *
import argparse


def _score_brisque_safe(image, kernel_size=7, sigma=7/6):
    # Use imquality's feature/predict but avoid passing 2D images into Brisque again.
    b1 = brisque.Brisque(image, kernel_size=kernel_size, sigma=sigma)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            down = skimage.transform.rescale(
                b1.image,
                1 / 2,
                order=2,
                mode="constant",
                anti_aliasing=False,
                channel_axis=None,
            )
        except TypeError:
            down = skimage.transform.rescale(
                b1.image,
                1 / 2,
                order=2,
                mode="constant",
                anti_aliasing=False,
                multichannel=False,
            )
    if down.ndim == 2:
        down = np.stack([down] * 3, axis=-1)
    b2 = brisque.Brisque(down, kernel_size=kernel_size, sigma=sigma)
    features = np.concatenate([b1.features, b2.features])
    scaled_features = brisque.scale_features(features)
    return brisque.predict(scaled_features)


def metrics(im_dir):
    avg_niqe = 0
    n = 0
    avg_brisque = 0
        
    for item in tqdm(sorted(glob.glob(im_dir))):
        n += 1
        
        im1 = Image.open(item).convert('RGB')
        im1 = np.array(im1)
        if im1.ndim == 2:
            im1 = np.stack([im1] * 3, axis=-1)
        score_brisque = _score_brisque_safe(im1)
        score_niqe = calculate_niqe(im1)
        
        
        avg_brisque += score_brisque
        avg_niqe += score_niqe

        torch.cuda.empty_cache()
    
    avg_brisque = avg_brisque / n
    avg_niqe = avg_niqe / n
    return avg_niqe, avg_brisque

if __name__ == '__main__':

    eval_parser = argparse.ArgumentParser(description='Eval')
    eval_parser.add_argument('--DICM', action='store_true', help='output DICM dataset')
    eval_parser.add_argument('--LIME', action='store_true', help='output LIME dataset')
    eval_parser.add_argument('--MEF', action='store_true', help='output MEF dataset')
    eval_parser.add_argument('--NPE', action='store_true', help='output NPE dataset')
    eval_parser.add_argument('--VV', action='store_true', help='output VV dataset')
    ep = eval_parser.parse_args()

    if ep.DICM:
        im_dir = './output/DICM/*.jpg'

    elif ep.LIME:
        im_dir = './output/LIME/*.bmp'

    elif ep.MEF:
        im_dir = './output/MEF/*.png'

    elif ep.NPE:
        im_dir = './output/NPE/*.jpg'

    elif ep.VV:
        im_dir = './output/VV/*.jpg'


    avg_niqe, avg_brisque = metrics(im_dir)
    print(avg_niqe)
    print(avg_brisque)
