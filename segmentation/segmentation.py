"""
I made use of this piece of code as a starting point to do the segmentation:
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html

With this code one can take a satellite image with mapbox and segment nature out of the picture.

We are using mapbox. Note that in using it in high amounts isn't for free!
Check the pricing: https://www.mapbox.com/pricing/

We also implemented some multithreading and multiprocessing to play around with. ;-)

It is far from perfect, but it is a start. Have fun updating it and let me know if you get something beautiful out of this.
"""

import pandas as pd
import matplotlib.pyplot as plt
import cv2
import requests
import shutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from os import listdir
from os.path import isfile, join

#Settings
thread_workers = 4
process_workers = 4
zoom = 17
resolution = '500x500'
mapboxAccessToken = '<get your own token>' #----> https://docs.mapbox.com/help/how-mapbox-works/access-tokens/
mapboxTilesetId = 'mapbox.satellite'

df = pd.read_csv('test_data.csv')
data = list(zip(df['id'], df['lengtegraad'], df['breedtegraad']))

def save_image(d):
    url = f'https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{d[1]},{d[2]},{zoom},0,0/{resolution}?access_token={mapboxAccessToken}'
    response = requests.get(url, stream=True)
    with open(f'segmentation/images/{d[0]}.png', 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)


def segment_1(photo='', show_steps=False):
    if not photo:
        return {"error": "no image to segment"}

    image = cv2.imread(f'segmentation/images/{photo}')

    if show_steps:
        plt.figure(figsize=(20, 10))
        plt.axis("off")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if show_steps:
        plt.figure(figsize=(20, 10))
        plt.imshow(thresh)
        plt.axis('off')
        plt.show()

    """
    The following piece of code is copy-paste from 
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html
    written by Alexander Mordvintsev & Abid K. Revision
    The numbers are fine-tuned to certain satellite images to get a better result.
    """
    # remove noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers += 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]

    if show_steps:
        plt.figure(figsize=(20, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    return image

#multi processing
def segmentation():
    photos = [f for f in listdir('images') if isfile(join('images', f))]
    with ProcessPoolExecutor() as executor:
        segmented_images = executor.map(segment_1, photos[0:10])
    return list(segmented_images)

if __name__ == '__main__':
    # Threading the satellite data
    with ThreadPoolExecutor(max_workers=thread_workers) as executor:
        set(executor.map(save_image, data))

    for i in segmentation():
        plt.figure(figsize=(20,10))
        plt.imshow(i)
        plt.axis('off')
        plt.show()