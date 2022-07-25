# NOTE: This script was used to download the large sized Google Maps images (1200x1200) using their (paid) static maps API. 
# The API_KEY and MAP_ID were removed but can be substituted by your own. Google provides a free 300$ per month for its API 
# and this is more than enough for this project (0.77$)

import os
from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np

if (not os.path.isdir("large")): os.mkdir("large")

# Parameters:
API_KEY = "" # insert your own api key
MAP_ID = "" # insert your own customized map id that only shows roads in white, rest is black.
Sample_grid_around_centers = True # Samples 9 images in a 3x3 grid around each center, instead of just 1 at center.


width = 1200 # 600 * 2
height = 1250 # 625 * 2
bottom_crop = 50 # to remove the logo
# This results in images of size 1200x1200 which can be nicely sliced to 400x400 images for training.


# Just some cities that show satellite images that look close enough to the sample images we were given.
# These are the region numbers, starting with 0.
centers = [
    (47.3713447,8.5212367), # Zürich 1
    (47.4179587,8.5082851), # Zürich 2
    (47.4203427,9.3785760), # St. Gallen 1
    (47.4420347,9.4125922), # St. Gallen 2
    (46.9520224,7.4492586), # Bern 1
    (46.9404830,7.4256420), # Bern 2
    (46.0147154,8.9577235), # Lugano 1
    (46.0068545,8.9321992), # Lugano 2
    (46.2016757,6.1446273), # Genf 1
    (46.1790658,6.1126206), # Genf 2
    (37.7502556,-122.1532643), # San Francisco 1
    (37.7226635,-122.4700353), # San Francisco 2
    (40.9158342,-74.1818808), # New York 1
    (40.6358011,-74.1154629), # New York 2
    (39.083321,-94.5334974), # Kansas City 1
    (39.1283702,-94.6504097), # Kansas City 2
    (33.596416,-112.1508858), # Phoenix 1
    (33.4741494,-111.9285227), # Phoenix 2
    (35.4965608,-97.5337639), # Oklahoma 1
    (35.5612061,-97.5589818) # Oklahoma 2
]

# Sample 3x3 adjacent area around the selected center.
# Translations for different adjacent maps to not overlap is about 0.0065 in both directions (approx. for northern hemisphere cities at least).
centers_extended = []
if (Sample_grid_around_centers):
    for c in centers:
        for dy in [-1.0,0.0,1.0]:
            for dx in [-1.0,0.0,1.0]:
                centers_extended.append(f"{c[0]+dx*0.007:3.7f},{c[1]+dy*0.007:3.7f}")
else:
    for c in centers:
        centers_extended.append(f"{c[0]:3.7f},{c[1]:3.7f}")


for image_index in range(0,len(centers_extended)):
    region_index =  int(np.floor(image_index/9))
    grid_index = image_index%9
    if(not Sample_grid_around_centers):
        region_index = image_index
        grid_index = 4

    filename_sat = f"large/gmap_region_{region_index:02}{grid_index}_sat.png"
    filename_seg = f"large/gmap_region_{region_index:02}{grid_index}_seg.png"

    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    url_sat = base_url+"?center="+centers_extended[image_index]+"&zoom=17&size=600x625&scale=2&maptype=satellite&key="+API_KEY
    url_seg = base_url+"?center="+centers_extended[image_index]+"&zoom=17&size=600x625&scale=2&maptype=roadmap&key="+API_KEY+"&map_id="+MAP_ID

    if(not os.path.exists(filename_sat)):
        buffer_sat = BytesIO(request.urlopen(url_sat).read())
        image_sat = Image.open(buffer_sat, formats=["png"])
        buffer_seg = BytesIO(request.urlopen(url_seg).read())
        image_seg = Image.open(buffer_seg, formats=["png"])

        # 1. crop the google logo off (about 45 pixels at the bottom). For safety we will cut off 50 pixels)
        image_sat = image_sat.crop((0,0,width,height-bottom_crop))
        image_seg = image_seg.crop((0,0,width,height-bottom_crop))
        # -> large images now 1200 x 1200

        # save google maps requested images in large format to be shared
        image_sat.save(filename_sat)
        image_seg.save(filename_seg)
    else:
        print(f"Large region {region_index:02}{grid_index} already downloaded!")