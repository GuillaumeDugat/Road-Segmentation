import os
from PIL import Image
import numpy as np
import glob
import time


def process_images(
    other_rotations,
    right_angles,
    divisions,
    dir="training_google",
):
    # Naming convention for sliced images:
    # {region_index:2}{grid_index:1}{flipped:1}{rotation_deg:3}{slice_index:2}_{"sat" or "seg"}.png

    # create directories for new dataset
    os.makedirs(dir, exist_ok=True)
    os.makedirs(os.path.join(dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(dir, "groundtruth"), exist_ok=True)

    sat_images = glob.glob(os.path.join("large", "*_sat.png"))

    starttime = time.perf_counter()
    image_count = 0

    width_offset = int((1200.0 - 400.0) / divisions - 1)
    height_offset = int((1200.0 - 400.0) / divisions - 1)

    # Slice each image into many small images:
        # per large image
        # 1. rotate the large image #rotations
        # 2. try to fit as many 20% shifted patches of size 400x400
        # 3. flip every possible patch to create a second mirrored copy
    for large_image in range(0,len(sat_images)):
        print(large_image, " ",end="")
        region_index = int(sat_images[large_image][-11:-9])
        grid_index = int(sat_images[large_image][-9:-8])

        filename_sat = sat_images[large_image]
        filename_seg = f"gmap_region_{region_index:02}{grid_index}_seg.png"

        image_sat = Image.open(filename_sat)
        image_seg = Image.open(os.path.join("large", filename_seg))

        for rot in other_rotations:
            rot_sat = image_sat.rotate(rot)
            rot_seg = image_seg.rotate(rot)

            # Compute a left and right linear function that gives the usable area of defined pixels when rotated.
            angle = rot%90
            costheta = np.cos(angle*np.pi/180)
            sintheta = np.sin(angle*np.pi/180)
            R = np.array([[costheta, sintheta], [-sintheta, costheta]]).T
            corners = np.array([[0.0,0.0], [1200.0,0.0], [0.0,1200.0], [1200.0,1200.0]]) - 600.0
            rotated_corners = np.matmul(corners,R) + 600.0

            # line 1 left side
            r = rotated_corners[1] - rotated_corners[0]
            uslope = r[0]/r[1]
            uoffset = abs(rotated_corners[0][0]) + rotated_corners[0][1]*uslope

            # line 2 left side
            r2 = rotated_corners[2] - rotated_corners[0]
            lslope = r2[0]/r2[1]
            loffset = abs(rotated_corners[0][0]) + rotated_corners[0][1]*lslope

            # line 3 right side
            r3 = rotated_corners[3] - rotated_corners[1]
            uslope2 = r3[0]/r3[1]
            uoffset2 = rotated_corners[1][0] - rotated_corners[1][1]*uslope2

            # line 4 right side
            r4 = rotated_corners[3] - rotated_corners[2]
            lslope2 = r4[0]/r4[1]
            loffset2 = rotated_corners[2][0] - rotated_corners[2][1]*lslope2

            for yoffset in range(0,divisions):
                top_offset = height_offset*yoffset

                y1 = top_offset
                ux = y1 * uslope - uoffset
                lx = y1 * lslope - loffset
                ux2 = y1 * uslope2 + uoffset2
                lx2 = y1 * lslope2 + loffset2
                xmint = max(0.0, ux, lx)
                xmaxt = min(1200.0, ux2, lx2)

                y2 = top_offset + 400.0
                ux2 = y2 * uslope - uoffset
                lx2 = y2 * lslope - loffset
                ux22 = y2 * uslope2 + uoffset2
                lx22 = y2 * lslope2 + loffset2
                xminb = max(0.0, ux2, lx2)
                xmaxb = min(1200.0, ux22, lx22)

                xstart = max(xmint, xminb)
                xend = min(xmaxt, xmaxb)

                for xiter, left_offset in enumerate(range(int(xstart), int(xend)-400, width_offset)):
                    window_bounds = (0+left_offset,0+top_offset,400+left_offset,400+top_offset)
                    
                    partial_sat = rot_sat.crop(window_bounds)
                    partial_seg = rot_seg.crop(window_bounds)

                    i = xiter + divisions*yoffset

                    # convert from P mode to RGB/L
                    partial_sat = partial_sat.convert('RGB')
                    partial_seg = partial_seg.convert('L')

                    partial_sat.save(os.path.join(dir, "images", f"{region_index:02}{grid_index}0{rot:03}{i:02}_sat.png"))
                    partial_seg.save(os.path.join(dir, "groundtruth", f"{region_index:02}{grid_index}0{rot:03}{i:02}_seg.png"))
                    
                    partial_sat_flip = partial_sat.transpose(Image.FLIP_LEFT_RIGHT)
                    partial_seg_flip = partial_seg.transpose(Image.FLIP_LEFT_RIGHT)
                    partial_sat_flip.save(os.path.join(dir, "images", f"{region_index:02}{grid_index}1{rot:03}{i:02}_sat.png"))
                    partial_seg_flip.save(os.path.join(dir, "groundtruth", f"{region_index:02}{grid_index}1{rot:03}{i:02}_seg.png"))
                    image_count += 4
        
        # For right angles the entire area can be used, so no unnecessary calculations wasted.
        for rot in right_angles:
            rot_sat = image_sat.rotate(rot)
            rot_seg = image_seg.rotate(rot)

            for yoffset in range(0,divisions):
                top_offset = height_offset*yoffset

                for xoffset in range(0,divisions):
                    left_offset = width_offset*xoffset
                    
                    window_bounds = (0+left_offset,0+top_offset,400+left_offset,400+top_offset)
                    
                    partial_sat = rot_sat.crop(window_bounds)
                    partial_seg = rot_seg.crop(window_bounds)

                    i = xoffset + divisions*yoffset

                    # convert from P mode to RGB/L
                    partial_sat = partial_sat.convert('RGB')
                    partial_seg = partial_seg.convert('L')

                    partial_sat.save(os.path.join(dir, "images", f"{region_index:02}{grid_index}0{rot:03}{i:02}_sat.png"))
                    partial_seg.save(os.path.join(dir, "groundtruth", f"{region_index:02}{grid_index}0{rot:03}{i:02}_seg.png"))
                    
                    partial_sat_flip = partial_sat.transpose(Image.FLIP_LEFT_RIGHT)
                    partial_seg_flip = partial_seg.transpose(Image.FLIP_LEFT_RIGHT)
                    partial_sat_flip.save(os.path.join(dir, "images", f"{region_index:02}{grid_index}1{rot:03}{i:02}_sat.png"))
                    partial_seg_flip.save(os.path.join(dir, "groundtruth", f"{region_index:02}{grid_index}1{rot:03}{i:02}_seg.png"))
                    image_count += 4


    endtime = time.perf_counter()
    print(f"\nTook: {endtime-starttime:.2}s to create {image_count} images")