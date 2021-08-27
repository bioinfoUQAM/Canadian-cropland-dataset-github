#note
# conda install -c conda-forge rasterio
# conda install -c anaconda pillow
# conda
# Etienne Lord, 29 Mars 2021

import sys
from optparse import OptionParser
import os
import cv2
import zipfile
import tempfile
import shutil
import rasterio
from rasterio.plot import show
import numpy as np
from PIL import Image, ImageEnhance

def RGB_spliter(file):
    print(str(file))
    with tempfile.TemporaryDirectory() as temp_dir:
        original, oext=os.path.splitext(str(file))
        B=np.zeros((10,10)) #Placeholser
        R=np.zeros((10,10))
        G=np.zeros((10,10))
        with zipfile.ZipFile(file,"r") as zip_ref:
            zip_ref.extractall(str(temp_dir))
            for filename in os.listdir(temp_dir):
                root, ext = os.path.splitext(str(filename))
                if root.endswith('B2'):
                    with rasterio.open(os.path.join(temp_dir,filename)) as im:
                        B=im.read(1)
                        B=(B * 255).astype(np.uint8)
                        #print (filename)
                if root.endswith('B3'):
                    with rasterio.open(os.path.join(temp_dir,filename)) as im:
                         G=im.read(1)
                         G=(G * 255).astype(np.uint8)
                if root.endswith('B4'):
                    with rasterio.open(os.path.join(temp_dir,filename)) as im:
                        R=im.read(1)
                        R=(R * 255).astype(np.uint8)
                        #print (filename)
            channels=np.empty([3,B.shape[0],B.shape[1]],dtype=np.uint8)
            channels2=np.empty([B.shape[0],B.shape[1],3],dtype=np.uint8)
            for index,image in enumerate([R,G,B]):
                channels2[:,:,index] = image

            im=Image.fromarray(channels2, 'RGB')

            # Enhance the image
            enhancer = ImageEnhance.Contrast(im)
            factor = 1.5 #gives original image
            im_output = enhancer.enhance(factor)
            enhancer = ImageEnhance.Brightness(im_output)
            im_output = enhancer.enhance(factor)

            # save the image as a png file
            im_output.save("{0}_RGB.png".format(original))

def three_channel_spliter(file, ending):
    print(str(file))
    with tempfile.TemporaryDirectory() as temp_dir:
        original, oext=os.path.splitext(str(file))
        C1=np.zeros((10,10)) #Placeholser
        with zipfile.ZipFile(file,"r") as zip_ref:
            zip_ref.extractall(str(temp_dir))
            for filename in os.listdir(temp_dir):
                root, ext = os.path.splitext(str(filename))
                if root.endswith("." + ending):
                    with rasterio.open(os.path.join(temp_dir,filename)) as im:
                        C1=im.read(1)
                        C1=(C1*255).astype(np.uint8)

                        #print (filename)
            channels=np.empty([3,C1.shape[0],C1.shape[1]],dtype=np.uint8)
            channels2=np.empty([C1.shape[0],C1.shape[1],3],dtype=np.uint8)
            for index, image in enumerate([C1, C1, C1]):
                channels2[:,:,index] = image

            im=Image.fromarray(channels2, 'RGB')
            print("original filename: ", original)

            im.save("{filename}_{extension}.png".format(filename=original, extension = ending))


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-i", "--input_image", dest="input_image", action="store", type="string",
                      help="Path of the image in zip format")
    (options, args) = parser.parse_args()
    if not options.input_image:   # if filename is not given
        parser.error('Error: path to image in ZipFormat must be specified. Pass -i to command line')
    spliter(options.input_image)


# Press the green button in the gutter to run the script.
