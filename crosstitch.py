#yes i know it says crosstitch, not cross stitch
import numpy as np
from PIL import Image

#GET RID OF THESE GLOBAL THINGS 
original = Image.open("Dog.jpg") #Reading file
original_info = np.asarray(original) #Array-ifiying

def main():
    
    """
    Now we need to split the image into boxes
    In the new image a pixels colour will be the colour average of all pixels
    in its respective box
    """
    new_info = np.fromfunction(average_colour, original_info.shape, dtype=int)
    new_image = Image.fromarray(new_info, mode="RGB")
    new_image.show()
    print(original_info.shape)

def average_colour(x,y,z):
    return original_info[x,y,z]*21

if __name__ == "__main__":
    main()
    

