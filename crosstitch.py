from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
flag=True
while flag:
    ct=0
    pic_width=0
    try:
        ct = int(str.strip(input("Enter ct 14,18 etc : ")))
        pic_width = float(str.strip(input("Enter width in inches : ")))
    except:
        print("Invalid Input")
    else:
        flag = not flag #break

original = Image.open("Dog.jpg") #Reading file
w, h = original.size
# number of blocks/stitches across the width
# min to make sure block number doesn't become huge and waste space
block_num = min( ct * pic_width , w) 

def main():
    pixelated = original.copy().resize((int(block_num),int(block_num*h/w)),4)
    pixelated.show()
    # .png has 4 properties per pixel R,G,B,Alpha , we're getting rid of the last one
    # .delete implicitly converts to array and deletes Alpha/Transparency
    pixelated = np.asarray(pixelated)[:,:,: 3 if np.asarray(pixelated).shape[2]>3 else None]

    unclustered = np.reshape(pixelated,(int(block_num)*int(block_num*h/w),3)) #long array with 3 properties each
    #K-means
    model = KMeans(n_clusters=12,n_init=8)
    cs_model=model.fit(unclustered)
    # mapping computed means (cluster_center) to each pixel (labels)
    clustered = np.array([cs_model.cluster_centers_[i] for i in cs_model.labels_])
    """
    For some insane reason I don't understand YET, when number of blocks is very large,
    when converting the array back to an image using C-order, the image gets super stretched
    and turns into stripes, like when you're rewinding a VHS.
    When block size is large it works perfectly fine. Will look into it later.
    To fix this I have :
    1. Used F order
    2. Flipped along horiz axis, and rotated 270 deg

    """
    clust_array = np.reshape(clustered,(int(block_num),int(block_num*h/w),3), order="F")
    new_image = Image.fromarray(clust_array.astype(np.uint8), mode="RGB")
    new_image.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.ROTATE_270).show()

if __name__ == "__main__":
    main()

