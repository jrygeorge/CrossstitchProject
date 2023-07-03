#yes i know it says crosstitch, not cross stitch
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
flag=True
block_size=0
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
# math to calculate block size, max to prevent it from being zero
block_size= int(max( w / (ct * pic_width), 1 ))
w=(w//block_size)*block_size
h=(h//block_size)*block_size
original=original.crop((0,0,w,h)) #cropping

def main():
    new_pic=original.copy().resize((int(w/block_size),int(h/block_size)),4).resize((w,h),4)
    new_pic.show()
    # .png has 4 properties per pixel R,G,B,Transparency, we're getting rid of the last one
    # .delete implicitly converts to array and deletes Transparency
    foo = np.delete(new_pic,3,2) if np.asarray(new_pic).shape[2]>3 else new_pic
    uncluster=np.reshape(foo,(w*h,3)) #long array with 3 properties each
    model = KMeans(n_clusters=15,n_init=8)
    cs_model=model.fit(uncluster)
    baz=cs_model.cluster_centers_
    gorb = np.array([baz[i].astype(int) for i in cs_model.labels_])
    lembas = np.reshape(gorb,(w,h,3))
    dot=Image.fromarray(lembas.astype(np.uint8), mode="RGB")
    dot.show()

if __name__ == "__main__":
    main()
    

