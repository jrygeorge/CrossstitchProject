#yes i know it says crosstitch, not cross stitch
from PIL import Image
import numpy as np
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

original = Image.open("bob.png") #Reading file
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
    foo=np.delete(new_pic,3,2) if len(np.asarray(new_pic))>3 else new_pic
    print(foo[200,200])
    uncluster=np.reshape(foo,(w*h,3)) #long array with 3 properties each
    print(uncluster[200])


if __name__ == "__main__":
    main()
    

