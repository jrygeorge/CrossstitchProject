#yes i know it says crosstitch, not cross stitch
import numpy as np
from PIL import Image
flag=True
while flag:
    try:
        bsize = int(input("Enter block size : "))
    except:
        print("Invalid Input")
    else:
        flag = not flag
original = Image.open("Dog.jpg") #Reading file
w, h = original.size
w=(w//bsize)*bsize
h=(h//bsize)*bsize
original=original.crop((0,0,w,h))
original_info = np.asarray(original) #Array-ifiying
new_info=original_info.copy()

def main():
    # new_info = np.fromfunction(average_colour, original_info.shape, dtype=int)
    # this way isnt working for some reason, ive deleted the function its calling
    # new_image = Image.fromarray(new_info, mode="RGB")
    jam=list(range(0,h,bsize))
    for i in range(0,h,bsize):
        for j in range(0,w,bsize):
            #print("{} {}".format(i,j))
            new_info[i:None if i+bsize>h else i+bsize,j:None if j+bsize>w else j+bsize]=(
            new_info[i:None if i+bsize>h else i+bsize,j:None if j+bsize>w else j+bsize].mean(axis=(0,1)))
    
    bim=Image.fromarray(new_info,mode="RGB")
    bim.show()

if __name__ == "__main__":
    main()
    

