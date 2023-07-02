#yes i know it says crosstitch, not cross stitch
import numpy as np
from PIL import Image
flag=True
while flag:
    try:
        bsize = int(str.strip(input("Enter block size : ")))
    except:
        print("Invalid Input")
    else:
        flag = not flag #break

original = Image.open("Dog.jpg") #Reading file
w, h = original.size
w=(w//bsize)*bsize
h=(h//bsize)*bsize
original=original.crop((0,0,w,h)) #cropping

def main():
    new_pic=original.copy().resize((int(w/bsize),int(h/bsize)),4).resize((w,h),4)
    new_pic.show()

if __name__ == "__main__":
    main()
    

