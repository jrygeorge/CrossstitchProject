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
original=original.crop((0,0,w,h))
original_info = np.asarray(original) #Array-ifiying

def main():
    new_pic=original.copy().resize((int(w/bsize),int(h/bsize)),4).resize((w,h),4)
    new_pic.show()
    new_info=np.asarray(new_pic)
    #below 5 lines tests if the new format is producing pure squares, not squares with other colours
    #reshaping to single line of pixels with 3 unit RGB array in each cell
    strim=new_info.copy().reshape((w*h,3))

    #display 500th pixel
    print(strim[500])
    #display 500th pixel directly from image
    #result : theyre the same :)
    print(new_pic.getpixel((500,1)))

    #finding all unique pixels/colours and displaying the number
    #if there are more colours than boxes (manual count), somethings wrong
    #result : more boxes than colours :)
    print(np.unique(strim,axis=0).shape)
    print(strim.shape)

if __name__ == "__main__":
    main()
    

