from PIL import Image, ImageDraw, ImageOps
import numpy as np
from sklearn.cluster import KMeans
import time
class cross_pattern:
    def __init__(self,original) -> None:
        self.original = original
        # choosing to derive w,h from self.original in each function where its needed
        # rather than using two extra instance variables, i just think its a bit ugly

    def dimensions(self, ct, pic_width):
        w, h = self.original.size
        # number of blocks/stitches across the width
        # min to make sure block number doesn't become huge and waste space
        self.block_num = min( ct * pic_width , w)
        self.pixelated = self.original.copy().resize((int(self.block_num),int(self.block_num*h/w)),4)
        # .png has 4 properties per pixel R,G,B,Alpha , we're getting rid of the last one
        self.pixelated.show()
        self.pixelated = (np.asarray(self.pixelated)
                          [:,:,: 3 if np.asarray(self.pixelated).shape[2]>3 else None])

    def rgb_cluster(self,cluster_num):
        w, h = self.original.size
        unclustered = (np.reshape(self.pixelated, #long array with 3 properties each
                                  (int(self.block_num)*int(self.block_num*h/w),3) )) 
        #K-means, n_init=1 as we're using k-means++, could just n_init=auto it instead
        model = KMeans(n_clusters=cluster_num,n_init=1)
        cs_model=model.fit(unclustered)
        # mapping computed means (cluster_centers) to each pixel (labels)
        clustered = np.array([cs_model.cluster_centers_[i] for i in cs_model.labels_])
        """
        For some insane reason I don't understand YET, when number of blocks is very large,
        when converting the array back to an image using C-order, the image gets super stretched
        and turns into stripes, like when you're rewinding a VHS.
        When block number is small it works perfectly fine. Will look into it later.
        To fix this I have :
        1. Used F order
        2. Flipped along horiz axis, and rotated 270 deg

        """
        self.clust_array = np.reshape(clustered,
                                 (int(self.block_num),int(self.block_num*h/w),3), order="F")
        new_image = (Image.fromarray(self.clust_array.astype(np.uint8), mode="RGB")
                     .transpose(Image.FLIP_TOP_BOTTOM)
                     .transpose(Image.ROTATE_270))
        return new_image

    def create_grid_symbols(self,img,x_start,y_start,x_end,y_end):
        #new pixel size, border thickness, extra thickness for every 10th line
        f, border, ten = 20, 4, 2
        # box width = pixels*f + width of vertical lines + extra for every 10th line and each border
        # similar for height
        border = 4
        ten = 2 #extra thickness we're adding for every 10th line i.e. if its zero, thickness will be 1
        grid = Image.new("RGB", ((x_end-x_start)*(f+1)+1+2*(border-1)+ten*((x_end-x_start)//10),
                                 (y_end-y_start)*(f+1)+1+2*(border-1)+ten*((y_end-y_start)//10)), (0, 0, 0))
        draw = ImageDraw.Draw(grid)
        for i in range(x_end - x_start):
            for j in range (y_end - y_start):
                x, y = border+f*i+i+(i//10)*ten, border+f*j+j+(j//10)*ten
                draw.rectangle((x, y, x-1+f, y-1+f),
                                fill=tuple(int(bob) for bob in self.clust_array[y_start+j,x_start+i]) )
        grid = grid.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.ROTATE_270)
        return grid
    
    def create_grid(self):
        f, border, ten = 20, 4, 2
        w, h, rgb = self.pixelated.shape
        grid = Image.new("RGB", ((w)*(f+1)+1+2*(border-1)+ten*((w)//10),
                                 (h)*(f+1)+1+2*(border-1)+ten*((h)//10)), (0, 0, 0))
        draw = ImageDraw.Draw(grid)
        for i in range(w):
            for j in range (h):
                x, y = border+f*i+i+(i//10)*ten, border+f*j+j+(j//10)*ten
                draw.rectangle((x, y, x-1+f, y-1+f),
                                fill=tuple(int(bob) for bob in self.clust_array[j,i]) )
        grid = grid.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.ROTATE_270)
        return grid
    
    def splitter(self):
        split=[]
        h, w, rgb = self.pixelated.shape
        for i in range(0,int(self.block_num),60):
            for j in range(0, int(self.block_num),40): # finds the the top left pixel of every section
                split.append(self.create_grid_symbols( # takes a 40*60 block from image starting at that ^ pixel
                    self.clust_array[i:i+60 if i+60<w else None,j:j+40 if j+40<h else None],
                                        i, j, i+60 if i+60<w else w-1, j+40 if j+40<h else h-1 ))
        return split
    
        
  
if __name__ == "__main__":
    t1=time.perf_counter()
    
    img = Image.open("Dog.jpg") #Reading file
    ct = int(str.strip(input("Enter ct (14, 18 etc) : ")))
    pic_width = float(str.strip(input("Enter width in inches : ")))
    clus = int(str.strip(input("Enter number of colours : ")))

    Dog = cross_pattern(img)
    Dog.dimensions(ct,pic_width)
    result = Dog.rgb_cluster(clus)
    bimp=Dog.splitter()
    for k in bimp:
        k.show()
    l=Dog.create_grid()
    l.show()

    t2=time.perf_counter()
    print(f"Time elapsed {t2-t1:0.5f}")
    

