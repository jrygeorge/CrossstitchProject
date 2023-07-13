from PIL import Image, ImageDraw, ImageOps, ImageFont
import numpy as np
from sklearn.cluster import KMeans
import time
from math import sqrt
import pandas as pd
class cross_pattern:
    def __init__(self, original, ct, pic_width) -> None:
        self.original = original
        self.rgb = pd.read_csv("dmc_rgb.csv")

        # choosing to derive w,h from self.original in each function where its needed
        # rather than using two extra instance variables, i just think its a bit ugly
        w, h = self.original.size
        # number of blocks/stitches across the width
        # min to make sure block number doesn't become huge and waste space
        self.block_num = min( ct * pic_width , w)
        self.pixelated = self.original.copy().resize((int(self.block_num),int(self.block_num*h/w)),4)
        # .png has 4 properties per pixel R,G,B,Alpha , we're getting rid of the last one
        # its faster this way than using .delete
        self.pixelated.show()
        self.pixelated = (np.asarray(self.pixelated)
                          [:,:,: 3 if np.asarray(self.pixelated).shape[2]>3 else None])

    def rgb_cluster(self,cluster_num):
        h, w, rgb = self.pixelated.shape
        unclustered = (np.reshape(self.pixelated, #long array with 3 properties each
                                  (h*w,3) )) 
        #K-means, n_init=1 as we're using k-means++, could just n_init=auto it instead
        model = KMeans(n_clusters=cluster_num,n_init=1)
        cs_model=model.fit(unclustered)
        symb = "ABCDEFGHIJKLMNOPQRSTUVWXYZæ234567890£$&]@?>¥Ø≡<#¶§ßabcdefghijklmnopqrstuvwxyz1234567890£$&]@?>¥Ø≡<#"
        # mapping computed means (cluster_centers) to each pixel (labels)
        clustered = np.array([cs_model.cluster_centers_[i] for i in cs_model.labels_])
        self.Map = pd.DataFrame({"Original" : [tuple(i) for i in np.unique(clustered, axis =0)],
                                 "Recolour" : [self.recolour(tuple(i)) for i in np.unique(clustered, axis =0)],
            "Symbols" : [i for i in symb[:np.unique(clustered, axis =0).shape[0]]]})
        self.Table = pd.DataFrame({"Original" : [tuple(cs_model.cluster_centers_[i]) for i in cs_model.labels_]})
        self.Table = self.Table.merge(self.Map,how="left",left_on="Original", right_on="Original",suffixes=("_l","_r"))
        clustered = np.array([list(i) for i in self.Table["Recolour"]])
        """
        For some insane reason I don't understand YET, when number of blocks is very large,
        when converting the array back to an image using C-order, the image gets super stretched
        and turns into stripes, like when you're rewinding a VHS.
        When block number is small it works perfectly fine. Will look into it later.
        To fix this I have :
        1. Used F order
        2. Flipped along horiz axis, and rotated 270 deg

        its probably coz i swapped the axes on some array, not sure
        """
        self.clust_array = np.reshape(clustered,
                                 (w,h,3), order="F")
        new_image = (Image.fromarray(self.clust_array.astype(np.uint8), mode="RGB")
                     .transpose(Image.FLIP_TOP_BOTTOM)
                     .transpose(Image.ROTATE_270))
        
        return new_image
    
    def recolour(self,col): # returns colour shortest distance away
        self.rgb["dist"] = (((self.rgb["R"]-col[0])**2 + (self.rgb["G"]-col[1])**2 + (self.rgb["B"]-col[2])**2)**0.5)
        k = self.rgb.iloc[self.rgb["dist"].idxmin()].loc[["R","G","B"]].values
        return k

    def create_grid_symbols(self,x_start,y_start,x_end,y_end):
        f, border, ten = 20, 4, 2
        grid = self.create_grid(x_start,y_start,x_end,y_end)
        grid = grid.transpose(Image.ROTATE_90).transpose(Image.FLIP_TOP_BOTTOM)
        draw = ImageDraw.Draw(grid)
        for i in range(x_end - x_start):
            for j in range (y_end - y_start):
                x, y = border+f*i+i+(i//10)*ten, border+f*j+j+(j//10)*ten
                letter = self.Map[list(self.Map["Recolour"])==self.clust_array[y_start+j,x_start+i]]["Symbols"].values[0]
                grid.paste(self.text_rotate(letter),(x,y),self.text_rotate(letter))
        grid = grid.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.ROTATE_270)
        return grid
    
    def text_rotate(self,letter):
        fontz = ImageFont.truetype("arial.ttf", 18)
        txt=Image.new('RGBA', (20,20), (0,0,0,0))
        textt = ImageDraw.Draw(txt)
        textt.text( (10, 10), letter,  font=fontz, fill="black", anchor="mm")
        txt = txt.transpose(Image.ROTATE_90).transpose(Image.FLIP_TOP_BOTTOM)
        return txt


    def create_grid(self,x_start=0,y_start=0,x_end=None,y_end=None):
        #new pixel size, border thickness, extra thickness for every 10th line
        f, border, ten = 20, 4, 2
        # box width = pixels*f + width of vertical lines + extra for every 10th line and each border
        # similar for height
        w, h, rgb = self.pixelated.shape
        if x_end == None and y_end == None :
            x_end = w
            y_end = h
        grid = Image.new("RGB", ((x_end-x_start)*(f+1)+1+2*(border-1)+ten*((x_end-x_start)//10),
                                 (y_end-y_start)*(f+1)+1+2*(border-1)+ten*((y_end-y_start)//10)), (0, 0, 0))
        draw = ImageDraw.Draw(grid)
        for i in range(x_end - x_start):
            for j in range (y_end - y_start):
                x, y = border+f*i+i+(i//10)*ten, border+f*j+j+(j//10)*ten
                draw.rectangle((x, y, x-1+f, y-1+f),
                                fill=tuple(self.clust_array[y_start+j,x_start+i]) )
        grid = grid.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.ROTATE_270)
        #grid = ImageOps.pad(grid,(1125,1500))
        return grid
    
    def splitter(self):
        split=[]
        w, h, rgb = self.pixelated.shape
        # finds the the top left pixel of every section
        # takes a 40*60 block from image starting at that ^ pixel
        for i in range(0,w,60):
            for j in range(0, h,40):
                split.append(self.create_grid_symbols( i, j, i+60 if i+60<w else w, j+40 if j+40<h else h ))
        return split
    
if __name__ == "__main__":
    t1=time.perf_counter()
    
    img = Image.open("Dog.jpg") #Reading file
    ct = int(str.strip(input("Enter ct (14, 18 etc) : ")))
    pic_width = float(str.strip(input("Enter width in inches : ")))
    clus = int(str.strip(input("Enter number of colours : ")))

    Sample = cross_pattern(img, ct, pic_width)
    result = Sample.rgb_cluster(clus)
    split_pattern = Sample.splitter()
    for k in split_pattern:
        k.show()
    l=Sample.create_grid()
    l.show()

    t2=time.perf_counter()
    print(f"Time elapsed : {t2-t1:0.5f}")
    

