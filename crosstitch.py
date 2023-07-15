from PIL import Image, ImageDraw, ImageOps, ImageFont, ImageEnhance
import numpy as np
from sklearn.cluster import KMeans
import time
import json
import pandas as pd
class cross_pattern:
    def __init__(self, original, ct, pic_width) -> None:
        self.original = original
        self.rgb = pd.read_csv("dmc_rgb.csv")
        self.grid_parameters = (70, 14, 9) # new pixel size, border thickness, extra thickness for every 10th line
        # choosing to derive w,h from self.original in each function where its needed
        # rather than using two extra instance variables, i just think its a bit ugly
        w, h = self.original.size
        # number of blocks/stitches across the width
        # min to make sure block number doesn't become huge and waste space
        self.block_num = min( ct * pic_width , w)
        self.pixelated = self.original.copy().resize((int(self.block_num),int(self.block_num*h/w)),4)
        # .png has 4 properties per pixel R,G,B,Alpha , we're getting rid of the last one
        # its faster this way than using .delete
        self.pixelated = (np.asarray(self.pixelated)
                          [:,:,: 3 if np.asarray(self.pixelated).shape[2]>3 else None])

    def rgb_cluster(self,cluster_num):
        h, w, rgb = self.pixelated.shape
        unclustered = (np.reshape(self.pixelated, #long array with 3 properties each
                                  (h*w,3) )) 
        #K-means, n_init=1 as we're using k-means++, could just n_init=auto it instead
        model = KMeans(n_clusters=cluster_num,n_init=1)
        cs_model=model.fit(unclustered)
        symb = "ABCDEFGHIJKLMNOPQRSTUVWXYZæ1234567890£$&@?>¥Ø≡<#¶§ßabcdefghijklmnopqrstuvwxyz"
        # mapping computed means (cluster_centers) to each pixel (labels)
        clustered = np.array([cs_model.cluster_centers_[i] for i in cs_model.labels_])

        self.Map = pd.DataFrame({"Original" : [tuple(i) for i in np.unique(clustered, axis =0)],
                                 "Recolour" : [self.recolour(tuple(i)) for i in np.unique(clustered, axis =0)],
            "Symbols" : [i for i in symb[:np.unique(clustered, axis =0).shape[0]]]})
        
        self.Table = pd.DataFrame({"Original" : [tuple(cs_model.cluster_centers_[i]) for i in cs_model.labels_]})
        self.Table = self.Table.merge(self.Map,how="left",left_on="Original", right_on="Original",suffixes=("_l","_r"))
        clustered = np.array([list(i) for i in self.Table["Recolour"]])

        self.clust_array = np.reshape(clustered,(h,w,3), order="C")
        new_image = Image.fromarray(self.clust_array.astype(np.uint8), mode="RGB")

        return new_image
    
    def recolour(self,col): # returns colour shortest distance away
        self.rgb["dist"] = (((self.rgb["R"]-col[0])**2 + (self.rgb["G"]-col[1])**2 + (self.rgb["B"]-col[2])**2)**0.5)
        k = self.rgb.iloc[self.rgb["dist"].idxmin()].loc[["R","G","B"]].values
        return k

    def create_grid_symbols(self,x_start,x_end,y_start,y_end):
        f, border, ten = self.grid_parameters
        grid = self.create_grid(x_start,x_end,y_start,y_end)
        enhance = ImageEnhance.Contrast(grid)
        grid = enhance.enhance(0.7)
        draw = ImageDraw.Draw(grid)
        font = ImageFont.truetype("arial.ttf",size=int(f*0.9))
        for i in range(y_end - y_start):
            for j in range(x_end - x_start):
                col = self.clust_array[i+y_start,j+x_start]
                mask = (list(self.Map["Recolour"]) == self.clust_array[i+y_start,j+x_start])
                letter = self.Map[mask]["Symbols"].values[0]
                draw.text((border+j*(f+1)+f/2+ten*(j//10),border+i*(f+1)+f/2+ten*(i//10)),letter,
                          fill= "#000000" if ((col[0]**2+col[1]**2+col[2]**2)**0.5) > 65 else "#cdcdcd",
                          font=font,anchor="mm")

        return grid

    def create_grid(self,x_start=0,x_end=None,y_start=0,y_end=None):
        f, border, ten = self.grid_parameters
        h, w, rgb = self.pixelated.shape
        full = False
        if x_end == None and y_end == None : # if this is the full sample image, set default values
            x_end = w
            y_end = h
        # box width = pixels*f + width of vertical lines + extra for every 10th line and each border
        grid = Image.new("RGB", ((x_end-x_start)*(f+1)+1+2*(border-1)+ten*((x_end-x_start)//10),
                                 (y_end-y_start)*(f+1)+1+2*(border-1)+ten*((y_end-y_start)//10)), (0, 0, 0))
        draw = ImageDraw.Draw(grid)
        for i in range(y_end - y_start):
            for j in range(x_end - x_start):
                #print(f"{y_end} // {y_start} // {x_end} // {x_start} // {i} // {j}")
                x, y = border+f*j+j+(j//10)*ten, border+f*i+i+(i//10)*ten
                draw.rectangle((x, y, x-1+f, y-1+f),
                                fill =tuple(self.clust_array[y_start+i,x_start+j]))
        #grid = ImageOps.pad(grid,(1125,1500))
        return grid
    
    def splitter(self):
        split=[]
        h, w, rgb = self.pixelated.shape
        for i in range(0,h,60):
            for j in range(0, w,40):
                m = self.create_grid_symbols(j, min(j+40,w), i, min(i+60,h) )
                split.append(m)
        return split
    
if __name__ == "__main__":
    t1=time.perf_counter()
    
    img = Image.open("Dog.jpg") #Reading file
    parameters = json.load(open("parameters.json"))

    Sample = cross_pattern(img, parameters["ct"], parameters["pic_width"] )
    result = Sample.rgb_cluster(parameters["clus"])
    split_pattern = Sample.splitter()
    for k in split_pattern:
        k.show()
    l=Sample.create_grid()
    l.show()

    t2=time.perf_counter()
    print(f"Time elapsed : {t2-t1:0.5f}")
    

