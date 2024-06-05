from PIL import Image, ImageDraw, ImageOps, ImageFont
import numpy as np
from sklearn.cluster import KMeans
import time
import json
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import reportlab.platypus.flowables
import io
from os import listdir

class cross_pattern:
    def __init__(self, original, ct, pic_width, clus) -> None:
        self.original = original
        self.colour_table = pd.read_csv("app/dmc_rgb.csv")
        self.colour_table["RGB"] = self.colour_table["R"].apply(str)+" "+self.colour_table["G"].apply(str)+" "+self.colour_table["B"].apply(str)
        self.grid_parameters = (70, 14, 9, 250) 
        # New pixel size, Border thickness, Extra thickness for every 10th line, White Outer border width
        w, h = self.original.size
        # Number of blocks/stitches across the width
        # min to make sure block number doesn't become huge and waste space
        self.block_num = min( ct * pic_width , w*2)
        self.pixelated = self.original.copy().resize((int(self.block_num),int(self.block_num*h/w)),4)
        # .png has 4 properties per pixel R,G,B,Alpha , removing Alpha here
        # Faster than using .delete
        self.pixelated = (np.asarray(self.pixelated)
                          [:,:,: 3 if np.asarray(self.pixelated).shape[2]>3 else None])
        self.rgb_cluster(clus)

    def rgb_cluster(self,cluster_num):
        h, w, rgb = self.pixelated.shape
        unclustered = (np.reshape(self.pixelated, #long array with 3 properties each
                                  (h*w,3) )) 
        #K-means, n_init=1 as we're using k-means++, could just n_init=auto it instead
        model = KMeans(n_clusters=cluster_num,n_init=1)
        cs_model = model.fit(unclustered)
        symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZæ1234567890£$&@?>¥Ø≡<#¶§ßabcdefghijklmnopqrstuvwxyz"
        # mapping computed means (cluster_centers) to each pixel (labels)
        clustered = np.array([cs_model.cluster_centers_[i] for i in cs_model.labels_])

        self.Map = pd.DataFrame({
            "Original" : [tuple(i) for i in np.unique(clustered, axis =0)],
            "Recolour" : [self.recolour(tuple(i))[0] for i in np.unique(clustered, axis =0)],
            "RecolourList" : [self.recolour(tuple(i))[1] for i in np.unique(clustered, axis =0)],
            "Symbols" : [symbol for symbol in symbols[:np.unique(clustered, axis =0).shape[0]]]
            })
        
        self.Summary = pd.DataFrame({"Original" : [tuple(cs_model.cluster_centers_[i]) for i in cs_model.labels_]})

        self.Summary = self.Summary.merge(self.Map,how="left", on="Original")

        clustered = np.array([list(i) for i in self.Summary["Recolour"]])

        self.clust_array = np.reshape(clustered,(h,w,3), order="C")
    
    def pdf_create(self):
        split_pattern = self.splitter()
        full_grid = self.create_grid()
        self.thread_table()
        
        jj = io.BytesIO()
        full_grid.save(jj,"PNG")
        jj.seek(0)
        rr = reportlab.platypus.flowables.Image(jj)
        """
        c = canvas.Canvas("hello.pdf")
        c.drawImage(rr,x=0,y=0 )
        c.showPage()
        c.save()"""
    
    def thread_table(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, frame_on=False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        v = self.Summary.copy() # Summary is a df with all pixels and their colours
        self.Summary.to_csv("no.csv")
        v = pd.DataFrame(v[["RecolourList","Symbols"]])
        v = pd.DataFrame(v.groupby(['RecolourList',"Symbols"]).size()).rename(columns={0:"Count"})
        v = v.reset_index()
        
        bb = v.merge(right=self.colour_table, how="left", left_on="RecolourList", right_on="RGB")
        results = bb[["Name","Code","Count"]]
        results = results.reset_index().T.reset_index().T
        #results = results.groupby([["Name","Code"]]).sum("Count")
        
        ax.table(results.values, colWidths=[0.1,0.4,0.1,0.1],cellLoc = 'center', rowLoc = 'center',loc='left', fontsize=40)
        
        #plt.savefig("bom.png", bbox_inches='tight', dpi=500)

        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        self.thread_tab = Image.open(buf)
        #self.thread_tab.show()
        
    def recolour(self,col): # Returns colour shortest distance away
        self.colour_table["dist"] = (
            ((self.colour_table["R"]-col[0])**2 + (self.colour_table["G"]-col[1])**2 + (self.colour_table["B"]-col[2])**2)**0.5)
        min_distance_index = self.colour_table["dist"].idxmin()
        closest_colour = self.colour_table.iloc[min_distance_index].loc[["R","G","B"]].values
        #print(f"{closest_colour[0]} {closest_colour[1]} {closest_colour[2]}")
        # returns RGB list and RGB string
        return closest_colour, f"{closest_colour[0]} {closest_colour[1]} {closest_colour[2]}"

    def create_grid_symbols(self,x_start,x_end,y_start,y_end):
        f, border, ten, axis = self.grid_parameters
        grid = self.create_grid(x_start,x_end,y_start,y_end)
        draw_object = ImageDraw.Draw(grid)
        font = ImageFont.truetype("arial.ttf",size=int(f*0.9))
        for i in range(y_end - y_start):
            for j in range(x_end - x_start):
                col = self.clust_array[i+y_start,j+x_start]
                mask = (list(self.Map["Recolour"]) == self.clust_array[i+y_start,j+x_start])
                letter = self.Map[mask]["Symbols"].values[0]
                draw_object.text((axis+border+j*(f+1)+f/2+ten*(j//10),axis+border+i*(f+1)+f/2+ten*(i//10)),letter,
                          fill= "#000000" if ((col[0]**2+col[1]**2+col[2]**2)**0.5) > 70 else "#cdcdcd",
                          font=font,anchor="mm")
        #new_grid = Image.new("RGB",(3650,5160),color="#ffffff")
        #new_grid.paste(grid, (120,420))
        return grid
    
    def desat(self, rgb:tuple ):
        # Tried using HSV but it wouldn't work
        wh = Image.new("RGBA",(1,1), color=(255,255,255,70))
        temp = Image.new("RGB",(1,1),color=rgb)
        # Takes a white image and pastes it on top of the current one, making it lighter
        temp.paste(wh,(0,0),wh)
        return tuple(np.asarray(temp)[0,0])

    def create_grid(self,x_start=0,x_end=None,y_start=0,y_end=None):
        f, border, ten, axis = self.grid_parameters
        h, w, rgb = self.pixelated.shape
        font = ImageFont.truetype("arial.ttf",size=f)
        full = False
        # If this is the full image, set default values
        if x_end == None and y_end == None :
            x_end = w
            y_end = h
            full = True
        # Box width = pixels*f + width of vertical lines + extra for every 10th line and each border
        # Will also remove one 10th line extra if the last line is a 10th line, so the border isnt super thick
        grid = Image.new("RGB", 
                ((x_end-x_start)*(f+1)+1+2*(border-1)+ten*(((x_end-x_start)//10)-(1 if (x_end-x_start)%10==0 else 0) ),
                (y_end-y_start)*(f+1)+1+2*(border-1)+ten*(((y_end-y_start)//10)-(1 if (y_end-y_start)%10==0 else 0)))
                ,(0, 0, 0))
        grid = ImageOps.expand(grid, axis, fill="#ffffff" )
        draw = ImageDraw.Draw(grid)
        for i in range(y_end - y_start):
            for j in range(x_end - x_start):
                # print(f"{y_end} // {y_start} // {x_end} // {x_start} // {i} // {j}")
                x, y = axis+border+f*j+j+(j//10)*ten, axis+border+f*i+i+(i//10)*ten
                if j%10 == 0 and i == 0:
                    draw.text((x-ten/2,axis-20),str(x_start+j),fill="#000000", font=font, anchor="ms")
                if i%10 == 0 and j == 0:
                    draw.text((axis-20,y-ten/2),str(y_start+i),fill="#000000", font=font, anchor="rm")
                col = tuple(self.clust_array[y_start+i,x_start+j])
                draw.rectangle((x, y, x-1+f, y-1+f),
                                fill = col if full else self.desat(col) )
        draw.text((f+1+x-ten/2,axis-20),str(1+x_start+j),fill="#000000", font=font, anchor="ms")
        draw.text((axis-20,f+1+y-ten/2),str(1+y_start+i),fill="#000000", font=font, anchor="rm")
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
    # Testing
    t1=time.perf_counter()
    
    img = Image.open("Dog.jpg") # Reading file
    parameters = json.load(open("parameters.json"))
    Sample = cross_pattern(img,
                            parameters["ct"],
                              parameters["pic_width"],
                              parameters["clus"] )
    Sample.pdf_create()

    t2=time.perf_counter()

    print(f"Time elapsed : {t2-t1:0.5f}")
    

