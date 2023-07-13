import pandas as pd
# code to get DMC thread codes and their RBG values
site = "https://floss.maxxmint.com/dmc_to_rgb.php" 
page = pd.read_html(site)
dmc = (page[0] # first table on the page (theres only one)
        .iloc[:,[1,2,4,5,6]] # selecting columns
        .rename(columns=({"DMC"       :"Code", # renaming columns
                        "Floss Name":"Name",
                        "Red"       :"R",
                        "Green"     :"G",
                        "Blue"      :"B"
                        }))
        .set_index("Code") # new index
        .drop_duplicates(keep="first")) # removing dupe colours (there was one)
# there were around 20 advert rows like below, now removing the last one after deduping
dmc = dmc[dmc.index!="(adsbygoogle = window.adsbygoogle || []).push({});"]
dmc.to_csv("dmc_rgb.csv")

