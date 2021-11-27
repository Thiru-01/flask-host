import os
import json
all_data={}
path = "F:/plantix/plantix/server/herbalDect/DataSet/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/"
for r, d, f in os.walk(path):
    for folder in d:
        name = str(folder).replace("__","").replace("Pepper,_","Pepper_")
        name.replace(" ","_").replace("_"," ")
        full_name = name.split('_',1)[0]
        print(name+"\n")
        all_data.update({folder:{"name":full_name,"disname":name,"desc":"","image_path":"images/","treatment":[],"uses":[]}})

with open("data.json","w") as outfile:
    json.dump(all_data,outfile,indent=4)
    
