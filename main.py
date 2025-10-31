import cv2 as cv
import numpy as np
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

main_df = pd.DataFrame({},columns=['Image_Name','Label','Orginial_Name','brightness','Original_height','Original_width','intensity_blue','intensity_green','intensity_red','edge_density'])

main_path = os.getenv('MAIN_PATH')
labels = [x for x in os.listdir(main_path)]
save_path = os.getenv('SAVE_PATH')
edge_path = os.getenv('EDGE_PATH')


print("Starting Image Normalization and Resizing Process")
for i in labels:
    DIR = os.path.join(main_path,i)
    for j in os.listdir(DIR):
        # Read Image
        img = cv.imread(os.path.join(DIR,j))
        # Convert to Grayscale
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        new_gray = cv.resize(gray,(256,256),cv.INTER_CUBIC)
        # Calculate Features
        bright = np.mean(img)
        b,g,r = cv.split(img)
        i_blue = np.mean(b).round(2)
        i_red = np.mean(r).round(2)
        i_green = np.mean(g).round(2)
        # Edge Detection and score calculation
        edged = cv.Canny(img,125,185)
        edged = cv.resize(edged,(256,256),cv.INTER_CUBIC)
        score = (np.sum(edged)/(img.shape[0]*img.shape[1]*255)).round(4)
        edge_success = cv.imwrite(os.path.join(edge_path,i,f'edged_{j}'),edged)
        #Adding checks for saving images
        if not edge_success:
            print(f'Edge Image {j} not saved successfully')
        
        success = cv.imwrite(os.path.join(save_path,i,f'normalised_{j}'),new_gray)
        if not success:
            print(f'Image {j} not saved successfully')
            continue
        # Append Data to DataFrame
        new_df = [  
            f'normalised_{j}',#name
            i,#label
            j,#o_name
            bright,# brightness
            img.shape[0], #o_ht
            img.shape[1] #o_wth
            ,i_blue,#intensity_blue
            i_green,#intensity_green
            i_red,#intensity_red
            score #edge_density
        ]
        main_df.loc[f'normalised_{j}'] = new_df
        print(f'Image {i}{j} processed and saved successfully')
 
# Final Output        
print("Image Normalization and Resizing Process Completed")
main_df.to_csv('image_data.csv',index=False)
