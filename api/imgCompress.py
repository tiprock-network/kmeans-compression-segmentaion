# %% [markdown]
# # Image Compression

# %%
import numpy as np
import cv2
import base64


# %%
def calculate_image_size(width, height, bits_per_pixel):
    return width * height * bits_per_pixel

# %%
def image_inputReadProcess():
    image=cv2.imread('../images/fish2.jpg')
    #convert image read from bgr to RGB
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #scale image
    image=image/255.0
    #return processed image
    return image

# %%
print(image_inputReadProcess())

# %%
#initialize random centroid
def initMeans(image,clusters):
    points=image.reshape((-1,image.shape[2]))
    m,n=points.shape
    
    means=np.zeros((clusters,n))
    
    for i in range(clusters):
        random=np.random.choice(m,size=10,replace=False)
        means[i]=np.mean(points[random],axis=0)
    
    return points,means
    

# %%
def distance(x1,y1,x2,y2):
    dist=np.square(x1-x2)+np.square(y1-y2)
    dist=np.sqrt(dist)
    return dist

# %%
def kMeans_Alg(p,mns,c):
    iterations=10
    m,n=p.shape
    index=np.zeros(m)
    
    while iterations>0:
        for j in range(m):
            minimum_dist=float('inf')
            temp=None
            
            for k in range(c):
                x1,y1=p[j,0],p[j,1]
                x2,y2=mns[k,0],mns[k,1]
                
                if distance(x1,y1,x2,y2)<=minimum_dist:
                    minimum_dist=distance(x1,y1,x2,y2)
                    temp=k
                    index[j]=k
        
        for k in range(c):
            c_points=p[index==k]
            if len(c_points)>0:
                mns[k]=np.mean(c_points,axis=0)
        iterations-=1
    
    return mns,index

# %%
from datetime import datetime
current_time = datetime.now()
formatted_time = current_time.strftime("%Y%m%d%H%M%S")


# %%
def image_compress(m, i, img, clusters):
    centroid = np.array(m)
    # Reshape the centroid array to match the shape of the original image
    recovered = centroid[i.astype(int), :].reshape(img.shape)
    
    # Saving the compressed image using OpenCV
    
    compressed_image_path =  f'../images/compressed_{formatted_time}-{clusters}_colors.png'
    cv2.imwrite(compressed_image_path, (recovered * 255).astype(np.uint8))
    
    
    return compressed_image_path


# %%
# Calculate sizes
import os
#change for this to be part of parameter in the API
clusters=3




# %%
def return_base64Image():
    img=image_inputReadProcess()
    # Get the original image shape from the processed image
    

    
    points,means=initMeans(img,clusters)
    means,index=kMeans_Alg(points,means,clusters)
    image_compress(means,index,img,clusters)
        

    # %%
    # Define image file paths
    original_image_path = '../images/fish2.jpg'
    compressed_image_path = f'../images/compressed_{formatted_time}-{clusters}_colors.png' 
    #compressed_image_pickle_path = f'./images/compressed_{formatted_time}-{clusters}_colors.pkl'


    # Get file sizes in bytes
    original_image_size_bytes = os.path.getsize(original_image_path)
    compressed_image_size_bytes = os.path.getsize(compressed_image_path)

    # Print sizes
    originalSize=f"Original image size: {original_image_size_bytes / 1024:.2f} KB"
    compressedSize=f"Compressed image size: {compressed_image_size_bytes / 1024:.2f} KB"
    ratio=f"Compression ratio: {original_image_size_bytes / compressed_image_size_bytes:.2f}"

    #serializing an image using base 64 encoding
    compressed_image=cv2.imread(compressed_image_path)
    _,imgBytes=cv2.imencode('.png',compressed_image)
    #encoded to base64 image
    image_Base64=base64.b64encode(imgBytes).decode('utf-8')
    
    return image_Base64,originalSize,compressedSize,ratio


# %% [markdown]
# # Image Segmentation 
# OpenCV K-Means





