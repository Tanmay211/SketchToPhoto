"""
SKETCH TO PHOTO CONVERTER
"""
import cv2
import numpy as np
import glob
import json
from PIL import Image
from google_images_download import google_images_download
json_file=open("/Users/sarthakdandriyal/Desktop/pro1/data.json","r",encoding="utf-8")
data=json.load(json_file)
json_file.close()
list=[]
print("sketch to photo conversion ")
s=data['background']
response = google_images_download.googleimagesdownload()
arguments = {"keywords":s,"limit":1}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
list.append(s)
path="/Users/sarthakdandriyal/Desktop/google-images-download-master/downloads/"
p=path+list[0]
p=p+"/*"
for f in glob.iglob(p):
    image=cv2.imread(f)
cv2.imwrite("/Users/sarthakdandriyal/Desktop/bgd.jpg",image)

image_file="/Users/sarthakdandriyal/Downloads/sketch.jpg"
im = Image.open(image_file)
a,b=im.size
img=cv2.imread("/Users/sarthakdandriyal/Desktop/bgd.jpg")
resized_image= cv2.resize(image, (a, b)) 
cv2.imwrite("/Users/sarthakdandriyal/Desktop/bgd.jpg",resized_image)
#downloading images

for obj in data['img']:       
    s=obj['name'] 
    response = google_images_download.googleimagesdownload()
    arguments = {"keywords":s,"limit":25}   #creating list of arguments
    paths = response.download(arguments)#passing the arguments to the function 
    list.append(s)
#hole filling
sketch="/Users/sarthakdandriyal/Downloads/sketch.jpg"
im_in = cv2.imread(sketch, cv2.IMREAD_GRAYSCALE);
th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV);
im_floodfill = im_th.copy()
 
# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = im_th.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
 
# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0,0), 255);
 
# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
 
# Combine the two images to get the foreground.
im_out = im_th | im_floodfill_inv

im =im_floodfill
cv2.imwrite("/Users/sarthakdandriyal/Desktop/vb.jpg",im)
li=[]
i=0
for obj in data['img']: 
    i=i+1
    s=obj['min_X']
    li.append(s)
    s=obj['max_X']
    li.append(s)
    s=obj['min_Y']
    li.append(s)
    s=obj['max_Y']
    li.append(s)
    #print(li)
    im_in = cv2.imread(sketch, cv2.IMREAD_GRAYSCALE);
    th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV);
    im_floodfill = im_th.copy()
     
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
     
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
     
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
     
    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    im =im_floodfill
    cv2.imwrite("/Users/sarthakdandriyal/Desktop/vb.jpg",im)
    crop_img = im[li[2]-5:li[3]+5,li[0]-5:li[1]+5]
    original = crop_img
    cv2.imwrite("/Users/sarthakdandriyal/Desktop/finale1.jpg",crop_img)
    image_file="/Users/sarthakdandriyal/Desktop/finale1.jpg"
    im = Image.open(image_file)
    aa,bb=im.size
    cv2.imwrite("/Users/sarthakdandriyal/Desktop/vb.jpg",crop_img)

    path="/Users/sarthakdandriyal/Desktop/google-images-download-master/downloads/"
    p=path+list[i]
    p=p+"/*"
    n=0
    for k in glob.iglob(p):
        pth=k
        n=n+1
        if n==1:
            break
           
    sim=0.0
    # load all the images
    sift = cv2.xfeatures2d.SIFT_create()
    kp_1, desc_1 = sift.detectAndCompute(original, None)
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    titles=[]
    all_images_to_compare=[]
    for f in glob.iglob(p):
        image=cv2.imread(f)
        titles.append(f)
        all_images_to_compare.append(image)
    for image_to_compare ,title in zip(all_images_to_compare,titles): 
    # 1) Check if 2 images are equals
       
        if original.shape == image_to_compare.shape:
            print("The images have same size and channels")
            difference = cv2.subtract(original, image_to_compare)
            b, g, r = cv2.split(difference)
         
            if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
                print("Similarity: 100%(equal size and channels)")
                
        # 2) Check for similarities between the 2 images
        
      
        kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
        
        matches = flann.knnMatch(desc_1, desc_2, k=2)
         
        good_points = []
        for m, n in matches:
            if m.distance < 0.6*n.distance:
                good_points.append(m) 
        # Define how similar they are
        number_keypoints = 0
        if len(kp_1) <= len(kp_2):
            number_keypoints = len(kp_1)
        else:
            number_keypoints = len(kp_2)
        if number_keypoints==0:
            number_keypoints=1
        ps=len(good_points) / number_keypoints
        if(ps>sim):
           sim=ps
           pth=title     
    #print(sim)
    #print(pth)       
    testimg=cv2.imread(pth)
    cv2.imwrite("/Users/sarthakdandriyal/Desktop/finale.jpg",testimg)
    image_file="/Users/sarthakdandriyal/Desktop/finale.jpg"
    im = Image.open(image_file)
    a,b=im.size
    img = testimg
    #print(a)
    #print(b)
    mask = np.zeros(img.shape[:2],np.uint8)
    
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    #a,b=cv2.imsize(img)
    
    rect = (int((b*1)/10),int((a*1)/10),int((b*9)/10),int((a*9)/10))
    #rect =(0,0,b,a)#coordinates of image (0,0,b,a)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    path="/Users/sarthakdandriyal/Desktop/google-images-download-master/downloads/"
    bp=path+list[0]
    bp=bp+".jpg"
    cv2.imwrite("/Users/sarthakdandriyal/Desktop/fi.jpg",img)
    image_file="/Users/sarthakdandriyal/Desktop/fi.jpg"
    im = Image.open(image_file)
    a,b=im.size
    img= cv2.resize(img, (aa,bb)) 
    cv2.imwrite("/Users/sarthakdandriyal/Desktop/fikd.jpg",img)
    img1=cv2.imread("/Users/sarthakdandriyal/Desktop/bgd.jpg")
    for c in range(bb-2):
        for j in range(aa-2):
            color = img[c,j]
            p = all(x!=0 for x in color)
            if p is True:
                img1[li[2]+c,li[0]+j]=color
    cv2.imwrite('/Users/sarthakdandriyal/Desktop/bgd.jpg',img1)
    li.clear()
cv2.imwrite('/Users/sarthakdandriyal/Desktop/bgd.jpg',img1)
print("THANK YOU(final output image is saved on the desktop by the name bgd.jpg)")

