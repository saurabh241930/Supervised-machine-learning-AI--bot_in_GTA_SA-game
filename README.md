# AI driving agent in GTA sandreas 
In this project we will create neural network based model which can drive any vehicle across open world of GTA SA

## SETUP 
### Image Processing using Open CV

**Plan of Attack:**
* we can pass each image of gameplay to open CV for processing
* run GTA sa in window mode in 800 x 600 res
* crop game part and send it to openCV

now if you basic ImageGrab library for taking without any crop arguments it should look something like this
<img src="https://i.imgur.com/wdnsEYH.png" border=0>

to crop it and saving to another directory you can use this

```python
import numpy as np
from PIL import ImageGrab
import cv2
import time
import os
import sys

# just so this doesn't go on forever:

def screen_record(): 
    last_time = time.time()
    index = 0
    orignal_path = "C:/Users/saurabh/Documents/AI_GTA/orignal"
    while(True):
        index = index + 1
        # 800x600 windowed mode
        printscreen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        print('saving image'+str(index))
        last_time = time.time()
        # Correct color issue
        cv2.imwrite(os.path.join(orignal_path,'image'+str(index)+'.jpg'),cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
            
screen_record()
```

it should save top-left gameplay portion in given directory

Now for processing image to see edges we can use open CV **Canny** filter

for that we can alter orignal into grayscale image then process that image into canny filter

you can use this 2 line for that in **screen_record** function

Note: add **processed_path** like you did for oignal image path

```python
processed_img = cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY)
 cv2.imwrite(os.path.join(processed_path,'image'+str(index)+'.jpg'),cv2.Canny(processed_img, threshold1 = 100, threshold2=150))
```

you can change threshold according to your game scene


### Orignal gameplay 
![g](https://media.giphy.com/media/1dJVL51rWFqUbr153P/giphy.gif)


### Processed in OpenCV
![](https://media.giphy.com/media/9PgnCjLLdNRoaVKavZ/giphy.gif)
![](https://media.giphy.com/media/8P7jcfz67JuzVbUy8y/giphy.gif)

 
### Removing the noise:
Since there are many data in processed image which we don't really need for our ROI(region of interest) we can cut that part

```python
def roi(img, vertices):
    
    #blank mask:
    mask = np.zeros_like(img)   
    
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, 255)
    
    #returning the image only where mask pixels are nonzero
    masked = cv2.bitwise_and(img, mask)
    return masked


vertices = np.array([[10,500],[10,300],[300,200],[500,200],[800,300],[800,500],], np.int32)

# where vertices would be co-ordinates of cut out area portion of your ROI
processed_img = roi(processed_img, [vertices])

```


<img src="https://i.imgur.com/mSzoR3V.jpg" border=0></a>


to remove gaps between detected lines you can blur the image and impose a line betweens gaps clouds and draw hough lines using openCV in roi

```python

def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0],coords[1]), (coords[2],coords[3]), [255,255,255], 3)
    except:
        pass

processed_img = cv2.GaussianBlur(processed_img, (3,3), 0 )
lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, 20, 15)
draw_lines(processed_img,lines)
```


<img src="https://i.imgur.com/Sva3PI8.png" border=0></a>


Now to add this lines to orignal image you can refer this links:

[find Lanes](https://towardsdatascience.com/finding-lane-lines-on-the-road-30cf016a1165)
[here](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html)

<img src="https://i.imgur.com/iuCzUZd.jpg" border=0>

## Making logic without using any AI

we will define go_left or go_right movement function based on the slopes value



```python
 if m1 < 0 and m2 < 0:
        PressKey(D)
        ReleaseKey(A)
        ReleaseKey(W)
    elif m1 > 0  and m2 > 0:
        PressKey(A)
        ReleaseKey(D)
        ReleaseKey(W)
    else:
        PressKey(W)
```
here's how algorithm is playing on first attemp on auto play 

<img src="attemp1.gif">

# Attempt2 (adding CNN classifier)
Now in this attempt we will use **convunational neural network** based classifier 

**Plan of atack :** 

1)We will create dataset of differnet driving scenarios along its corresponding output (i.e **[A,W,D]**)
basically one-hot array of appropriate direction at that frame [left,foward,right]

2) For collecting dataset we will record gameplay frames and key press and encode it one-hot error

```python
def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
    [A,W,D] boolean values.
    '''
    output = [0,0,0]

    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:
        output[1] = 1
    return output
    ```
    

<img src="auto_driving.gif" width="600" height="400">
