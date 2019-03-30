# AI driving agent in GTA sandreas 
In this project we will create neural network based model which can drive any vehicle across open world of GTA SA


# INDEX

* [**`Approach 1(Open CV way)`**](https://github.com/saurabh241930/AI_in_GTA_SA#approach-one-open-cv)
* [**`Approach 2(Image Classifier way)`**](https://github.com/saurabh241930/AI_in_GTA_SA#approach-two-convunational-neural-network-classifier-method)
* [**`Approach 3(Object Detection way)`**](https://github.com/saurabh241930/AI_in_GTA_SA#approach-three-real-time-object-detection-method)
* [**`Approach 4(Neural Network + Genetic algorithm way)`**]()




________________________________________________________________________________________________________________________________________


## `APPROACH ONE (OPEN CV)`
### SETUP 
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

________________________________________________________________________________________________________________________________________

## `APPROACH TWO (Convunational Neural Network Classifier method)`
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
  and saving it as np array  
  
  
  ```python
  file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    training_data = []
    
  ```
  
  ## data pre-processing
  
  Loading data into memory and creating dataframe using pandas
  
  ```python
  train_data = np.load('training_data.npy')
  df = pd.DataFrame(train_data)
  
  lefts = []
rights = []
forwards = []

shuffle(train_data)

for data in train_data:
    img = data[0]
    choice = data[1]

    if choice == [1,0,0]:
        lefts.append([img,choice])
    elif choice == [0,1,0]:
        forwards.append([img,choice])
    elif choice == [0,0,1]:
        rights.append([img,choice])
    else:
        print('no matches')
  ```
  
  Now naturally due to game nature the **foward** data will be much more compare to **left & right**
  
  so we have to take care of that part
  
  ```python
  forwards = forwards[:len(lefts)][:len(rights)]
lefts = lefts[:len(forwards)]
rights = rights[:len(forwards)]

final_data = forwards + lefts + rights
shuffle(final_data)

np.save('training_data.npy', final_data)
  ```
  this will equalize all three of category
  
  
  ### cross checking data and its output
  
  ```python
  import numpy
import PIL
from PIL import Image
from matplotlib.pyplot import imshow
import random

random_frame = random.randint(1,14574)


image_arr = df_new[0].iloc[random_frame]
move = df_new[1].iloc[random_frame]

img = PIL.Image.fromarray(image_arr)


if move == [0,1,0]:
    print("foward")
elif move == [1,0,0]:
    print("left")
elif move == [0,0,1]:
    print("right")
    
    
print("frame : "+str(random_frame))


img
```
this will print image and its output

## start training

For neural network model we will use **Alexnet**

```python
# creating model instance

WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'pygtaSa-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2',EPOCHS)

model = alexnet(WIDTH, HEIGHT, LR)
```

loading training data into memory


```python
train_data = np.load('training_data.npy')

train = train_data[:-500]
test = train_data[-500:]

#(-1,w,h,1) reshape because -1 means that the length in that dimension is inferred. 
# This is done based on the constraint that the number of elements in an
# ndarray or Tensor when reshaped must remain the same. In the tutorial,
# each image is a row vector (784 elements) and 
# there are lots of such rows (let it be n, so there are 784n elements).
# So, when you write

# x_image = tf.reshape(x, [-1, 28, 28, 1])

# TensorFlow can infer that -1 is n.

#ref : https://stackoverflow.com/questions/41848660/why-the-negative-reshape-1-in-mnist-tutorial/41848962



X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
test_y = [i[1] for i in test]
```

Start training

this should save model files in directoy to use that for training 

```python

model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)
```
  
To test the model

```python
# test_model.py

import numpy as np
from grabscreen import grab_screen
import cv2
import time
from directkeys import PressKey,ReleaseKey, W, A, S, D
from alexnet import alexnet
from getkeys import key_check

import random

WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'pygta5-car-fast-{}-{}-{}-epochs-300K-data.model'.format(LR, 'alexnetv2',EPOCHS)

t_time_long = 0.5
t_time = 0.05


def straight():
##    if random.randrange(4) == 2:
##        ReleaseKey(W)
##    else:
    PressKey(W)
    time.sleep(t_time)
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    PressKey(W)
    PressKey(A)
    #ReleaseKey(W)
    ReleaseKey(D)
    #ReleaseKey(A)
    time.sleep(t_time)
    ReleaseKey(A)

def right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    #ReleaseKey(W)
    #ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(D)
    
model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

def main():
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    while(True):
        
        if not paused:
            # 800x600 windowed mode
            #screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
            screen = grab_screen(region=(0,40,800,600))
            print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (160,120))

            prediction = model.predict([screen.reshape(160,120,1)])[0]
            print(prediction)

            turn_thresh = .75
            fwd_thresh = 0.70

            if prediction[1] > fwd_thresh:
                straight()
            elif prediction[0] > turn_thresh:
                left()
            elif prediction[2] > turn_thresh:
                right()
            else:
                straight()

        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

main()       


```

Here's result on auto_driving mode

<img src="auto_driving.gif" width="800" height="600">

________________________________________________________________________________________________________________________________________

## `APPROACH THREE (Real time Object Detection method)`

In this method we will integrate Awesome **Tensorflow Object Detcion API** directly to our gameplay 

Now to use that you have to install all the required libraries mentioned on its site

To directly integrate with our screen capture you can replace official **session detection** part with this in tutorial notebook from tensorflow

```python

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      #screen = cv2.resize(grab_screen(region=(0,40,1280,745)), (WIDTH,HEIGHT))
      screen = cv2.resize(grab_screen(region=(0,40,1280,745)), (800,400))
      image_np = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)

      cv2.imshow('window',image_np)
      if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break

```

