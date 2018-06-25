# behavioral-cloning
Teaching a model to drive a car autonomously while staying within the lane lines.

## Building the model
I used NVIDIA's End-to-End model for self driving cars, as reported [here](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

The architecture of the model is summarize in the bellow image:
![architecture](https://github.com/hanieh-hassanzadeh/behavioral-cloning/blob/master/Images/architecture1.png)

This architecture is define in `create_model.py`. The variables needed to build the model are also included in this file; namely:

```
number_of_epochs = 15

number_of_samples_per_epoch = 20032

number_of_validation_samples = 6400
 
learning_rate = 1e-4

activation_relu = 'relu'
```
 
## Preparing input data
For input data, I collected two sets of data from training mode of the simulator; one in the default direction and one in the reverse direction. Then, I combined this data with Udasity provided data. Then, cropped the 0.35 of the top and 0.1 of the bottom of images, where no useful information could be obtained for the model. I randomly applied brightness correction, flipping, and shearing to the images. In the end I resized the images to size (64, 64), to feed to the model. 

To over come overfitting, I didn't add to the number of the images from this step. Instead, for a given input image, either I used the image as is or randomly applied any of the mentioned methods above. These functions are defined in `functions.py`. The effect of each of the methods is demonstrated bellow.

| Actual Image         	|     Final resized	        					| 
|:---------------------:|:---------------------------------------------:| 
|![image](https://github.com/hanieh-hassanzadeh/behavioral-cloning/blob/master/Images/image.jpg) |![resized](https://github.com/hanieh-hassanzadeh/behavioral-cloning/blob/master/Images/resize.jpg)|

|Cropped                |   Flipped                                     |
|:---------------------:|:---------------------------------------------:|
|![crop](https://github.com/hanieh-hassanzadeh/behavioral-cloning/blob/master/Images/crop.jpg)|![flip](https://github.com/hanieh-hassanzadeh/behavioral-cloning/blob/master/Images/flip.jpg)|

|Brightness             | Sheared                                       |
|:---------------------:|:---------------------------------------------:|
|![crop](https://github.com/hanieh-hassanzadeh/behavioral-cloning/blob/master/Images/gamma.jpg)|![flip](https://github.com/hanieh-hassanzadeh/behavioral-cloning/blob/master/Images/shear.jpg)|


### Right and left camera images 
I applied a steering coefficient of 0.229 to the `left`, and 0.2 to `right` camera images. I did so to correct the tendency of the model agresively steering to the left.

### Data generator
Due to memory limit, I used data generator for training/validation rather than storing the training/validation data in memory.
This is done through `generate_next_batch` function in `functions.py`.

## performance of the model
### Running the model
The main program is called 'model.py'. To run the model, I do

`python model.py`

, which results in a final training loss of 0.0118 and a validation loss of 0.0119 for 15 epochs. It could get even better with larger epoch numbers, but I used 15 epoches since running the model is very expensive. Bellow, the output of the model creation is shown

![epoch](https://github.com/hanieh-hassanzadeh/behavioral-cloning/blob/master/Images/loss.png)

### Saving the model
I saved the model as `model.h4`.

## Testing the model on the simulator

To test the model on the simulator, first, I ran this command:

`python drive.py model.h4 runv`

Then, I open the simulator in `autonomous` mode to run the built model and save the snapshot of the video frames in a folder called `runv`.

### Some considerations

Before feeding the images to the model in `drive.py`, I did two pre-processing steps; namely cropping (0.35 of top and 0.1 of bottom) and resizing to (64, 64), the same a s done when buildeing the model.

I used the following lines to set and control the throttle in a reasonable range.

```
throttle = (desired_speed - float(speed)) * 0.5
if throttle<0:
             throttle=0
 ```

## Final video
To produce the final video, I run the command:

`python video.py runv`, which generated the video file of from the output images.

The final generated video file is [here](https://github.com/hanieh-hassanzadeh/behavioral-cloning/blob/master/runv.mp4)

## Discussion

The most challenging tasks here for me were to teach the model not to steer too hard to the left, and to take just enough steering angle in the sharp turns. I over come this problem by using different methods that I explained in the data preperation section, also applying different steering angles for left and right camera images. In addition, I used the data thet I have gathered running the simularor in default and reverse direction.

The problem that a real car may face using models trained by real images could be that not all the roads look the same. 
