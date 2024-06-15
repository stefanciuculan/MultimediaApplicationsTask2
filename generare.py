import os

# setting jax backend
os.environ["KERAS_BACKEND"] = "jax"  # @param ["tensorflow", "jax", "torch"]

from tensorflow import data as tf_data
import tensorflow_datasets as tfds
import keras
import keras_cv
import numpy as np
import cv2
from keras_cv import bounding_box
import os
from keras_cv import visualization
import tqdm
import matplotlib.pyplot as plt
import pyttsx3
import time

# load a pretrained model for object detection
pretrained_model = keras_cv.models.YOLOV8Detector.from_preset("yolo_v8_m_pascalvoc", bounding_box_format="xywh")

model = keras_cv.models.StableDiffusion(
    img_width=512, img_height=512, jit_compile=False
)

def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")

# generate images using a text prompt and the Stable Diffusion model
images = model.text_to_image("A fuchsia colored bottle, on meadow background", batch_size=3)

# plot the AI-generated images
plot_images(images) #1

# reading a image using a path
image = cv2.imread('./image.png')
image = np.array(image)

inference_resizing = keras_cv.layers.Resizing(
    640, 640, pad_to_aspect_ratio=True, bounding_box_format="xywh"
)
image_batch = inference_resizing([image])

# define the class ID and the class mapping
class_id = ["Bottle"]
class_mapping = dict(zip(range(len(class_id)), class_id))

# detect object using the image batch 
y_pred = pretrained_model.predict(image_batch)

# visualize bounding boxes on the image
visualization.plot_bounding_box_gallery(
    image_batch,
    value_range=(0, 255),
    rows=1,
    cols=1,
    y_pred=y_pred,
    scale=5,
    font_scale=0.7,
    bounding_box_format="xywh",
    class_mapping=class_mapping
) # 2a

image_rgb = cv2.imread('./image.png')
image_rgb = np.array(image_rgb)

# specify background color range (green)
l_obj_color = np.array([6, 46, 3])
u_obj_color = np.array([210, 255, 210])

# create the color mask
mask = cv2.inRange(image_rgb, l_obj_color, u_obj_color)

# apply the mask
masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask = mask)

# display the mask and the masked image
cv2.imshow('Mask', mask) # 2b
cv2.waitKey(0)
cv2.imshow('Masked Image', masked_image) # 2c
cv2.waitKey(0)
cv2.destroyAllWindows()

# save them
cv2.imwrite('mask.png', mask)
cv2.imwrite('maskedImage.png', masked_image)

# convert to grayscale both the original image and the masked one and then save them
grayscale_masked = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
cv2.imwrite('grayMasked.png', grayscale_masked)

grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
cv2.imwrite('grayImage.png', grayscale_image) # 2d

# select the images that will be included in the video
image_paths = ['image.png', 'grayImage.png', 'mask.png', 'maskedImage.png', 'grayMasked.png']
outputVideo_path = './outputVideo.mp4'

# choose the frame rate and size
fps = 0.5
frame_size = (512,512)

# create the video and choose the codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter(outputVideo_path, fourcc, fps, frame_size)

# include the images in the video
for image_path in image_paths:
    image = cv2.imread(image_path)
    output.write(image)

output.release() # 3

# text that will be added on the video frames
image_descriptions = [
    "Original image",
    "Grayscale original image",
    "Mask", "Masked image",
    "Grayscale masked image"]

outputAudio_path = './outputAudioWithText.mp3'
outputVideo_path = './outputVideoWithText.mp4'

# choose the frame rate and size
fps = 0.5
frame_size = (512,512)

# initialize the text to speech engine
engine = pyttsx3.init()

# create the video and choose the codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter(outputVideo_path, fourcc, fps, frame_size)

# variable to concatenate descriptions
audio_text = ""

# set the initial time
current_time = time.time()

for image_path, description in zip(image_paths, image_descriptions):
    image = cv2.imread(image_path)
    
    # text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    font_scale = 1
    color = (0, 0, 0)
    thickness = 2
    
    # add text to the image     
    image_with_text = cv2.putText(image, description, org, font, font_scale, color, thickness, cv2.LINE_AA)

    output.write(image_with_text)

    # calculate the time difference
    time_difference = time.time() - current_time
    
    # wait for the remaining time
    time.sleep(max(0, 2 - time_difference))

    # concatenate descriptions
    audio_text = audio_text + description + " "

# save audio file
engine.save_to_file(audio_text, outputAudio_path)
engine.runAndWait()
current_time = time.time()

output.release() # 4