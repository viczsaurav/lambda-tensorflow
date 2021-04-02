import json
import boto3
import numpy as np
import PIL.Image as Image
import pandas as pd

from tensorflow import keras

import tensorflow as tf
import tensorflow_hub as hub

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

IMAGE_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT)
model = tf.keras.Sequential([hub.KerasLayer("model/")])
model.build([None, IMAGE_WIDTH, IMAGE_HEIGHT, 3])

imagenet_labels = np.array(open('model/ImageNetLabels.txt').read().splitlines())
s3 = boto3.resource('s3')


def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    img = readImageFromBucket(key, bucket_name).resize(IMAGE_SHAPE)
    img = np.array(img) / 255.0

    prediction = model.predict(img[np.newaxis, ...])
    predicted_class = imagenet_labels[np.argmax(prediction[0], axis=-1)]

    print('ImageName: {0}, Prediction: {1}'.format(key, predicted_class))


def readImageFromBucket(key, bucket_name):
    bucket = s3.Bucket(bucket_name)
    obj = bucket.Object(key)
    response = obj.get()
    return Image.open(response['Body'])
