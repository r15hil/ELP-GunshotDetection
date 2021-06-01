import os
import sys
import tensorflow as tf
import numpy as np
from PIL import Image
from object_detection import ObjectDetection
import pandas as pd
from pydub import AudioSegment
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib import figure
import time
import json
import cv2
import PIL
#import skimage.io




MODEL_FILENAME = r'C:\Users\olihu\Downloads\custom-vision-model\custom-vision-model\model.pb'
LABELS_FILENAME = r'C:\Users\olihu\Downloads\custom-vision-model\custom-vision-model\labels.txt'


class TFObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow SavedModel"""

    def __init__(self, model_filename, labels):
        super(TFObjectDetection, self).__init__(labels)
        model = tf.saved_model.load(os.path.dirname(model_filename))
        self.serve = model.signatures['serving_default']

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis, :, :, (2, 1, 0)]  # RGB -> BGR
        inputs = tf.convert_to_tensor(inputs)
        outputs = self.serve(inputs)
        return np.array(outputs['outputs'][0])

def main(file_path,image_id):
    # Load labels
    with open(LABELS_FILENAME, 'r') as f:
        labels = [l.strip() for l in f.readlines()]

    od_model = TFObjectDetection(MODEL_FILENAME, labels)

    detected = {
        "filename": [],
        "offset": [],
        "confidence": [],
    }


    num_detected = 0
    overlap = 6 * 1000
    twelve = 12 * 1000
    missing =0
    image = Image.open(file_path)
    predictions = od_model.predict_image(image)
    print(predictions)
    hit = False
    save_to = file_path.replace('images','testing_images' )
    for k in predictions:
        if (k['probability'] >= 0.50):
            hit = True
            num_detected += 1
            print(num_detected)

            print(k)
            detected['filename'].append(image_id)

            detected['confidence'].append(k['probability'])
            ratio_w = 640 / 432
            ratio_h = 480 / 288
            image = cv2.imread(file_path)
            top_left = (int(k['boundingBox']['left'] * 640/ ratio_w), int(k['boundingBox']['top'] * 480/ ratio_h))
            bottom_right = (int((k['boundingBox']['left'] + k['boundingBox']['width']) * 640/ ratio_w),
                            int((k['boundingBox']['top'] + k['boundingBox']['height']) * 480/ ratio_h))
            print(top_left, bottom_right)
            colour = (255, 0, 0)
            thickness = 2
            image = cv2.rectangle(image, bottom_right, top_left, colour, thickness)
            cv2.imwrite(save_to, image)

        elif (k['probability'] >= 0.5):
            print(k['probability'], " file: %s .jpg " %(image_id))
    if not hit:
        image = cv2.imread(file_path)
        cv2.imwrite(save_to, image)
        missing =1
    return missing
if __name__ == '__main__':


    #imgall.show()
    #cv2.imshow('image', img)
    missing_cout =0
    image_ids = open(r'C:\Users\olihu\Downloads\custom-vision-model\custom-vision-model\python\gun_train.txt').read().strip().splitlines()
    print(image_ids)
    for i in range(len(image_ids)):
        image_path = r'C:\Users\olihu\Downloads\custom-vision-model\custom-vision-model\cache\images\%s.jpg' % (image_ids[i])
        print(image_path)
        img = cv2.imread(image_path)
        #imgall.show()
        #cv2.imshow('image', img)
        #cv2.waitKey()

        missing = main(image_path,image_ids[i])
        missing_cout = missing_cout +missing
    # This closes all open windows
    # Failure to place this will cause your program to hang
    #cv2.destroyAllWindows()
    print('missing_count',missing_cout)
