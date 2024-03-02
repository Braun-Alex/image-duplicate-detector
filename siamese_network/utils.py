import os
import requests
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

IMAGE_PIXELS = 224
IMAGE_CHANNELS = 3


def get_dataset_length(dataset_path):
    with open(dataset_path) as dataset:
        network_dataset = json.load(dataset)
    return len(network_dataset['data']['results'])


def process_dataset(json_dataset, predict_target, start_index=None, end_index=None):
    answers = []

    if not predict_target:
        image_pair_urls = [(image_pair['representativeData']['image1']['imageUrl'],
                            image_pair['representativeData']['image2']['imageUrl'])
                           for image_pair in json_dataset['data']['results']]

        answers = [int(image_pair['answers'][0]['answer'][0]['id']) for image_pair
                   in json_dataset['data']['results']]
        answers = np.array(answers, dtype='int')

    else:
        image_pair_urls = [(image_pair['representativeData']['image1']['imageUrl'],
                            image_pair['representativeData']['image2']['imageUrl'])
                           for image_pair in json_dataset['data']['results']]

    image_pairs = process_images(image_pair_urls[start_index:end_index])

    return image_pairs, answers


def load_image(url):
    response = requests.get(url)

    image = np.asarray(bytearray(response.content), dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image


def preprocess_image(image, new_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, new_size)

    image = cv2.convertScaleAbs(image, alpha=(1.0 / 255.0))
    image = np.expand_dims(image, axis=0)
    return image


def load_and_preprocess_image(url, new_size=(IMAGE_PIXELS, IMAGE_PIXELS)):
    image = load_image(url)
    image = preprocess_image(image, new_size)
    return image


def process_images(url_pairs):
    left_urls, right_urls = zip(*url_pairs)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        left_images = np.array(list(executor.map(load_and_preprocess_image, left_urls)))
        right_images = np.array(list(executor.map(load_and_preprocess_image, right_urls)))

    left_images = np.vstack(left_images)
    right_images = np.vstack(right_images)

    return [left_images, right_images]


def save_predictions(json_dataset, predictions, predictions_path):
    with open(json_dataset) as dataset:
        network_dataset = json.load(dataset)

    task_ids = [image_pair['taskId'] for image_pair in network_dataset['data']['results']]

    predictions = tf.round(np.array(predictions).flatten()).numpy().astype(int)

    predictions_df = pd.DataFrame({
        'taskId': task_ids,
        'answer': predictions
    })

    predictions_df.to_csv(predictions_path, index=False)
