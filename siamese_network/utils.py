import os
import requests
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import cv2

IMAGE_PIXELS = 224
IMAGE_CHANNELS = 3


def process_dataset(json_dataset, predict_target):
    image_pair_urls = [(image_pair['representativeData']['image1']['imageUrl'],
                        image_pair['representativeData']['image2']['imageUrl'])
                       for image_pair in json_dataset['data']['results']]

    answers = []

    if not predict_target:
        answers = [int(image_pair['answers'][0]['answer'][0]['id']) for image_pair in json_dataset['data']['results']]
        answers = np.array(answers, dtype='int')
    image_pairs = process_images(image_pair_urls)

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
    first_urls, second_urls = zip(*url_pairs)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        first_images = np.array(list(executor.map(load_and_preprocess_image, first_urls)))
        second_images = np.array(list(executor.map(load_and_preprocess_image, second_urls)))

    first_images = np.vstack(first_images)
    second_images = np.vstack(second_images)

    return [first_images, second_images]


def save_predictions(json_dataset, predictions, predictions_path):
    task_ids = [image_pair['taskId'] for image_pair in json_dataset['data']['results']]

    predictions_df = pd.DataFrame({
        'taskId': task_ids,
        'answer': predictions
    })

    predictions_df.to_csv(predictions_path, index=False)
