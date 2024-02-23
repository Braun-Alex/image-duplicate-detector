import os
import json
from siamese_network.utils import process_dataset, save_predictions, IMAGE_PIXELS, IMAGE_CHANNELS
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras import applications, layers, models, Input, optimizers, losses, metrics


def create_convnext_network(image_shape):
    convnext_network = applications.ConvNeXtBase(include_top=False, input_shape=image_shape, pooling='avg')
    convnext_network.trainable = False
    return convnext_network


def create_siamese_network(image_shape=(IMAGE_PIXELS, IMAGE_PIXELS, IMAGE_CHANNELS)):
    first_input = Input(shape=image_shape, name='first_input')
    second_input = Input(shape=image_shape, name='second_input')

    base_network = create_convnext_network(image_shape)

    first_branch = base_network(first_input)
    second_branch = base_network(second_input)

    merged_network = layers.concatenate([first_branch, second_branch], name='concatenation_layer')
    merged_network = layers.Dense(128, activation='relu')(merged_network)
    network_outputs = layers.Dense(1, activation='sigmoid')(merged_network)
    siamese_network = models.Model(inputs=[first_input, second_input], outputs=network_outputs, name='siamese_network')

    siamese_network.compile(optimizer=optimizers.Adam(), loss=losses.BinaryCrossentropy(),
                            weighted_metrics=[metrics.Precision(), metrics.Recall(), metrics.F1Score()])
    return siamese_network


def process_network_dataset(dataset_path, predict_target):
    with open(dataset_path) as dataset:
        network_dataset = json.load(dataset)
    if not predict_target:
        [first_images, second_images], answers = process_dataset(network_dataset, predict_target=False)

        first_images = Dataset.from_tensor_slices(first_images)
        second_images = Dataset.from_tensor_slices(second_images)
        answers = Dataset.from_tensor_slices(answers).map(lambda answer: tf.cast(answer, tf.float32))

        xy = Dataset.zip(((first_images, second_images), answers)).batch(32)

        return xy
    else:
        [first_images, second_images], _ = process_dataset(network_dataset, predict_target=True)

        first_images = Dataset.from_tensor_slices(first_images)
        second_images = Dataset.from_tensor_slices(second_images)

        image_pairs = Dataset.zip((first_images, second_images)).batch(32)

        return image_pairs


def train_network(dataset_path):
    if not os.path.exists(dataset_path):
        print(f"The training dataset {dataset_path} does not exist!")

    print("Processing the training dataset...")
    xy = process_network_dataset(dataset_path, False)

    print("Instantiating the Siamese ConvNeXt network...")
    siamese_network = create_siamese_network()

    print("Training the network...")
    siamese_network.fit(xy, epochs=3, workers=os.cpu_count(),
                        use_multiprocessing=True)

    siamese_network.save('siamese_network.keras')
    print("The network has been trained successfully!")


def test_network(network_path, dataset_path):
    if not os.path.exists(network_path):
        print(f"The network {network_path} does not exist!")

    if not os.path.exists(dataset_path):
        print(f"The testing dataset {dataset_path} does not exist!")

    print("Importing the trained Siamese ConvNeXt network...")
    network = models.load_model(network_path)

    print("Processing the testing dataset...")
    xy = process_network_dataset(dataset_path, False)

    print("Testing the network...")
    loss, w_precision, w_recall, w_f1_score = network.evaluate(xy, workers=os.cpu_count(),
                                                               use_multiprocessing=True)

    print(f"The network has been tested!\nLoss: {loss}\nWeighted precision: {w_precision}\n"
          f"Weighted recall: {w_recall}\nWeighted F1-score: {w_f1_score}")


def predict_using_network(network_path, dataset_path, predictions_path):
    if not os.path.exists(network_path):
        print(f"The network {network_path} does not exist!")

    if not os.path.exists(dataset_path):
        print(f"The dataset {dataset_path} does not exist!")

    print("Importing the trained Siamese ConvNeXt network...")
    network = models.load_model(network_path)

    print("Processing the dataset...")
    image_pairs = process_network_dataset(dataset_path, predict_target=True)

    print("Predicting by the network...")
    predictions = network.predict(image_pairs, workers=os.cpu_count(), use_multiprocessing=True)

    save_predictions(dataset_path, predictions, predictions_path)
    print("The predictions have been successfully saved!")
