import os
import json
from siamese_network.utils import get_dataset_length, process_dataset, save_predictions, IMAGE_PIXELS, IMAGE_CHANNELS
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras import applications, layers, models, Input, optimizers, metrics, backend, Model


def cosine_similarity(vectors):
    x, y = vectors
    x = backend.l2_normalize(x, axis=-1)
    y = backend.l2_normalize(y, axis=-1)
    return -backend.sum(x * y, axis=-1, keepdims=True)


def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = backend.square(y_pred)
    margin_square = backend.square(backend.maximum(margin - y_pred, 0))
    return backend.mean(y_true * square_pred + (1 - y_true) * margin_square)


def create_base_network(image_shape):
    base_network = applications.ConvNeXtBase(include_top=False, input_shape=image_shape)
    base_network.trainable = False
    x = base_network.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    return Model(base_network.input, x)


def create_siamese_network(image_shape=(IMAGE_PIXELS, IMAGE_PIXELS, IMAGE_CHANNELS)):
    left_input = Input(shape=image_shape)
    right_input = Input(shape=image_shape)

    base_network = create_base_network(image_shape)

    left_branch = base_network(left_input)
    right_branch = base_network(right_input)

    distance = layers.Lambda(cosine_similarity)([left_branch, right_branch])
    prediction = layers.Dense(1, activation='sigmoid')(distance)

    siamese_network = models.Model(inputs=[left_input, right_input], outputs=prediction)

    siamese_network.compile(optimizer=optimizers.Adam(), loss=contrastive_loss,
                            weighted_metrics=[metrics.Precision(), metrics.Recall(), metrics.F1Score(threshold=0.5)])
    return siamese_network


def process_network_dataset(dataset_path, predict_target, batch_size=None, start_index=None, end_index=None):
    with open(dataset_path) as dataset:
        network_dataset = json.load(dataset)
    if not predict_target:
        [left_images, right_images], answers = process_dataset(network_dataset, predict_target=False)

        left_images = Dataset.from_tensor_slices(left_images)
        right_images = Dataset.from_tensor_slices(right_images)
        answers = Dataset.from_tensor_slices(answers).map(lambda answer: tf.cast(answer, tf.float32))

        xy = Dataset.zip(((left_images, right_images), answers)).batch(batch_size)

        return xy
    else:
        [left_images, right_images], _ = process_dataset(network_dataset, predict_target=True, start_index=start_index,
                                                         end_index=end_index)

        return [left_images, right_images]


def train_network(dataset_path, epochs, network_name, batch_size):
    if not os.path.exists(dataset_path):
        print(f"The training dataset {dataset_path} does not exist!")

    print("Processing the training dataset...")
    xy = process_network_dataset(dataset_path, predict_target=False, batch_size=batch_size)

    print("Instantiating the Siamese ConvNeXt network...")
    siamese_network = create_siamese_network()

    print("Training the network...")
    siamese_network.fit(xy, epochs=epochs, workers=os.cpu_count(),
                        use_multiprocessing=True)

    siamese_network.save(f"saved_model/{network_name}")
    print("The network has been trained successfully!")


def test_network(network_path, dataset_path, batch_size):
    if not os.path.exists(network_path):
        print(f"The network {network_path} does not exist!")

    if not os.path.exists(dataset_path):
        print(f"The testing dataset {dataset_path} does not exist!")

    print("Importing the trained Siamese ConvNeXt network...")
    network = models.load_model(network_path, custom_objects={'contrastive_loss': contrastive_loss})

    print("Processing the testing dataset...")
    xy = process_network_dataset(dataset_path, predict_target=False, batch_size=batch_size)

    print("Testing the network...")
    results = network.evaluate(xy, workers=os.cpu_count(), use_multiprocessing=True)

    print(f"The network has been tested!\nTest loss: {results[0]}\nTest precision: {results[1]}"
          f"\nTest recall: {results[2]}\nTest F1-score: {results[3]}")


def predict_using_network(network_path, dataset_path, predictions_path, batch_size, chunks):
    if not os.path.exists(network_path):
        print(f"The network {network_path} does not exist!")

    if not os.path.exists(dataset_path):
        print(f"The dataset {dataset_path} does not exist!")

    print(f"Importing the trained Siamese ConvNeXt network...")
    network = models.load_model(network_path, custom_objects={'contrastive_loss': contrastive_loss})

    dataset_length = get_dataset_length(dataset_path)
    chunk_size = dataset_length // chunks
    all_predictions = []

    for chunk in range(chunks):
        start_index = chunk * chunk_size
        if chunk != chunks - 1:
            end_index = start_index + chunk_size
        else:
            end_index = dataset_length

        print(f"Processing the chunk {chunk + 1} of the dataset...")
        left_images, right_images = process_network_dataset(dataset_path, predict_target=True, start_index=start_index,
                                                            end_index=end_index)

        print(f"Predicting by the network for the chunk {chunk + 1} of the dataset...")
        predictions = network.predict([left_images, right_images], batch_size=batch_size, workers=os.cpu_count(),
                                      use_multiprocessing=True)

        all_predictions.extend(predictions)

    save_predictions(dataset_path, all_predictions, predictions_path)
    print("The predictions have been successfully saved!")
