import tensorflow as tf
from siamese_network.network import train_network, test_network, predict_using_network
import requests
import numpy as np
import pandas as pd
import cv2


def main():
    print("Image deduplication using Siamese ConvNeXt Network")
    print("Choose a command to execute")
    print("1) Train the Siamese network")
    print("2) Test the Siamese network")
    print("3) Make predictions using the Siamese network")

    command = input("Enter the number of the command: ")

    if command == "1":
        dataset_path = input("Enter the path to the training JSON dataset: ")
        train_network(dataset_path)
    elif command == "2":
        network_path = input("Enter the path to the trained Keras model")
        dataset_path = input("Enter the path to the testing JSON dataset")
        test_network(network_path, dataset_path)
    elif command == "3":
        network_path = input("Enter the path to the trained Keras model")
        dataset_path = input("Enter the path to the JSON dataset for predictions")
        predictions_path = input("Enter a filename to save the prediction results in CSV format")
        predict_using_network(network_path, dataset_path, predictions_path)
    else:
        print("Entered the invalid command. Exiting...")


if __name__ == '__main__':
    main()
