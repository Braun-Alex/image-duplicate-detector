from siamese_network.network import train_network, test_network, predict_using_network


def main():
    print("Image deduplication using Siamese ConvNeXt Network")
    print("Choose a command to execute")
    print("1) Train the Siamese network")
    print("2) Test the Siamese network")
    print("3) Make predictions using the Siamese network")

    command = input("Enter the number of the command: ")
    batch_size = int(input("Enter the batch size for the dataset processing: "))

    if command == "1":
        dataset_path = input("Enter the path to the training JSON dataset: ")
        epochs = int(input("Enter the number of epochs for the network training: "))
        network_name = input("Enter the name of the network to save: ")
        train_network(dataset_path, epochs, network_name, batch_size)
    elif command == "2":
        network_path = input("Enter the path to the trained Keras model: ")
        dataset_path = input("Enter the path to the testing JSON dataset: ")
        test_network(network_path, dataset_path, batch_size)
    elif command == "3":
        network_path = input("Enter the path to the trained Keras model: ")
        dataset_path = input("Enter the path to the JSON dataset for predictions: ")
        predictions_path = input("Enter a filename to save the prediction results in CSV format: ")
        chunks = int(input("Enter the number of chunks to split the dataset while processing: "))
        predict_using_network(network_path, dataset_path, predictions_path, batch_size, chunks)
    else:
        print("Entered the invalid command. Exiting...")


if __name__ == '__main__':
    main()
