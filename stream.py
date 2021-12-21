#! /usr/bin/python3

import time
import json
import pickle
import socket
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Run using python3 stream.py to use CIFAR dataset and default batch_size as 100
# Run using python3 stream.py -f <input_file> -b <batch_size> to use a custom file/dataset and batch size
# Run using python3 stream.py -e True to stream endlessly in a loop
parser = argparse.ArgumentParser(
    description='Streams a file to a Spark Streaming Context')
parser.add_argument('--file', '-f', help='File to stream', required=False,
                    type=str, default="cifar")    # path to file for streaming
parser.add_argument('--batch-size', '-b', help='Batch size',
                    required=False, type=int, default=100)  # default batch_size is 100
parser.add_argument('--endless', '-e', help='Enable endless stream',
                    required=False, type=bool, default=False)  # looping disabled by default

TCP_IP = "localhost"
TCP_PORT = 6100


def connectTCP():   # connect to the TCP server -- there is no need to modify this function
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((TCP_IP, TCP_PORT))
    s.listen(1)
    print(f"Waiting for connection on port {TCP_PORT}...")
    connection, address = s.accept()
    print(f"Connected to {address}")
    return connection, address


# separate function to stream CIFAR batches since the format is different
def sendCIFARBatchFileToSpark(tcp_connection, input_batch_file):
    # load the entire dataset
    with open(f'cifar/{input_batch_file}', 'rb') as batch_file:
        batch_data = pickle.load(batch_file, encoding='bytes')

    # obtain image data and labels
    data = batch_data[b'data']
    data = list(map(np.ndarray.tolist, data))
    labels = batch_data[b'labels']
    # setting feature size to form the payload later
    feature_size = len(data[0])
    # iterate over batches of size batch_size
    for image_index in tqdm(range(0, len(data)-batch_size+2, batch_size)):
        # load batch of images
        image_data_batch = data[image_index:image_index+batch_size]
        image_label = labels[image_index:image_index +
                             batch_size]        # load batch of labels
        payload = dict()
        for mini_batch_index in range(len(image_data_batch)):
            payload[mini_batch_index] = dict()
            for feature_index in range(feature_size):  # iterate over features
                payload[mini_batch_index][f'feature{feature_index}'] = image_data_batch[mini_batch_index][feature_index]
            payload[mini_batch_index]['label'] = image_label[mini_batch_index]
        # print(payload)    # uncomment to see the payload being sent
        # encode the payload and add a newline character (do not forget the newline in your dataset)
        send_batch = (json.dumps(payload) + '\n').encode()
        try:
            tcp_connection.send(send_batch)  # send the payload to Spark
        except BrokenPipeError:
            print(
                "Either batch size is too big for the dataset or the connection was closed")
        except Exception as error_message:
            print(f"Exception thrown but was handled: {error_message}")
        time.sleep(5)


def streamCIFARDataset(tcp_connection, dataset_type='cifar'):
    print("Starting to stream CIFAR data")
    CIFAR_BATCHES = [
        'data_batch_1',
        'data_batch_2',   # uncomment to stream the second training dataset
        'data_batch_3',   # uncomment to stream the third training dataset
        'data_batch_4',   # uncomment to stream the fourth training dataset
        'data_batch_5',    # uncomment to stream the fifth training dataset
        # 'test_batch'      # uncomment to stream the test dataset
    ]
    for batch in CIFAR_BATCHES:
        sendCIFARBatchFileToSpark(tcp_connection, batch)
        time.sleep(5)


def sendPokemonBatchFileToSpark(tcp_connection, input_batch_file):
    # load the entire dataset
    with open(f'pokemon/{input_batch_file}.pickle', 'rb') as batch_file:
        batch_data = pickle.load(batch_file)

    # obtain image data and labels
    data = batch_data['img']
    labels = batch_data['label']
    # iterate over batches of size batch_size
    for image_index in tqdm(range(0, len(data)-batch_size+2, batch_size)):
        # load batch of images
        image_data_batch = data[image_index:image_index+batch_size]
        image_label = labels[image_index:image_index +
                             batch_size]        # load batch of labels
        payload = dict()
        for mini_batch_index in range(len(image_data_batch)):
            payload[mini_batch_index] = dict()
            payload[mini_batch_index]["img"] = image_data_batch[mini_batch_index]
            # if you want to flatten out the matrix, use payload[mini_batch_index]["img"] = np.asarray(image_data_batch[mini_batch_index]).flatten().tolist()
            payload[mini_batch_index]['label'] = image_label[mini_batch_index]
        # print(payload)    # uncomment to see the payload being sent
        # encode the payload and add a newline character (do not forget the newline in your dataset)
        send_batch = (json.dumps(payload) + '\n').encode()
        try:
            tcp_connection.send(send_batch)  # send the payload to Spark
        except BrokenPipeError:
            print(
                "Either batch size is too big for the dataset or the connection was closed")
        except Exception as error_message:
            print(f"Exception thrown but was handled: {error_message}")
        time.sleep(5)


def streamPokemonDataset(tcp_connection, dataset_type='pokemon'):
    print("Starting to stream Pokemon data")
    POKEMON_BATCHES = [
        'train_batch_1',
        # 'train_batch_2',   # uncomment to stream the second training dataset
        # 'train_batch_3',   # uncomment to stream the third training dataset
        # 'train_batch_4',   # uncomment to stream the fourth training dataset
        # 'train_batch_5',    # uncomment to stream the fifth training dataset
        # 'test_batch'      # uncomment to stream the test dataset
    ]
    for batch in POKEMON_BATCHES:
        sendPokemonBatchFileToSpark(tcp_connection, batch)
        time.sleep(5)


def streamDataset(tcp_connection, dataset_type):    # function to stream a dataset
    # this is the function you need to recreate to work with custom datasets
    # if your dataset has multiple files (train, test, etc), modify and use this function to stream your dataset
    print(f"Starting to stream {dataset_type} dataset")
    DATASETS = [    # list of files in your dataset to stream
        "train",
        # "test"    # uncomment to stream the test dataset
    ]
    for dataset in DATASETS:
        streamCSVFile(tcp_connection, f'{dataset_type}/{dataset}.csv')
        time.sleep(5)


def streamCSVFile(tcp_connection, input_file):    # stream a CSV file to Spark
    '''
    Each batch is streamed as a JSON file and has the following shape. 
    The outer indices are the indices of each row in a batch and go from 0 - batch_size-1
    The inner indices are the indices of each column in a row and go from 0 - feature_size-1

    {
        '0':{
            'feature0': <value>,
            'feature1': <value>,
            ...
            'featureN': <value>,
        }
        '1':{
            'feature0': <value>,
            'feature1': <value>,
            ...
            'featureN': <value>,
        }
        ...
        'batch_size-1':{
            'feature0': <value>,
            'feature1': <value>,
            ...
            'featureN': <value>,
        }
    }
    '''

    df = pd.read_csv(input_file)  # load the entire dataset
    values = df.values.tolist()  # obtain the values of the dataset
    # loop through batches of size batch_size lines
    for i in tqdm(range(0, len(values)-batch_size+2, batch_size)):
        send_data = values[i:i+batch_size]  # load batch of rows
        payload = dict()    # create a payload
        # iterate over the batch
        for mini_batch_index in range(len(send_data)):
            payload[mini_batch_index] = dict()  # create a record
            # iterate over the features
            for feature_index in range(len(send_data[0])):
                # add the feature to the record
                payload[mini_batch_index][f'feature{feature_index}'] = send_data[mini_batch_index][feature_index]
        # print(payload)    # uncomment to see the payload being sent
        # encode the payload and add a newline character (do not forget the newline in your dataset)
        send_batch = (json.dumps(payload) + '\n').encode()
        try:
            tcp_connection.send(send_batch)  # send the payload to Spark
        except BrokenPipeError:  # this indicates that the message length of the payload is more than what is allowed via TCP
            print(
                "Either batch size is too big for the dataset or the connection was closed")
        except Exception as error_message:
            print(f"Exception thrown but was handled: {error_message}")
        time.sleep(5)


def streamFile(tcp_connection, input_file):  # stream a newline delimited file to Spark
    '''
    Each batch is streamed as newline delimited text and has the following shape.

    line1\n
    line2\n
    ...
    '''
    with open(input_file, 'r') as file:
        data = file.readlines()  # open the file and read every line
        total_lines = len(data)
        # loop through batches of size batch_size lines
        for i in tqdm(range(0, total_lines-batch_size+2, batch_size)):
            send_data = data[i:i+batch_size]    # load batch of lines
            # encode the payload and add a newline character (do not forget the newline in your dataset)
            send_batch = (json.dumps(send_data) + '\n').encode()
            try:
                tcp_connection.send(send_batch)  # send the payload to Spark
            except BrokenPipeError:
                print(
                    "Either batch size is too big for the dataset or the connection was closed")
            except Exception as error_message:
                print(f"Exception thrown but was handled: {error_message}")
            time.sleep(5)


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    input_file = args.file
    batch_size = args.batch_size
    endless = args.endless

    tcp_connection, _ = connectTCP()

    # to stream a custom dataset, uncomment the elif block and create your own dataset streamer function (or modify the existing one)
    if input_file == "cifar":
        _function = streamCIFARDataset
    elif input_file == "pokemon":
        _function = streamPokemonDataset
    elif input_file in ["crime", "sentiment", "spam"]:
        _function = streamDataset
    # elif input_file == "my dataset":
    #     _function = streamMyDataset
    else:
        _function = streamFile

    if endless:
        while True:
            _function(tcp_connection, input_file)
    else:
        _function(tcp_connection, input_file)

    tcp_connection.close()

# Setup your own dataset streamer by following the examples above.
# If you wish to stream a single newline delimited file, use streamFile()
# If you wish to stream a CSV file, use streamCSVFile()
# If you wish to stream any other type of file(JSON, XML, etc.), write an appropriate function to load and stream the file
