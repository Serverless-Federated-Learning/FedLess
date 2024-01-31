import csv
import numpy as np
import os
from processing_utils import get_mel_and_label


# mapp client number to sample list
def get_fedscale_client_file_lists(mapping_file_path):
    fedscale_client_map = {}
    fedscale_unique_client_ids = {}
    read_first = True
    print("fetching fedscale client mappings")
    with open(mapping_file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        for row in csv_reader:
            if read_first:
                read_first = False
            else:
                fedscale_client_id, sample_name = row[0], row[1]
                if fedscale_client_id not in fedscale_unique_client_ids:
                    # add client idx to list of unique ids
                    fedscale_unique_client_ids[fedscale_client_id] = len(
                        fedscale_unique_client_ids
                    )
                    fedscale_client_map[fedscale_client_id] = []
                fedscale_client_map[fedscale_client_id] = fedscale_client_map[
                    fedscale_client_id
                ] + [sample_name]

        print(len(fedscale_unique_client_ids))
        return fedscale_client_map


def get_fedless_client_map(fedscale_client_map, ratio):

    fedless_client_map = {}
    fedless_client_idx = 0
    fedscale_mapping_idx = 0
    fedscale_clientid_list = list(fedscale_client_map)
    print("mapping fedscale clients to fedles clients with ratio", ratio)
    while fedscale_mapping_idx < len(fedscale_clientid_list):
        print("mapping client with idx:", fedless_client_idx)
        r = 0
        fedless_client_map[fedless_client_idx] = []
        while r < ratio and fedscale_mapping_idx < len(fedscale_clientid_list):
            # print("trying ratio:", r, fedscale_mapping_idx)
            fedscale_client_idx = fedscale_clientid_list[fedscale_mapping_idx]
            fedless_client_map[fedless_client_idx] = np.append(
                fedless_client_map[fedless_client_idx],
                fedscale_client_map[fedscale_client_idx],
            )

            # print('map: ',len(fedless_client_map[fedless_client_idx]))

            fedscale_mapping_idx += 1
            r += 1
        # print("-- done client with idx:", fedless_client_idx)
        fedless_client_idx += 1
    return fedless_client_map


def createZip(dataset_path, client_files, output_name):
    data = []
    labels = []
    for file_name in client_files:
        img_data, img_label = get_mel_and_label(os.path.join(dataset_path, file_name))
        data.append(img_data.numpy())
        labels.append(img_label.numpy())
    data = np.array(data)
    labels = np.array(labels)
    np.savez(output_name, data=data, labels=labels)


MAPPING_PATH = "./google_speech/client_data_mapping/"
DATASET_PATH = "./google_speech"
# ratio to the number of clients: each fedless client is responsible for the data of 8 fedscale clients
# 8 for train 0-270
# 1 for val 0-214
# 1 for test 0-215
# ratio = 1
# fedscale_client_map = get_fedscale_client_file_lists(MAPPING_PATH + f"{data_type}.csv")
# fedless_client_map = get_fedless_client_map(fedscale_client_map, ratio)


for data_type, ratio in zip(["train", "test", "val"], [8, 1, 1]):
    print(f"prep data for: {data_type}")

    save_path = f"./google_speech/npz/{data_type}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fedscale_client_map = get_fedscale_client_file_lists(
        MAPPING_PATH + f"{data_type}.csv"
    )
    fedless_client_map = get_fedless_client_map(fedscale_client_map, ratio)
    print(len(fedless_client_map))
    s = 0
    for i in fedless_client_map:
        s += len(fedless_client_map[i])
        createZip(
            DATASET_PATH + f"/{data_type}",
            fedless_client_map[i],
            f"{save_path}/client_{i}.npz",
        )
        print(f"saved data for client '{i}'")

# make global test data
