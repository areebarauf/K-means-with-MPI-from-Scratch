from math import sqrt, floor
import numpy as np
import pandas as pd
from mpi4py import MPI
import matplotlib.pyplot as plt




def init_centroids(dataset, clusters):
    n = np.shape(dataset)[1]
    m = np.shape(dataset)[0]
    centroids = np.mat(np.zeros((clusters, n)))
    composite = np.mat(np.sum(dataset, axis=1))
    dataset = np.append(composite.T, dataset, axis=1)
    dataset.sort(axis=0)
    split_size = floor(m / clusters)
    vector_func = np.vectorize(_get_mean)
    for j in range(k):
        if j == k - 1:
            centroids[j:] = vector_func(np.sum(dataset[j * split_size:, 1:], axis=0), split_size)
        else:
            centroids[j:] = vector_func(np.sum(dataset[j * split_size:(j + 1) * split_size, 1:], axis=0), split_size)
    return centroids


def _get_mean(sums, split_size):
    value = sums / split_size
    return value


def eulcid_dist(datapoint, centroid):
    dist = np.linalg.norm(datapoint - centroid)
    return dist


def minimum(arr):
    min_index = np.argmin(arr)
    return min_index + 1


def get_cluster_filtered(dictA, clusters):
    resultant = []
    for i in range(clusters):
        resultant_list = [[keys for keys, values in dictA.items() if values == i + 1]]
        resultant.append(resultant_list)
    return resultant


def speedup_graph(serial_time, process_time):
    times_plot = []
    speed_up_time = serial_time / process_time
    times_plot.append(speed_up_time)
    return times_plot


def label_assignment(dataset, centroids):
    distance_array = np.array([])
    dicts = {}
    distances = []
    for i, row in enumerate(dataset):
        distance1 = []
        for j in range(len(centroids)):
            distance1.append(eulcid_dist(row, centroids[j]))
            distances.append(distance1)
            min_index = minimum(distance1)
            dicts[i] = min_index
        distance_array = np.append(distance_array, distances)
    return dicts


def calculate_sum(keys_list, dataset):
    updated_center = []
    for i, row_ in enumerate(keys_list):
        for cluster in row_:
            converted_df = pd.DataFrame(dataset)
            df2 = converted_df.loc[converted_df.index[cluster]]
            df2_count = len(df2)
            sum_df = df2.sum(axis=0)
            df_tr = sum_df.transpose()
            transposed_mean = df_tr.to_numpy()
        updated_center.append([df2_count, transposed_mean])
    return updated_center


def return_sum(dataset, centroids):
    label_dic = label_assignment(dataset, centroids)
    list_res = get_cluster_filtered(label_dic, len(centroids))
    return_sum_ = calculate_sum(list_res, dataset)
    return return_sum_


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
root = 0
k = 3
start_time = MPI.Wtime()

if rank == root:
    df = pd.read_csv('Absenteeism_at_work.csv')
    # df1 = df.iloc[:30, :2]
    convert_df_array = df.to_numpy()
    # print('Array Data:', convert_df_array.shape)

    Array_chunks = np.array_split(convert_df_array, size)
    Initializing_centroids = init_centroids(convert_df_array, k)
    # ('Initializing_centroids:', Initializing_centroids)

else:
    Array_chunks = None
    Initializing_centroids = None
    step = None

scatter_Data = comm.scatter(Array_chunks, root=0)
broadcast_centroids = comm.bcast(Initializing_centroids, root=0)

goahead = True
iteration_count = 0

while goahead:
    current_centroid = broadcast_centroids
    update_centroid = []
    ind_sum = []
    count_cluster = []
    # this method returns which instance goes in which rank
    label_dict = label_assignment(scatter_Data, broadcast_centroids)
    # this method returns number of instances in each cluster and their attribute sum
    sum_array = return_sum(scatter_Data, broadcast_centroids)

    counts = np.zeros(k)

    # here separating the count and sum vector in two different variables
    for i in range(len(sum_array)):
        counts[i] = sum_array[i][0]
        ind_sum.append(sum_array[i][1])
    ind_sum = np.array(ind_sum)

    count_cluster = np.array(counts)

    # using reduce to get the sum with count array from each rank
    summed_counts = comm.reduce(count_cluster, root=root)
    resultant_sums = comm.reduce(ind_sum, root=root)

    if summed_counts is not None:
        if resultant_sums is not None:
            for x in range(len(summed_counts)):
                update_centroid.append(resultant_sums[x] / summed_counts[x])
                # print('Updated centroid:', update_centroid)

    broadcast_centroids = comm.bcast(update_centroid, root=root)

    iteration_count += 1
    if np.array_equal(broadcast_centroids, current_centroid):
        goahead = False
    else:
        continue

end_time = MPI.Wtime()
Net_time = end_time - start_time
all_times = comm.gather(Net_time, root=0)

if rank == root:
    ser_time = 319.960
    times = np.vstack(all_times)
    time_sum = np.sum(times)
    speedup_time = speedup_graph(ser_time, time_sum)
    print('Speed+up time array:', speedup_time)
    print('Total Time for processes is Net_time=%.3f' % time_sum)
    print(f'Global Centroids:{broadcast_centroids} and number of iterations:{iteration_count}')
