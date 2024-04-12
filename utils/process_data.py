import copy
import datetime
import gudhi as gd
import math
import networkx as nx
import numpy as np
import os
import os.path as osp
import pandas as pd
import pickle
import random
import time
import torch

from tqdm import tqdm

from sg2dgm import PersistenceImager as pimg


def add_line_graph_Edges(edge_list, nodes):
    for i in range(len(nodes)-1):
        for j in range(i+1, len(nodes)):
            edge_list.append([nodes[i], nodes[j]])
            edge_list.append([nodes[j], nodes[i]])

    return edge_list


def get_graph_info(g, compute_PD=True, is_directed=False):
    subgraph = g

    if len(subgraph.edges()) == 0:
        return 0

    edge_ind = 0
    filtration_val = []

    for edge in subgraph.edges():
        u1, v1 = edge[0], edge[1]
        if 'ind' not in subgraph[u1][v1]:
            filtration_val.append(subgraph[u1][v1]['weight'])
            subgraph[u1][v1]['ind'] = edge_ind
            edge_ind += 1

    if is_directed:
        source_edge_index = []
        for node in subgraph.nodes:
            edge_neighbors = []
            for n in subgraph.neighbors(node):
                edge_neighbors.append(subgraph[node][n]['ind'])

            if len(edge_neighbors) > 1:
                source_edge_index = add_line_graph_Edges(source_edge_index, edge_neighbors)

        sink_edge_index = []
        for node in subgraph.nodes:
            edge_neighbors = []
            for n in subgraph.predecessors(node):
                edge_neighbors.append(subgraph[n][node]['ind'])

            if len(edge_neighbors) > 1:
                sink_edge_index = add_line_graph_Edges(sink_edge_index, edge_neighbors)

        if len(source_edge_index) == 0 or len(sink_edge_index) == 0:
            return 0
        else:
            source_edge_index = torch.Tensor(source_edge_index).transpose(0, 1).long()
            sink_edge_index = torch.Tensor(sink_edge_index).transpose(0, 1).long()
    else:
        line_edge_index = []
        for node in subgraph.nodes:
            edge_neighbors = []
            for n in subgraph.neighbors(node):
                edge_neighbors.append(subgraph[node][n]['ind'])

            if len(edge_neighbors) > 1:
                line_edge_index = add_line_graph_Edges(line_edge_index, edge_neighbors)

        if len(line_edge_index) == 0:
            return 0
        else:
            line_edge_index = torch.Tensor(line_edge_index).transpose(0, 1).long()

    if compute_PD:
        edge_barcode = dowkerSourceFiltration(subgraph)
        edge_index = edge_barcode[:, :3]
        edge_num = edge_barcode.shape[0]

        barcode = np.zeros(shape=(edge_num, 4))
        for i in range(edge_num):
            u1, v1 = int(edge_barcode[i, 0]), int(edge_barcode[i, 1])
            edge_ind = subgraph[u1][v1]['ind']
            barcode[edge_ind, :] = edge_barcode[i, 3:]

        barcode0 = edge_barcode[:, 3:5]
        barcode1 = edge_barcode[:, 5:]
        pers_imager = pimg.PersistenceImager(resolution=5)
        pers_img = pers_imager.transform(np.concatenate((barcode0, barcode1))).reshape(-1)

        result = dict()
        result['barcode'] = barcode
        result['PI'] = pers_img
        result['filtration_val'] = filtration_val
        if is_directed:
            result['source_edge_index'] = source_edge_index
            result['sink_edge_index'] = sink_edge_index
        else:
            result['line_edge_index'] = line_edge_index
        result['edge_index'] = edge_index

        return result

    else:
        edge_list = sorted(subgraph.edges(data='weight'), key=lambda x: x[2])
        edge_index = np.array(edge_list)

        result = dict()
        result['filtration_val'] = filtration_val
        if is_directed:
            result['source_edge_index'] = source_edge_index
            result['sink_edge_index'] = sink_edge_index
        else:
            result['line_edge_index'] = line_edge_index
        result['edge_index'] = edge_index

        return result


def split_graph(weighted_edge_list):
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(weighted_edge_list)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    graph = nx.convert_node_labels_to_integers(graph)

    weights = nx.get_edge_attributes(graph, 'weight').values()
    max_weight = max(weights)
    min_weight = min(weights) - 1
    for u, v, weight in graph.edges(data='weight'):
        graph[u][v]['weight'] = (weight - min_weight) / (max_weight - min_weight)

    return graph


def read_weibo(name, edge_num_lower):
    dataset_path = '/DNDN/data1/weibo_dataset'
    sub_dataset_path = os.path.join(dataset_path, name)
    graph_dirs = get_files_in_directory(sub_dataset_path)

    weighted_edge_lists = []
    for i in tqdm(range(len(graph_dirs)), desc='processing dataset '+name):
        df = pd.read_csv(graph_dirs[i])
        df.sort_values('weight', inplace=True)

        data = df[['source', 'target', 'weight']].values.tolist()
        if len(data) < edge_num_lower:
            continue

        weighted_edge_lists.append(data)

    return sorted(weighted_edge_lists, key=lambda x: len(x))


def read_citation(dataset_name):
    if dataset_name=='Cit-HepPh':
        
        edge_list_path = os.path.join(dataset_path, 'cit-HepPh', 'Cit-HepPh.txt')
        node_weight_path = os.path.join(dataset_path, 'cit-HepPh', 'cit-HepPh-dates.txt')
    else: 
        edge_list_path = os.path.join(dataset_path, 'cit-HepTh', 'Cit-HepTh.txt')
        node_weight_path = os.path.join(dataset_path, 'cit-HepTh', 'Cit-HepTh-dates.txt')
        node_weight2_dir = os.path.join(dataset_path, 'cit-HepTh', 'cit-HepTh-abstracts')
        
    time_stamp_0 = datetime.datetime.strptime('1992-01-01', "%Y-%m-%d").timestamp()
    node_weight_dict = dict()
    with open(node_weight_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if not line.startswith('#'):
                node, date = line.strip().split('\t')
                node = int(node)
                weight = datetime.datetime.strptime(date, "%Y-%m-%d").timestamp()-time_stamp_0
                if node in node_weight_dict.keys():
                    if weight > node_weight_dict[node]:
                        continue
                node_weight_dict[node] = weight
                
    if dataset_name=='cit-HepTh':
        dir_list = []
        for dir_name in os.listdir(node_weight2_dir):
            file_dir = osp.join(node_weight2_dir, dir_name)
            dir_list.append(file_dir)
        
        for file_dir in dir_list:
            year = file_dir.split('/')[-1]
            for node_weight in os.listdir(file_dir):
                with open(osp.join(file_dir, node_weight), 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        if line.startswith('Date:'):
                            node = int(node_weight.split('.')[0])
        
                            date = line.split()
                            if date[1].endswith(',') or date[1].isalpha():
                                date = date[2:4]
                            elif '-' in date[1]:
                                date = date[1].split('-')[:2]
                            elif '/' in date[1]:
                                date = date[1].split('/')[:2]
                            elif ':' in date[4]:
                                if date[5].isdigit():
                                    date = date[2:4]
                                else:
                                    date = date[1:3]
                            else:
                                date = date[1:3]
        
                            date.append(year)
                            if date[0].isdigit():
                                if len(date[1]) > 3:
                                    date[1] = date[1][:3]
                                date = '{} {} {}'.format(date[0], date[1], date[2])
                            else:
                                if len(date[0]) > 3:
                                    date[0] = date[0][:3]
                                date = '{} {} {}'.format(date[1], date[0], date[2])
        
                            if all(char.isdigit() for char in date.split()):
                                weight = datetime.datetime.strptime(date, "%d %m %Y").timestamp()-time_stamp_0
                            else:
                                weight = datetime.datetime.strptime(date, "%d %b %Y").timestamp()-time_stamp_0
                            if node in node_weight_dict.keys():
                                if weight > node_weight_dict[node]:
                                    continue
                            node_weight_dict[node] = weight

    weighted_edge_list = []
    with open(edge_list_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if not line.startswith('#'):
                src, tgt = line.strip().split('\t')
                src, tgt = int(src), int(tgt)
                if src in node_weight_dict.keys():
                    weighted_edge_list.append((src, tgt, node_weight_dict[src]))

    return sorted(weighted_edge_list, key=lambda x: x[2])


def read_bitcoin(dataset_name):
    if dataset_name=='alpha':
        edge_list_path = os.path.join(dataset_path, 'soc-sign-bitcoin-alpha', 'soc-sign-bitcoinalpha.csv')
    else: 
        edge_list_path = os.path.join(dataset_path, 'soc-sign-bitcoin-otc', 'soc-sign-bitcoinotc.csv')
    

    weighted_edge_list = []
    df = pd.read_csv(edge_list_path, header=None)
    for index, row in df.iterrows():
        weighted_edge_list.append((int(row[0]), int(row[1]), float(row[3])))

    return sorted(weighted_edge_list, key=lambda x: x[2])


def read_question(dataset_name):
    if dataset_name=='askubuntu':
        edge_list_path = os.path.join(dataset_path, 'sx-askubuntu', 'sx-askubuntu.txt')
    elif dataset_name=='mathoverflow':
        edge_list_path = os.path.join(dataset_path, 'sx-mathoverflow', 'sx-mathoverflow.txt')
    elif dataset_name=='stackoverflow':
        edge_list_path = os.path.join(dataset_path, 'sx-stackoverflow', 'sx-stackoverflow.txt')
    else:
        edge_list_path = os.path.join(dataset_path, 'sx-superuser', 'sx-superuser.txt')
    weighted_edge_list = []
    with open(edge_list_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            src, tgt, weight = line.split()
            weighted_edge_list.append((int(src), int(tgt), int(weight)))

    return sorted(weighted_edge_list, key=lambda x: x[2])


def get_split_graph_list_big_nonoverlap(weighted_edge_list):
    def get_ed_idx(ed_idx):
        while ed_idx < total_edge_num and weighted_edge_list[ed_idx][2] == weighted_edge_list[ed_idx-1][2]:
            ed_idx -= 1
        return ed_idx

    small_graph_list = []
    big_graph_list = []

    total_edge_num = len(weighted_edge_list)
    cycle_step = math.floor(total_edge_num/big_graph_num)
    if cycle_step <= big_graph_edge_num+small_graph_edge_num:
        raise ValueError('graph is too small to satisfy the request of parameters.\n'
                         'cycle step: {}, big graph edge_num: {}, small graph edge num: {}'
                         .format(cycle_step, big_graph_edge_num, small_graph_edge_num))

    small_step = min(math.floor((cycle_step-big_graph_edge_num-small_graph_edge_num)/(small_big_ratio-1)), small_graph_edge_num)
    if small_step*(small_big_ratio-1)+small_graph_edge_num+big_graph_edge_num > cycle_step:
        raise ValueError('graph is too small to satisfy the request of parameters.\n'
                         'cycle step: {}, big graph edge_num: {}, small graph edge num: {}'
                         .format(cycle_step, big_graph_edge_num, small_graph_edge_num))

    cycle_st_idx = 0
    for i in range(big_graph_num):
        small_st_idx = cycle_st_idx
        for j in range(small_big_ratio):
            small_ed_idx = small_st_idx + small_graph_edge_num
            small_ed_idx = get_ed_idx(small_ed_idx)

            graph = split_graph(weighted_edge_list[small_st_idx:small_ed_idx])
            small_graph_list.append(graph)

            small_st_idx += small_step
            small_st_idx = get_ed_idx(small_st_idx)

        small_st_idx = small_ed_idx
        small_ed_idx = min(small_st_idx + big_graph_edge_num, cycle_st_idx + cycle_step)
        small_ed_idx = get_ed_idx(small_ed_idx)

        if small_ed_idx >= total_edge_num:
            graph = split_graph(weighted_edge_list[small_st_idx:])
        else:
            graph = split_graph(weighted_edge_list[small_st_idx:small_ed_idx])

        big_graph_list.append(graph)
        cycle_st_idx = small_ed_idx

    print('small graph edge num: {}, small graph step: {}, overlap rate: {}\nbig graph edge num: {}'
          .format(small_graph_edge_num, small_step, (small_graph_edge_num-small_step)/small_graph_edge_num,
                  big_graph_edge_num))
    graph_stat(small_graph_list)
    graph_stat(big_graph_list)

    return small_graph_list, big_graph_list


def get_split_graph_list_big_overlap(weighted_edge_list):
    def get_ed_idx(ed_idx):
        while ed_idx < total_edge_num and weighted_edge_list[ed_idx][2] == weighted_edge_list[ed_idx-1][2]:
            ed_idx -= 1
        return ed_idx

    small_graph_list = []
    big_graph_list = []

    total_edge_num = len(weighted_edge_list)
    small_graph_num = small_big_ratio * big_graph_num
    small_edge_theory_num = small_graph_num * small_graph_edge_num
    big_edge_theory_num = big_graph_num * big_graph_edge_num

    big_st_idx = int(small_edge_theory_num / (small_edge_theory_num + big_edge_theory_num) * total_edge_num)
    big_st_idx = get_ed_idx(big_st_idx)

    if big_st_idx <= small_graph_edge_num or total_edge_num-big_st_idx <= big_graph_edge_num:
        raise ValueError('graph is too small to satisfy the request of parameters.\n'
                         'big graph num: {}, big graph edge num: {}, small graph num: {}, small graph edge num: {}'
                         .format(big_graph_num, big_graph_edge_num, small_graph_num, small_graph_edge_num))

    small_step = min(math.floor((big_st_idx-small_graph_edge_num)/(small_graph_num-1)), small_graph_edge_num)
    small_st_idx = 0
    for i in range(small_graph_num):
        small_ed_idx = small_st_idx + small_graph_edge_num
        small_ed_idx = get_ed_idx(small_ed_idx)

        graph = split_graph(weighted_edge_list[small_st_idx:small_ed_idx])
        small_graph_list.append(graph)

        small_st_idx += small_step
        small_st_idx = get_ed_idx(small_st_idx)

    big_step = min(math.floor((total_edge_num-big_st_idx-big_graph_edge_num)/(big_graph_num-1)), big_graph_edge_num)
    for i in range(big_graph_num):
        big_ed_idx = big_st_idx + big_graph_edge_num
        big_ed_idx = get_ed_idx(big_ed_idx)

        if big_ed_idx >= total_edge_num:
            graph = split_graph(weighted_edge_list[big_st_idx:])
        else:
            graph = split_graph(weighted_edge_list[big_st_idx:big_ed_idx])

        big_graph_list.append(graph)
        big_st_idx += big_step
        big_st_idx = get_ed_idx(big_st_idx)

    print('small graph edge num: {}, small graph step: {}, overlap rate: {}\nbig graph edge num: {}, big graph step: {}, overlap rate: {}'
          .format(small_graph_edge_num, small_step, (small_graph_edge_num-small_step)/small_graph_edge_num,
                  big_graph_edge_num, big_step, (big_graph_edge_num-big_step)/big_graph_edge_num))
    graph_stat(small_graph_list)
    graph_stat(big_graph_list)

    return small_graph_list, big_graph_list


def sample_graph(weighted_edge_list, target_edge_num, over_rate=0.5):
    total_edge_num = len(weighted_edge_list)
    if target_edge_num > total_edge_num:
        target_edge_num = int(total_edge_num/2)
        print('graph is too small, sample edge num: {}'.format(target_edge_num))

    target_graph_num = math.floor((total_edge_num-target_edge_num)/((1-over_rate) * target_edge_num)) + 1

    if target_graph_num == 1:
        small_step = total_edge_num
    else:
        small_step = math.floor((total_edge_num - target_edge_num) / (target_graph_num - 1))
    # small_step = min(math.floor((total_edge_num - target_edge_num) / (target_graph_num - 1)), target_edge_num)
    # if (target_edge_num-small_step)/target_edge_num > over_rate:
    #     target_graph_num = math.floor((total_edge_num - target_edge_num) / ((1 - over_rate) * target_edge_num)) + 1

    graph_list = []
    small_st_idx = 0
    for i in range(target_graph_num):
        small_ed_idx = small_st_idx + target_edge_num
        if small_ed_idx >= total_edge_num:
            graph = split_graph(weighted_edge_list[small_st_idx:])
        else:
            graph = split_graph(weighted_edge_list[small_st_idx:small_ed_idx])
        graph_list.append(graph)

        small_st_idx += small_step

    print('graph num: {}, graph edge num: {}, graph step: {}, overlap rate: {}'
          .format(target_graph_num, target_edge_num, small_step, (target_edge_num-small_step)/target_edge_num))

    # graph_stat(graph_list)
    return graph_list


def graph_stat(graph_list):
    num_graphs = 0
    total_nodes = 0
    total_edges = 0

    for graph in graph_list:
        num_graphs += 1
        total_nodes += graph.number_of_nodes()
        total_edges += graph.number_of_edges()

    avg_nodes = total_nodes / num_graphs
    avg_edges = total_edges / num_graphs

    print('graph num: {}, avg node: {}, avg edge: {}'.format(num_graphs, avg_nodes, avg_edges))


def get_files_in_directory(directory):
    file_list = []
    for file_name in os.listdir(directory):
        file_path = osp.join(directory, file_name)
        if osp.isfile(file_path):
            file_list.append(file_path)
    return file_list


def add_random_weight(weight):
    if weight < 0.5:
        weight += random.uniform(0, 1e-5)
    else:
        weight -= random.uniform(0, 1e-5)
    return weight


def inf_check(pd):
    for j in range(pd.shape[0]):
        if pd[j, 1] == np.inf:
            pd[j, 1] = 1
    return pd


def sort_barcode(barcode):
    return sorted(barcode, key=lambda x: (x[0], -x[1]))


def dowkerSourceFiltration(graph):


    def addDowkerComplex(edge):


        source_node = edge[0]
        sink_node = edge[1]
        edge_weight = edge[2]

        if source_node in source_to_sink_dict.keys():
            source_to_sink_dict[source_node].append(sink_node)
        else:
            source_to_sink_dict[source_node] = [sink_node]


        if sink_node not in filtered_simplex_dict.keys():
            simplices.insert([sink_node], filtration=edge_weight)
            filtered_simplex_dict[sink_node] = dict()


        if len(source_to_sink_dict[source_node]) > 1:
            for i in range(len(source_to_sink_dict[source_node])-1):
                node = source_to_sink_dict[source_node][i]

                simplex_node = [node, sink_node]
                simplex_node.sort()

                if simplex_node[1] not in filtered_simplex_dict[simplex_node[0]].keys():
                    simplices.insert(simplex_node, filtration=edge_weight)
                    filtered_simplex_dict[simplex_node[0]][simplex_node[1]] = set()

        if len(source_to_sink_dict[source_node]) > 2:
            for i in range(len(source_to_sink_dict[source_node]) - 2):
                node1 = source_to_sink_dict[source_node][i]
                for j in range(i+1, len(source_to_sink_dict[source_node]) - 1):
                    node2 = source_to_sink_dict[source_node][j]

                    simplex_node = [node1, node2, sink_node]
                    simplex_node.sort()

                    if simplex_node[2] not in filtered_simplex_dict[simplex_node[0]][simplex_node[1]]:
                        simplices.insert(simplex_node, filtration=edge_weight)
                        filtered_simplex_dict[simplex_node[0]][simplex_node[1]].add(simplex_node[2])

    def add_barcode(barcode, is_birth, dim):
        if is_birth:
            bd_time_ind = 3
            bd_time = barcode[0]
        else:
            bd_time_ind = 4
            bd_time = barcode[1]

        bd_time_ind += 2*dim
        barcode_ind = 3 + 2*dim

        for i in range(edge_num):
            weight = edge_barcode[i][2]
            if weight == bd_time:
                if edge_barcode[i, bd_time_ind] != 0:
                    if dim == 0:
                        raise AttributeError('This situation shouldn''t appear')
                    else:
                        break
                edge_barcode[i, barcode_ind:barcode_ind+2] = barcode
                break

    edge_list = sorted(graph.edges(data='weight'), key=lambda x: x[2])
    edge_num = len(edge_list)
    assert edge_num > 0

    source_to_sink_dict = dict()
    filtered_simplex_dict = dict()
    simplices = gd.SimplexTree()

    for i in range(edge_num):
        addDowkerComplex(edge_list[i])


    simplices.persistence()
    barcode = []
    for i in range(2):
        barcode.append(inf_check(simplices.persistence_intervals_in_dimension(i)))

    edge_barcode = np.zeros((edge_num, 7))
    edge_barcode[:, :3] = edge_list

    for d in range(2):
        for i in range(barcode[d].shape[0]):
            add_barcode(barcode[d][i, :], True, d)
            if barcode[d][i, 1] != 1:
                add_barcode(barcode[d][i, :], False, d)

    for i in range(edge_num):
        weight = edge_barcode[i][2]
        if all(edge_barcode[i, 3:] == 0):
            edge_barcode[i, 3:5] = [weight, weight]

    return edge_barcode


def is_same_barcode(barcode1, barcode2):
    if len(barcode1) == len(barcode2):
        return (np.array(barcode1) == np.array(barcode2)).all()
    else:
        return False


def is_same_interval(interval1, interval2):
    if is_same_decimal(interval1[0], interval2[0]) and is_same_decimal(interval1[1], interval2[1]):
        return True
    else:
        return False


def is_same_decimal(a, b):
    if abs(a - b) < 1e-10:
        return True
    else:
        return False


def get_interval(barcode1, barcode2):

    def interval_judge(barcode1, barcode2):
        for interval1 in barcode1:
            same_flag = []
            for interval2 in barcode2:
                if is_same_interval(interval1, interval2):
                    same_flag.append(1)
                    break
                else:
                    same_flag.append(0)

            if not any(same_flag):
                result.append(interval1)

    result = []
    interval_judge(barcode1, barcode2)
    interval_judge(barcode2, barcode1)

    interval_num = len(result)
    if interval_num == 0 or interval_num > 2:
        raise AttributeError('This situation shouldn''t appear')
    elif interval_num == 1:
        return result[0], 0
    else:
        return result[1], result[0]


def add_WKDN_weight(graph):
    alpha = 1
    miu = 1

    copy_graph = copy.deepcopy(graph)
    shell_dict = dict()
    k = 0
    shell = nx.k_shell(copy_graph, k)
    while len(copy_graph.nodes) > 0:
        for node in shell.nodes:
            shell_dict[node] = k
        copy_graph.remove_nodes_from(list(shell.nodes))
        k += 1
        shell = nx.k_shell(copy_graph, k)

    for node1, node2 in graph.edges:
        graph[node1][node2]['weight'] = (alpha*graph.degree(node1)+miu*shell_dict[node1]) * \
                                        (alpha*graph.degree(node2)+miu*shell_dict[node2])

    return graph



def graph_reduction(graph, mode, reduction_num=1, reduction_prop=1.0):
    assert reduction_num >= 0

    if mode == 'degree':
        alternative_nodes = [n for n in graph.nodes if graph.degree(n) <= reduction_num]
    elif mode == 'shell':
        alternative_nodes = []
        for i in range(reduction_num+1):
            alternative_nodes += [n for n in nx.k_shell(graph, i).nodes]
    else:
        raise AttributeError('no mode named {}'.format(mode))

    nodes_to_remove = random.sample(alternative_nodes, int(len(alternative_nodes)*reduction_prop))
    graph.remove_nodes_from(nodes_to_remove)

    if type(graph) == nx.Graph:
        graph = graph.subgraph(max(nx.connected_components(graph), key=len))
    else:
        graph = graph.subgraph(max(nx.weakly_connected_components(graph), key=len))

    return graph


def construct_gc_bench_dataset(dataset_path, dataset_name, filt='dowker', small_graph_prop=0.8, has_node_attr=False, is_directed=True):

    adj_file = osp.join(dataset_path, dataset_name, '{}_A.txt'.format(dataset_name))
    indicator_file = osp.join(dataset_path, dataset_name, '{}_graph_indicator.txt'.format(dataset_name))
    graph_label_file = osp.join(dataset_path, dataset_name, '{}_graph_labels.txt'.format(dataset_name))
    node_attribute_file = osp.join(dataset_path, dataset_name, '{}_node_attributes.txt'.format(dataset_name))

    total_graph = nx.Graph()
    with open(adj_file, "r") as f:
        for line in f:
            row = line.strip().split(", ")
            total_graph.add_edge(int(row[0]), int(row[1]))


    total_graph = add_WKDN_weight(total_graph)

    graph_node = dict()
    with open(indicator_file, "r") as f:
        nodeID = 1
        for line in f:
            graphID = int(line)
            if graphID not in graph_node.keys():
                graph_node[graphID] = list()
            graph_node[graphID].append(nodeID)
            nodeID += 1

    graph_label = list()
    with open(graph_label_file, "r") as f:
        for line in f:
            graph_label.append(int(line))

    if has_node_attr:
        with open(node_attribute_file, "r") as f:
            nodeID = 1
            for line in f:
                row = line.strip().split(",")
                attr = np.array(list(map(float, row)))
                if nodeID in total_graph.nodes:
                    total_graph.nodes[nodeID]['attr'] = attr
                nodeID += 1

    label_set = set()

    graph_dict = dict()
    graph_node_num = dict()
    for graphID, nodes in graph_node.items():
        graph_dict[graphID] = dict()
        graph_dict[graphID]['graph'] = nx.convert_node_labels_to_integers(nx.subgraph(total_graph, nodes))

        label = graph_label[graphID-1]-1

        graph_dict[graphID]['label'] = label



        if label not in graph_node_num.keys():
            graph_node_num[label] = list()
        graph_node_num[label].append((graphID, len(nodes)))

    for label in graph_node_num.keys():
        graph_node_num[label].sort(key=lambda x: x[1])

    small_graph_idx = list()
    big_graph_idx = list()
    for label in graph_node_num.keys():
        small_graph_idx += [graphID for graphID, _ in graph_node_num[label][:int(small_graph_prop*len(graph_node_num[label]))]]
        big_graph_idx += [graphID for graphID, _ in graph_node_num[label][int(small_graph_prop*len(graph_node_num[label])):]]

    random.shuffle(small_graph_idx)
    random.shuffle(big_graph_idx)

    dict_store = dict()
    dict_store['small_graph'] = dict()
    dict_store['big_graph'] = dict()

    for i in tqdm(small_graph_idx):
        g = graph_dict[i]['graph']
        g.remove_edges_from(nx.selfloop_edges(g))

        original_edge_index = np.array([[e[0], e[1], e[2]] for e in g.edges(data='weight')])

        if len(g.nodes()) > 50:
            g = graph_reduction(g, mode='shell', reduction_num=1, reduction_prop=0.8)
        if len(g.edges()) == 0:
            continue

        if is_directed:
            graph = nx.DiGraph()
            max_weight = max(nx.get_edge_attributes(g, 'weight').values())
            for u, v, w in g.edges(data='weight'):
                graph.add_weighted_edges_from([(u, v, add_random_weight(w / max_weight))])
                graph.add_weighted_edges_from([(v, u, add_random_weight(w / max_weight))])
        else:
            graph = nx.Graph()
            max_weight = max(nx.get_edge_attributes(g, 'weight').values())
            for u, v, w in g.edges(data='weight'):
                graph.add_weighted_edges_from([(u, v, add_random_weight(w / max_weight))])

        if is_directed:
            dict_store['small_graph'][i] = get_graph_info(graph, compute_PD=True, is_directed=True)
        else:
            dict_store['small_graph'][i] = get_graph_info(graph, compute_PD=True, is_directed=False)

        if isinstance(dict_store['small_graph'][i], int):
            print('del empty graph {}'.format(i))
            del dict_store['small_graph'][i]
            continue

        dict_store['small_graph'][i]['original_edge_index'] = original_edge_index
        dict_store['small_graph'][i]['label'] = graph_dict[i]['label']

    for i in tqdm(big_graph_idx):
        g = graph_dict[i]['graph']
        g.remove_edges_from(nx.selfloop_edges(g))
        original_edge_index = np.array([[e[0], e[1], e[2]] for e in g.edges(data='weight')])
        g = graph_reduction(g, mode='shell', reduction_num=1, reduction_prop=0.8)

        if is_directed:
            graph = nx.DiGraph()
            max_weight = max(nx.get_edge_attributes(g, 'weight').values())
            for u, v, w in g.edges(data='weight'):
                graph.add_weighted_edges_from([(u, v, add_random_weight(w / max_weight))])
                graph.add_weighted_edges_from([(v, u, add_random_weight(w / max_weight))])
        else:
            graph = nx.Graph()
            max_weight = max(nx.get_edge_attributes(g, 'weight').values())
            for u, v, w in g.edges(data='weight'):
                graph.add_weighted_edges_from([(u, v, add_random_weight(w / max_weight))])

        node_num = len(graph.nodes())
        if node_num > 30000:
            print('graph is too big -- node {}'.format(node_num))
            continue

        if is_directed:
            dict_store['big_graph'][i] = get_graph_info(graph, compute_PD=True, is_directed=True)
        else:
            dict_store['big_graph'][i] = get_graph_info(graph, compute_PD=True, is_directed=False)

        dict_store['big_graph'][i]['original_edge_index'] = original_edge_index
        dict_store['big_graph'][i]['label'] = graph_dict[i]['label']

    if is_directed:
        save_name = os.path.join(dataset_path, 'GCB_{}_{}_di.pkl'.format(dataset_name, filt))
    else:
        save_name = os.path.join(dataset_path, 'GCB_{}_{}_und.pkl'.format(dataset_name, filt))
    with open(save_name, 'wb') as f:
        pickle.dump(dict_store, f, protocol=4)


def REDDIT_stat():
    reddit_path = './DNDN/data1/GCBench/GCB_REDDIT-BINARY_dowker.pkl'

    with open(reddit_path, 'rb') as f:
        dict_store = pickle.load(f)

    sg_list = []
    bg_list = []

    for s in dict_store['small_graph'].keys():
        g = nx.DiGraph()
        g.add_weighted_edges_from(dict_store['small_graph'][s]['original_edge_index'])
        sg_list.append(g)

    for s in dict_store['big_graph'].keys():
        g = nx.DiGraph()
        g.add_weighted_edges_from(dict_store['big_graph'][s]['original_edge_index'])
        bg_list.append(g)

    graph_stat(sg_list)
    graph_stat(bg_list)
    print('1')


def get_gudhi_time():
    dataset_name = 'social'  # REDDIT-BINARY REDDIT-MULTI-5K REDDIT-MULTI-12K citation bitcoin question social

    dataset_dir = './DNDN/data1/SocioPatternsDataset'
    dataset_path = osp.join(dataset_dir, 'GCB_{}_dowker.pkl'.format(dataset_name))

    with open(dataset_path, 'rb') as f:
        dict_store = pickle.load(f)

    print('testing dataset {}'.format(dataset_name))
    use_time = 0
    for s in tqdm(dict_store['big_graph'].keys()):
        g = nx.DiGraph()
        g.add_weighted_edges_from(dict_store['big_graph'][s]['edge_index'])

        t0 = time.time()
        _ = dowkerSourceFiltration(g)
        t1 = time.time()

        use_time += t1 - t0

    print('{} use time: {}s'.format(dataset_name, use_time/len(dict_store['big_graph'])))


def construct_sp_dataset(dataset_name, dataset_path):
    def get_pd_result(g):
        g = graph_reduction(g, mode='shell', reduction_num=1, reduction_prop=0.8)
        graph = nx.DiGraph()
        for u, v, w in g.edges(data='weight'):
            graph.add_weighted_edges_from([(u, v, add_random_weight(w))])
        return get_graph_info(graph, compute_PD=True, is_directed=True)
    print('processing {}...'.format(dataset_name))
    if dataset_name in ['cit-HepPh', 'cit-HepTh']:
        weighted_edge_list = read_citation(dataset_name)
    if dataset_name in ['alpha', 'otc']:
        weighted_edge_list = read_bitcoin(dataset_name)
    if dataset_name in ['askubuntu', 'mathoverflow', 'stackoverflow', 'superuser']:
        weighted_edge_list = read_question(dataset_name)
        
    small_graph_list, big_graph_list = get_split_graph_list_big_overlap(weighted_edge_list)
    # small_graph_list, big_graph_list = get_split_graph_list_big_nonoverlap(weighted_edge_list)

    dict_store = dict()
    dict_store['small_graph'] = dict()
    dict_store['big_graph'] = dict()

    for i in tqdm(range(len(small_graph_list))):
        g = small_graph_list[i]
        original_edge_index = np.array([[e[0], e[1], e[2]] for e in g.edges(data='weight')])
        dict_store['small_graph'][i] = get_pd_result(g)

        if isinstance(dict_store['small_graph'][i], int):
            print('del empty graph {}'.format(i))
            del dict_store['small_graph'][i]
            continue

        dict_store['small_graph'][i]['original_edge_index'] = original_edge_index

    for i in tqdm(range(len(big_graph_list))):
        g = big_graph_list[i]
        original_edge_index = np.array([[e[0], e[1], e[2]] for e in g.edges(data='weight')])
        dict_store['big_graph'][i] = get_pd_result(g)
        dict_store['big_graph'][i]['original_edge_index'] = original_edge_index

    save_name = os.path.join(dataset_path, 'GCB_{}_{}.pkl'.format(dataset_name, 'dowker'))
    with open(save_name, 'wb') as f:
        pickle.dump(dict_store, f, pickle.HIGHEST_PROTOCOL)

    print('{} processed'.format(dataset_name))


def construct_weibo_dataset(dataset_path, dataset_name):
    def get_pd_result(g):
        g = graph_reduction(g, mode='shell', reduction_num=1, reduction_prop=0.8)
        graph = nx.DiGraph()
        for u, v, w in g.edges(data='weight'):
            graph.add_weighted_edges_from([(u, v, add_random_weight(w))])
        return get_graph_info(graph, compute_PD=True, is_directed=True)

    edge_num_lower = 800
    print('processing {}...'.format(dataset_name))

    weighted_edge_lists = read_weibo(dataset_name, edge_num_lower)
    use_graph_num_to_small = int(0.8*len(weighted_edge_lists))

    small_graph_list = []
    big_graph_list = []

    for weighted_edge_list in weighted_edge_lists[:use_graph_num_to_small]:
        small_graph_list += sample_graph(weighted_edge_list, 500, over_rate=0)

    for weighted_edge_list in weighted_edge_lists[use_graph_num_to_small:]:
        big_graph_list += sample_graph(weighted_edge_list, 2500, over_rate=0)

    dict_store = dict()
    dict_store['small_graph'] = dict()
    dict_store['big_graph'] = dict()

    for i in tqdm(range(len(small_graph_list))):
        g = small_graph_list[i]
        original_edge_index = np.array([[e[0], e[1], e[2]] for e in g.edges(data='weight')])
        dict_store['small_graph'][i] = get_pd_result(g)

        if isinstance(dict_store['small_graph'][i], int):
            print('del empty graph {}'.format(i))
            del dict_store['small_graph'][i]
            continue

        dict_store['small_graph'][i]['original_edge_index'] = original_edge_index

    for i in tqdm(range(len(big_graph_list))):
        g = big_graph_list[i]
        original_edge_index = np.array([[e[0], e[1], e[2]] for e in g.edges(data='weight')])
        dict_store['big_graph'][i] = get_pd_result(g)

        if isinstance(dict_store['big_graph'][i], int):
            print('del empty graph {}'.format(i))
            del dict_store['big_graph'][i]
            continue

        dict_store['big_graph'][i]['original_edge_index'] = original_edge_index

    save_name = os.path.join(dataset_path, 'GCB_{}_{}.pkl'.format(dataset_name, 'dowker'))
    with open(save_name, 'wb') as f:
        pickle.dump(dict_store, f, protocol=4)

    print('{} processed'.format(dataset_name))


def add_label_to_dataset():
    def add_label(subdataset, graph_idx, label):
        for idx in subdataset['small_graph']:
            result_dataset['small_graph'][graph_idx] = subdataset['small_graph'][idx]
            result_dataset['small_graph'][graph_idx]['label'] = label
            graph_idx += 1
        for idx in subdataset['big_graph']:
            result_dataset['big_graph'][graph_idx] = subdataset['big_graph'][idx]
            result_dataset['big_graph'][graph_idx]['label'] = label
            graph_idx += 1

        return graph_idx

    subdataset1_path = osp.join(dataset_path, 'GCB_social-askubuntu_dowker.pkl')
    subdataset2_path = osp.join(dataset_path, 'GCB_social-mathoverflow_dowker.pkl')
    subdataset3_path = osp.join(dataset_path, 'GCB_question-stackoverflow_dowker.pkl')
    subdataset4_path = osp.join(dataset_path, 'GCB_question-superuser_dowker.pkl')
    result_dataset_path = osp.join(dataset_path, 'GCB_question_dowker.pkl')

    with open(subdataset1_path, 'rb') as f:
        subdataset1 = pickle.load(f)
    with open(subdataset2_path, 'rb') as f:
        subdataset2 = pickle.load(f)
    with open(subdataset3_path, 'rb') as f:
        subdataset3 = pickle.load(f)
    with open(subdataset4_path, 'rb') as f:
        subdataset4 = pickle.load(f)

    result_dataset = dict()
    result_dataset['small_graph'] = dict()
    result_dataset['big_graph'] = dict()

    graph_idx = 1
    graph_idx = add_label(subdataset1, graph_idx, 0)
    graph_idx = add_label(subdataset2, graph_idx, 1)
    graph_idx = add_label(subdataset3, graph_idx, 2)
    graph_idx = add_label(subdataset4, graph_idx, 3)

    with open(result_dataset_path, 'wb') as f:
        pickle.dump(result_dataset, f, protocol=4)


def cal_graph_node_edge(dict_store):
    node_num = 0
    edge_num = 0

    for i in dict_store['small_graph']:
        nodes = set()
        nodes.update(set(dict_store['small_graph'][i]['original_edge_index'][:, :2].flatten()))
        node_num += len(nodes)
        edge_num += len(dict_store['small_graph'][i]['filtration_val'])

    node_num /= len(dict_store['small_graph'])
    edge_num /= len(dict_store['small_graph'])
    print('small node num: {}, small edge num: {}'.format(node_num, edge_num))

    node_num = 0
    edge_num = 0

    for i in dict_store['big_graph']:
        nodes = set()
        nodes.update(set(dict_store['big_graph'][i]['original_edge_index'][:, :2].flatten()))
        node_num += len(nodes)
        edge_num += len(dict_store['big_graph'][i]['filtration_val'])

    node_num /= len(dict_store['big_graph'])
    edge_num /= len(dict_store['big_graph'])
    print('big node num: {}, big edge num: {}'.format(node_num, edge_num))


if __name__ == "__main__":
    dataset_path = './DNDN/raw_data'
    format = 'REDDIT-MULTI-12K'
    # 
    small_graph_edge_num = 1000
    big_graph_edge_num = 5000
    big_graph_num = 40
    small_big_ratio = 4

    # get_gudhi_time()
    # sys.exit()
    
    if format in ['REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K']:
        construct_gc_bench_dataset(dataset_path=dataset_path, dataset_name=format, has_node_attr=False)
    if format == 'citation':
        for name in ['cit-HepPh', 'cit-HepTh']:
            construct_sp_dataset(dataset_path=dataset_path, dataset_name=name)
            add_label_to_dataset()
    if format == 'question':
        for name in ['askubuntu', 'mathoverflow', 'stackoverflow', 'superuser']:
            construct_sp_dataset(dataset_path=dataset_path, dataset_name=name)
            add_label_to_dataset()
    if format == 'bitcoin':
        for name in ['alpha', 'otc']:
            construct_sp_dataset(dataset_path=dataset_path, dataset_name=name)
            add_label_to_dataset()
    if format in ['social']:
        construct_weibo_dataset(dataset_path=dataset_path, dataset_name=name)
        add_label_to_dataset()