from graph_data.citation_graph_data import citation_graph_reconstruction, citation_train_valid_test
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from graph_data.citation_graph_data import citation_graph_reconstruction


def prob_distribution_calculator(data: ndarray, bins):
    count, bins_count = np.histogram(data, bins=bins)
    # finding the PDF of the histogram using count values
    pdf = count / sum(count)
    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf = np.cumsum(pdf)
    return bins_count, pdf, cdf


def citation_graph_data_analysis(dataset: str):
    graph, node_features, n_entities, n_classes, n_feats = citation_graph_reconstruction(dataset=dataset)
    in_degrees = graph.in_degrees().numpy()
    train_num, train_node_idx = citation_train_valid_test(graph=graph, data_type='train')
    valid_num, valid_node_idx = citation_train_valid_test(graph=graph, data_type='valid')
    test_num, test_node_idx = citation_train_valid_test(graph=graph, data_type='test')
    train_in_degrees = in_degrees[train_node_idx]
    valid_in_degrees = in_degrees[valid_node_idx]
    test_in_degrees = in_degrees[test_node_idx]
    train_in_degrees.sort()
    valid_in_degrees.sort()
    test_in_degrees.sort()

    # print(train_in_degrees)
    # print(valid_in_degrees)
    # print(test_in_degrees)

    # train_max = train_in_degrees.max()
    # valid_sum = np.sum(test_in_degrees >= train_max)
    # print(valid_sum)

    # train_bins = np.arange(start=1, stop=100, step=10)
    # print(train_bins)
    #
    # _, train_pdf, _ = prob_distribution_calculator(graph_data=train_in_degrees, bins=train_bins)
    # _, valid_pdf, _ = prob_distribution_calculator(graph_data=valid_in_degrees, bins=train_bins)
    # _, test_pdf, _ = prob_distribution_calculator(graph_data=test_in_degrees, bins=train_bins)
    #
    # print(train_pdf)
    # print(valid_pdf)
    # print(test_pdf)


    # plt.legend()
    # plt.show()
    return graph

if __name__ == '__main__':
    # graph = citation_graph_data_analysis(dataset='cora')
    citation_graph_reconstruction(dataset='cora')