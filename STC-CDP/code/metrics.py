from typing import List, Union, Set
import numpy as np


def compare_comm(pred_comm: Union[List, Set],
                 true_comm: Union[List, Set]):
    
    intersect = set(true_comm) & set(pred_comm)
    p = len(intersect) / len(pred_comm)
    r = len(intersect) / len(true_comm)
    f = 2 * p * r / (p + r + 1e-9)
    j = len(intersect) / (len(pred_comm) + len(true_comm) - len(intersect))
    return p, r, f, j


def eval_scores(pred_comms, true_comms, tmp_print=False):
    pred_scores = np.zeros((len(pred_comms), 4))
    truth_scores = np.zeros((len(true_comms), 4))

    for i, pred_comm in enumerate(pred_comms):
        np.max([compare_comm(pred_comm, true_comms[j])
                for j in range(len(true_comms))], 0, out=pred_scores[i])

    for j, true_comm in enumerate(true_comms):
        np.max([compare_comm(pred_comms[i], true_comm)
                for i in range(len(pred_comms))], 0, out=truth_scores[j])
    truth_scores[:, :2] = truth_scores[:, [1, 0]]

    if tmp_print:
        print("P, R, F, J AvgAxis0: ", pred_scores.mean(0))
        print("P, R, F, J AvgAxis1: ", truth_scores.mean(0))

    mean_score_all = (pred_scores.mean(0) + truth_scores.mean(0)) / 2.

    comm_nodes = {node for com in true_comms for node in com}
    pred_nodes = {node for com in pred_comms for node in com}
    percent = len(list(comm_nodes & pred_nodes)) / len(comm_nodes)

    if tmp_print:
        print(f"AvgF1: {mean_score_all[2]:.4f} AvgJaccard: {mean_score_all[3]:.4f}  "
              f"Detect percent: {percent:.4f}")

    return round(mean_score_all[2], 4), round(mean_score_all[3], 4)
