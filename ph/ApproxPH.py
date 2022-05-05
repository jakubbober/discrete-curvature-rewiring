import pickle

import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
import math
import ot
import matplotlib
from scipy.optimize import curve_fit
from gudhi.wasserstein.barycenter import lagrangian_barycenter as bary
from gtda.plotting import plot_diagram
import gudhi


# setting parameters
from torch_geometric.utils import to_networkx
from tqdm import tqdm

from compute_curvature import compute_curvature
from ph import compute_ph
from utils import load_data


matplotlib.use('WebAgg')


def get_subsample(large_set, nb_sub_size, nb_sub):
	"""
	sample multiple subsets from original point set
	each subset contains nb_sub_size points
	"""
	row_total = large_set.shape[0]
	subsample_set = []
	for i in range(nb_sub):
		row_sample = np.random.choice(row_total, nb_sub_size, replace = False, p = None)
		# print(row_sample)
		subsample_set.append(large_set[row_sample, :][:, row_sample])
		# print(np.count_nonzero(large_set[row_sample, :][:, row_sample]))
	return subsample_set


def get_PD(C, min_persistence):
	"""
	compute the persistence diagram for VR filtration of point clouds
	"""
	# noinspection PyUnresolvedReferences
	rips = gd.RipsComplex(distance_matrix=C)
	rips_st = rips.create_simplex_tree(max_dimension=2)
	pers = rips_st.persistence(min_persistence=min_persistence)
	diag = rips_st.persistence_intervals_in_dimension(1)
	return pers, diag


if __name__ == '__main__':
	# # a = np.array([[1,2,3,4],
	# # 			  [5,6,7,8],
	# # 			  [9,10,11,12],
	# # 			  [13,14,15,16]])
	# # print(get_subsample(a, 3, 5))
	# # data = load_data('../data', 'Cora')
	# with open('../jctops/rewired', 'rb') as f:
	# 	data = pickle.load(f)
	# G = to_networkx(data, node_attrs=['x']).to_undirected()
	# # print(len(G.edges))
	# G, C = compute_curvature(G, 'formanCurvature')
	# # subs = get_subsample(C, 1000, 3)
	# subs = [C[1000 * i:1000 * (i + 1), 1000 * i:1000 * (i + 1)] for i in range(6)]
	# # print(subs)
	# # print(subs)
	# # diags = [compute_ph.compute_ph_giotto(C[1000:2000, 1000:2000])[0, :, [0, 1]].T]
	# diags = []
	# print(diags)
	# for A in tqdm(subs):
	# 	# print(A, len(A))
	# 	# pers, diag = get_PD(A, -1)
	# 	points = compute_ph.compute_ph_giotto(A)
	# 	if len(points[0]) > 1:
	# 		diags.append(np.array(points[0, :, [0, 1]].T))
	# 	# diags += points[0, :, [0, 1]]
	# 	# print(points)
	# 	print(points[0, :, [0, 1]].T)
	# print(diags)
	# wmean, _ = bary(diags, verbose=True)
	# # print(1)
	# # print(wmean)
	with open('forman_Cora_approx_ph', 'rb') as f:
		wmean1 = pickle.load(f)
	with open('forman_Cora_approx_ph2', 'rb') as f:
		wmean2 = pickle.load(f)
	# print(wmean[:, :2])
	#
	# wmean = np.hstack((np.array(wmean), np.ones([len(wmean), 1])))
	#
	# print(wmean)
	# fig = plot_diagram(wmean)
	# fig.show()
	# with open('forman_Cora_approx_ph2', 'wb') as f:
	# 	pickle.dump(wmean[:, :2], f)
	# fig = plot_diagram(wmean)
	# print(2)
	# fig.show()
	print(gudhi.bottleneck_distance(wmean1, wmean2))
