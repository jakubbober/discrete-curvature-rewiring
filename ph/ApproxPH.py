import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
import math
import ot
import matplotlib
from scipy.optimize import curve_fit
from gudhi.wasserstein.barycenter import lagrangian_barycenter as bary
from gtda.plotting import plot_diagram


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
	data = load_data('../data', 'Cora')
	G = to_networkx(data, node_attrs=['x'])
	G, C = compute_curvature(G, 'formanCurvature')
	subs = get_subsample(C, 200, 10)
	# print(subs)
	# print(subs)
	diags = []
	for A in tqdm(subs):
		# print(A, len(A))
		# pers, diag = get_PD(A, -1)
		points = compute_ph.compute_ph_giotto(A)
		diags.append(points[0, :, [0, 1]])
	print(diags)
	wmean, _ = bary(diags, verbose=True)
	print(1)
	fig = plot_diagram(wmean)
	print(2)
	fig.show()
