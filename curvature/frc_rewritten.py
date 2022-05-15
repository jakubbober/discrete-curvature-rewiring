import networkx as nx
import math


def augmented(G, v1, v2):
    v1_nbr = set(G.neighbors(v1))
    # v1_nbr.remove(v2)
    v2_nbr = set(G.neighbors(v2))
    # v2_nbr.remove(v1)

    # face = v1_nbr & v2_nbr
    face = v1_nbr.intersection(v2_nbr)
    # print(len(face))
    # print(4 - G.degree[v1] - G.degree[v2] + 3 * len(face))
    return 4 - G.degree[v1] - G.degree[v2] + 3 * len(face)


class FormanRicci:
    def __init__(self, G: nx.Graph, weight="weight", method="augmented", verbose="ERROR"):
        """A class to compute Forman-Ricci curvature for all nodes and edges in G.
        Parameters
        ----------
        G : NetworkX graph
            A given NetworkX graph, unweighted graph only for now, edge weight will be ignored.
        weight : str
            The edge weight used to compute Ricci curvature. (Default value = "weight")
        method : {"1d", "augmented"}
            The method used to compute Forman-Ricci curvature. (Default value = "augmented")
            - "1d": Computed with 1-dimensional simplicial complex (vertex, edge).
            - "augmented": Computed with 2-dimensional simplicial complex, length <=3 (vertex, edge, face).
        verbose: {"INFO","DEBUG","ERROR"}
            Verbose level. (Default value = "ERROR")
                - "INFO": show only iteration process log.
                - "DEBUG": show all output logs.
                - "ERROR": only show log if error happened.
        """

        self.G = G.copy()
        self.weight = weight
        self.method = method

        if not nx.get_edge_attributes(self.G, self.weight):
            for (v1, v2) in self.G.edges():
                self.G[v1][v2][self.weight] = 1.0
        if not nx.get_node_attributes(self.G, self.weight):
            for v in self.G.nodes():
                self.G.nodes[v][self.weight] = 1.0
        if self.G.is_directed():
            self.G = self.G.to_undirected()

    def compute_ricci_curvature_edge(self, v1, v2):
        if self.method == '1d':
            v1_nbr = set(self.G.neighbors(v1))
            v1_nbr.remove(v2)
            v2_nbr = set(self.G.neighbors(v2))
            v2_nbr.remove(v1)

            w_e = 1
            w_v1 = 1
            w_v2 = 1
            ev1_sum = sum([w_v1 / math.sqrt(w_e * 1) for v in v1_nbr])
            ev2_sum = sum([w_v2 / math.sqrt(w_e * 1) for v in v2_nbr])

            self.G[v1][v2]["formanCurvature"] = w_e * (w_v1 / w_e + w_v2 / w_e - (ev1_sum + ev2_sum))
        elif self.method == 'augmented':
            v1_nbr = set(self.G.neighbors(v1))
            # v1_nbr.remove(v2)
            v2_nbr = set(self.G.neighbors(v2))
            # v2_nbr.remove(v1)

            # face = v1_nbr & v2_nbr
            face = v1_nbr.intersection(v2_nbr)
            # w_e = 1
            # w_f = 1  # Assume all face have weight 1
            # w_v1 = 1
            # w_v2 = 1
            #
            # sum_ef = sum([w_e / w_f for _ in face])
            # sum_ve = sum([w_v1 / w_e + w_v2 / w_e])
            #
            # sum_ehef = 0  # Always 0 for cycle = 3 case.
            # sum_veeh = sum([w_v1 / math.sqrt(w_e * 1) for v in (v1_nbr - face)] +
            #                [w_v2 / math.sqrt(w_e * 1) for v in (v2_nbr - face)])
            #
            # self.G[v1][v2]["formanCurvature"] = w_e * (sum_ef + sum_ve - math.fabs(sum_ehef - sum_veeh))
            self.G[v1][v2]["formanCurvature"] = 4 - self.G.degree[v1] - self.G.degree[v2] + 4 * len(face)



    def compute_ricci_curvature(self):
        """Compute Forman-ricci curvature for all nodes and edges in G.
        Node curvature is defined as the average of all it's adjacency edge.
        Returns
        -------
        G: NetworkX graph
            A NetworkX graph with "formanCurvature" on nodes and edges.
        Examples
        --------
        To compute the Forman-Ricci curvature for karate club graph:
            >>> G = nx.karate_club_graph()
            >>> frc = FormanRicci(G)
            >>> frc.compute_ricci_curvature()
            >>> frc.G[0][2]
            {'weight': 1.0, 'formanCurvature': -7.0}
        """

        if self.method == "1d" or self.method == "augmented":
            # Edge Forman curvature
            for (v1, v2) in self.G.edges():
                self.compute_ricci_curvature_edge(v1, v2)

        else:
            assert True, 'Method %s not available. Support methods: {"1d","augmented"}' % self.method
