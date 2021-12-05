"""
File containing the implementation of Forman Curvature for Graph Neural Networks.
Implementation based on the "Forman curvature for complex networks" paper.
"""
import numpy as np


class Vertex:
    """
    Class representing graph vertex.
    """

    def __init__(self, weight):
        """
        Initialize a vertex; keep track of edges connected to the vertex upon edge initialization.
        :param weight: weight of the vertex.
        """
        self.edges = set()
        self.weight = weight
        self.fc = None

    def deg(self):
        """
        Compute the degree of the vertex
        :return: Integer vertex degree.
        """
        return len(self.edges)

    def compute_fc(self):
        """
        Compute Forman Curvature for the vertex.
        """
        self.fc = 1 / self.deg() * sum([edge.fc for edge in self.edges])


class Edge:
    """
    Class representing graph edge.
    """

    def __init__(self, start, end, weight):
        """
        Initialize an edge.
        :param start: Pointer to the start vertex.
        :param end: Pointer to the end vertex.
        :param weight: Weight of the edge.
        """
        self.start = start
        start.edges.add(self)
        self.end = end
        end.edges.add(self)
        self.weight = weight
        self.fc = None

    def compute_fc(self):
        """
        Compute Forman Curvature for the edge.
        """
        self.fc = self.weight * (self.start.weight / self.weight + self.end.weight / self.weight
                                 - sum([self.start.weight / np.sqrt(self.weight * start_edges.weight)
                                        for start_edges in self.start.edges])
                                 - sum([self.start.weight / np.sqrt(self.weight * end_edges.weight)
                                        for end_edges in self.end.edges]))
