def compute_ricci_curvature_edge(G, v1, v2, curv_type):
    if curv_type == '1d':
        return 4 - G.degree[v1] - G.degree[v2]
    elif curv_type == 'augmented':
        v1_nbr = set(G.neighbors(v1))
        v2_nbr = set(G.neighbors(v2))
        triangles = v1_nbr.intersection(v2_nbr)
        return 4 - G.degree[v1] - G.degree[v2] + 3 * len(triangles)
    elif curv_type == 'haantjes':
        v1_nbr = set(G.neighbors(v1))
        v2_nbr = set(G.neighbors(v2))
        triangles = v1_nbr.intersection(v2_nbr)
        return len(triangles)



def compute_ricci_curvature(G, curv_type):
    if curv_type == "1d" or curv_type == "augmented" or curv_type == 'haantjes':
        # Edge Forman curvature
        for (v1, v2) in G.edges():
            G[v1][v2][curv_type] = compute_ricci_curvature_edge(G, v1, v2, curv_type)
    else:
        assert True, 'Method %s not available.' % curv_type
