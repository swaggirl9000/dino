# pyg-lib 0.4.0 is installed; torch_geometric now requires >=0.6.0 for radius_graph.
# Patch before any library imports so DIG/PONITA get the torch_cluster version.
try:
    from torch_cluster import radius_graph as _rg
    import torch_geometric.nn as _pyg_nn
    _pyg_nn.radius_graph = _rg
except Exception:
    pass

from .ect_module import ECTPlugIn
