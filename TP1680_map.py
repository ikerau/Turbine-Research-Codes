import numpy as np
from pycycle.maps.map_data import MapData


TP1680Map = MapData()

# TP-1680 (Moffitt 1980) turbine map — area_frac=1.00
# Generated from meanline off-design solver
# Tt_ref=378.0 K  Pt_ref=241325 Pa

TP1680Map.defaults = {}
TP1680Map.defaults['alphaMap'] = 1.0
TP1680Map.defaults['NpMap']    = 8081.0
TP1680Map.defaults['PRmap']    = 4.078

TP1680Map.alphaMap = np.array([0.0, 1.0, 2.0])  # replicated — no cooling model yet
TP1680Map.NpMap    = np.array([np.float64(8081.0)])
TP1680Map.PRmap    = np.array([np.float64(1.3808), np.float64(1.5228), np.float64(1.6648), np.float64(1.8067), np.float64(1.9487), np.float64(2.0906), np.float64(2.2326), np.float64(2.3745), np.float64(2.5165), np.float64(2.6584), np.float64(2.8004), np.float64(2.9423), np.float64(3.0843), np.float64(3.2262), np.float64(3.3682), np.float64(3.5101), np.float64(3.6521), np.float64(3.7941), np.float64(3.936), np.float64(4.078)])

TP1680Map.effMap = np.array([
    [  # alpha = 0.0
        [0.8726, 0.8939, 0.9021, 0.9054, 0.9069, 0.9074, 0.9072, 0.9066, 0.9057, 0.9046, 0.9033, 0.9018, 0.8999, 0.8973, 0.8927, 0.8853, 0.8742, 0.8590, 0.8385, 0.8096]
    ],
    [  # alpha = 1.0
        [0.8726, 0.8939, 0.9021, 0.9054, 0.9069, 0.9074, 0.9072, 0.9066, 0.9057, 0.9046, 0.9033, 0.9018, 0.8999, 0.8973, 0.8927, 0.8853, 0.8742, 0.8590, 0.8385, 0.8096]
    ],
    [  # alpha = 2.0
        [0.8726, 0.8939, 0.9021, 0.9054, 0.9069, 0.9074, 0.9072, 0.9066, 0.9057, 0.9046, 0.9033, 0.9018, 0.8999, 0.8973, 0.8927, 0.8853, 0.8742, 0.8590, 0.8385, 0.8096]
    ]
])

TP1680Map.WpMap = np.array([
    [  # alpha = 0.0
        [7.4530, 7.7836, 7.9771, 8.0846, 8.1365, 8.1513, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508]
    ],
    [  # alpha = 1.0
        [7.4530, 7.7836, 7.9771, 8.0846, 8.1365, 8.1513, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508]
    ],
    [  # alpha = 2.0
        [7.4530, 7.7836, 7.9771, 8.0846, 8.1365, 8.1513, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508, 8.1508]
    ]
])

TP1680Map.Npts = TP1680Map.NpMap.size

TP1680Map.units = {}
TP1680Map.units['NpMap'] = 'rpm'
TP1680Map.units['WpMap'] = 'lbm/s'

TP1680Map.param_data = []
TP1680Map.output_data = []

TP1680Map.param_data.append({'name': 'alphaMap', 'values': TP1680Map.alphaMap,
                              'default': 1.0, 'units': None})
TP1680Map.param_data.append({'name': 'NpMap', 'values': TP1680Map.NpMap,
                              'default': 8081.0, 'units': 'rpm'})
TP1680Map.param_data.append({'name': 'PRmap', 'values': TP1680Map.PRmap,
                              'default': 4.078, 'units': None})

TP1680Map.output_data.append({'name': 'WpMap', 'values': TP1680Map.WpMap,
                               'default': np.mean(TP1680Map.WpMap), 'units': 'lbm/s'})
TP1680Map.output_data.append({'name': 'effMap', 'values': TP1680Map.effMap,
                               'default': np.mean(TP1680Map.effMap), 'units': None})
