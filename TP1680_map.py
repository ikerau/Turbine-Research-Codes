import numpy as np
from pycycle.maps.map_data import MapData


TP1680Map = MapData()

# TP-1680 (Moffitt 1980) turbine map — area_frac=1.00
# Generated from meanline off-design solver
# Tt_ref=378.0 K  Pt_ref=241325 Pa

TP1680Map.defaults = {}
TP1680Map.defaults['alphaMap'] = 1.0
TP1680Map.defaults['NpMap']    = 8081.0
TP1680Map.defaults['PRmap']    = 4.083

TP1680Map.alphaMap = np.array([0.0, 1.0, 2.0])  # replicated — no cooling model yet
TP1680Map.NpMap    = np.array([np.float64(8081.0)])
TP1680Map.PRmap    = np.array([np.float64(1.3808), np.float64(1.5231), np.float64(1.6653), np.float64(1.8075), np.float64(1.9497), np.float64(2.0919), np.float64(2.2342), np.float64(2.3764), np.float64(2.5186), np.float64(2.6608), np.float64(2.803), np.float64(2.9452), np.float64(3.0875), np.float64(3.2297), np.float64(3.3719), np.float64(3.5141), np.float64(3.6563), np.float64(3.7986), np.float64(3.9408), np.float64(4.083)])

TP1680Map.effMap = np.array([
    [  # alpha = 0.0
        [0.8943, 0.9014, 0.9046, 0.9062, 0.9070, 0.9073, 0.9071, 0.9065, 0.9056, 0.9045, 0.9031, 0.9017, 0.8999, 0.8973, 0.8928, 0.8854, 0.8743, 0.8590, 0.8385, 0.8096]
    ],
    [  # alpha = 1.0
        [0.8943, 0.9014, 0.9046, 0.9062, 0.9070, 0.9073, 0.9071, 0.9065, 0.9056, 0.9045, 0.9031, 0.9017, 0.8999, 0.8973, 0.8928, 0.8854, 0.8743, 0.8590, 0.8385, 0.8096]
    ],
    [  # alpha = 2.0
        [0.8943, 0.9014, 0.9046, 0.9062, 0.9070, 0.9073, 0.9071, 0.9065, 0.9056, 0.9045, 0.9031, 0.9017, 0.8999, 0.8973, 0.8928, 0.8854, 0.8743, 0.8590, 0.8385, 0.8096]
    ]
])

TP1680Map.WpMap = np.array([
    [  # alpha = 0.0
        [7.4851, 7.7867, 7.9701, 8.0747, 8.1258, 8.1404, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401]
    ],
    [  # alpha = 1.0
        [7.4851, 7.7867, 7.9701, 8.0747, 8.1258, 8.1404, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401]
    ],
    [  # alpha = 2.0
        [7.4851, 7.7867, 7.9701, 8.0747, 8.1258, 8.1404, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401, 8.1401]
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
                              'default': 4.083, 'units': None})

TP1680Map.output_data.append({'name': 'WpMap', 'values': TP1680Map.WpMap,
                               'default': np.mean(TP1680Map.WpMap), 'units': 'lbm/s'})
TP1680Map.output_data.append({'name': 'effMap', 'values': TP1680Map.effMap,
                               'default': np.mean(TP1680Map.effMap), 'units': None})
