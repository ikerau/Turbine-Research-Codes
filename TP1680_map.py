import numpy as np
from pycycle.maps.map_data import MapData


TP1680Map = MapData()

# TP-1680 (Moffitt 1980) turbine map — area_frac=1.00
# Generated from meanline off-design solver
# Tt_ref=378.0 K  Pt_ref=241325 Pa

TP1680Map.defaults = {}
TP1680Map.defaults['alphaMap'] = 1.0
TP1680Map.defaults['NpMap']    = 8081.0
TP1680Map.defaults['PRmap']    = 4.035

TP1680Map.alphaMap = np.array([0.0, 1.0, 2.0])  # replicated — no cooling model yet
TP1680Map.NpMap    = np.array([np.float64(8081.0)])
TP1680Map.PRmap    = np.array([np.float64(1.3806), np.float64(1.5203), np.float64(1.66), np.float64(1.7997), np.float64(1.9394), np.float64(2.0791), np.float64(2.2188), np.float64(2.3585), np.float64(2.4983), np.float64(2.638), np.float64(2.7777), np.float64(2.9174), np.float64(3.0571), np.float64(3.1968), np.float64(3.3365), np.float64(3.4762), np.float64(3.6159), np.float64(3.7556), np.float64(3.8954), np.float64(4.0351)])

TP1680Map.effMap = np.array([
    [  # alpha = 0.0
        [0.8761, 0.8977, 0.9066, 0.9104, 0.9123, 0.9131, 0.9131, 0.9126, 0.9118, 0.9108, 0.9096, 0.9083, 0.9066, 0.9044, 0.9003, 0.8933, 0.8825, 0.8671, 0.8459, 0.8143]
    ],
    [  # alpha = 1.0
        [0.8761, 0.8977, 0.9066, 0.9104, 0.9123, 0.9131, 0.9131, 0.9126, 0.9118, 0.9108, 0.9096, 0.9083, 0.9066, 0.9044, 0.9003, 0.8933, 0.8825, 0.8671, 0.8459, 0.8143]
    ],
    [  # alpha = 2.0
        [0.8761, 0.8977, 0.9066, 0.9104, 0.9123, 0.9131, 0.9131, 0.9126, 0.9118, 0.9108, 0.9096, 0.9083, 0.9066, 0.9044, 0.9003, 0.8933, 0.8825, 0.8671, 0.8459, 0.8143]
    ]
])

TP1680Map.WpMap = np.array([
    [  # alpha = 0.0
        [7.5018, 7.8295, 8.0173, 8.1187, 8.1656, 8.1761, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757]
    ],
    [  # alpha = 1.0
        [7.5018, 7.8295, 8.0173, 8.1187, 8.1656, 8.1761, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757]
    ],
    [  # alpha = 2.0
        [7.5018, 7.8295, 8.0173, 8.1187, 8.1656, 8.1761, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757, 8.1757]
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
                              'default': 4.035, 'units': None})

TP1680Map.output_data.append({'name': 'WpMap', 'values': TP1680Map.WpMap,
                               'default': np.mean(TP1680Map.WpMap), 'units': 'lbm/s'})
TP1680Map.output_data.append({'name': 'effMap', 'values': TP1680Map.effMap,
                               'default': np.mean(TP1680Map.effMap), 'units': None})
