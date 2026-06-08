import numpy as np
from pycycle.maps.map_data import MapData


EngineHPTMap = MapData()

# Engine HPT map — generated from forward_design() + meanline off-design solver
# Tt_ref=1316.7 K  Pt_ref=1367885 Pa  N_design=8070 rpm
# NpMap in pyCycle units: N/sqrt(Tt_degR)  Tref=1 degR

EngineHPTMap.defaults = {}
EngineHPTMap.defaults['alphaMap'] = 1.0
EngineHPTMap.defaults['NpMap']    = 165.7675
EngineHPTMap.defaults['PRmap']    = 3.965

EngineHPTMap.alphaMap = np.array([0.9, 1.0, 1.1])  # NGV throat area fractions
EngineHPTMap.NpMap    = np.array([np.float64(165.7675)])
EngineHPTMap.PRmap    = np.array([np.float64(1.2185), np.float64(1.4697), np.float64(1.7209), np.float64(1.9721), np.float64(2.2233), np.float64(2.4745), np.float64(2.7257), np.float64(2.9769), np.float64(3.228), np.float64(3.4792), np.float64(3.7304), np.float64(3.9816), np.float64(4.2328), np.float64(4.484), np.float64(4.7352), np.float64(4.9863), np.float64(5.2375), np.float64(5.4887), np.float64(5.7399), np.float64(5.9911)])

EngineHPTMap.effMap = np.array([
    [  # alpha = area_frac = 0.90
        [0.8374, 0.8852, 0.8967, 0.9008, 0.9010, 0.8984, 0.8955, 0.8936, 0.8925, 0.8921, 0.8924, 0.8921, 0.8905, 0.8869, 0.8807, 0.8718, 0.8595, 0.8436, 0.8224, 0.7903]
    ],
    [  # alpha = area_frac = 1.00
        [0.8374, 0.8852, 0.8967, 0.9008, 0.9010, 0.8984, 0.8955, 0.8936, 0.8925, 0.8921, 0.8924, 0.8921, 0.8905, 0.8869, 0.8807, 0.8718, 0.8595, 0.8436, 0.8224, 0.7903]
    ],
    [  # alpha = area_frac = 1.10
        [0.8374, 0.8852, 0.8967, 0.9008, 0.9010, 0.8984, 0.8955, 0.8936, 0.8925, 0.8921, 0.8924, 0.8921, 0.8905, 0.8869, 0.8807, 0.8718, 0.8595, 0.8436, 0.8224, 0.7903]
    ]
])

EngineHPTMap.WpMap = np.array([
    [  # alpha = area_frac = 0.90
        [34.6800, 36.2460, 36.7229, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380]
    ],
    [  # alpha = area_frac = 1.00
        [34.6800, 36.2460, 36.7229, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380]
    ],
    [  # alpha = area_frac = 1.10
        [34.6800, 36.2460, 36.7229, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380, 36.7380]
    ]
])

EngineHPTMap.Npts = EngineHPTMap.NpMap.size

EngineHPTMap.units = {}
EngineHPTMap.units['NpMap'] = 'rpm'
EngineHPTMap.units['WpMap'] = 'lbm/s'

EngineHPTMap.param_data = []
EngineHPTMap.output_data = []

EngineHPTMap.param_data.append({'name': 'alphaMap', 'values': EngineHPTMap.alphaMap,
                                'default': 1.0, 'units': None})
EngineHPTMap.param_data.append({'name': 'NpMap', 'values': EngineHPTMap.NpMap,
                                'default': 165.7675, 'units': 'rpm'})
EngineHPTMap.param_data.append({'name': 'PRmap', 'values': EngineHPTMap.PRmap,
                                'default': 3.965, 'units': None})

EngineHPTMap.output_data.append({'name': 'WpMap', 'values': EngineHPTMap.WpMap,
                                 'default': np.mean(EngineHPTMap.WpMap), 'units': 'lbm/s'})
EngineHPTMap.output_data.append({'name': 'effMap', 'values': EngineHPTMap.effMap,
                                 'default': np.mean(EngineHPTMap.effMap), 'units': None})
