import numpy as np
from pycycle.maps.map_data import MapData


EngineHPTMap = MapData()

# Engine HPT map — generated from forward_design() + meanline off-design solver
# Tt_ref=1316.7 K  Pt_ref=1367885 Pa  N_design=8070 rpm
# NpMap in pyCycle units: N/sqrt(Tt_degR)  Tref=1 degR

EngineHPTMap.defaults = {}
EngineHPTMap.defaults['alphaMap'] = 1.0
EngineHPTMap.defaults['NpMap']    = 165.7675
EngineHPTMap.defaults['PRmap']    = 4.111

EngineHPTMap.alphaMap = np.array([1.0])  # NGV throat area fractions
EngineHPTMap.NpMap    = np.array([np.float64(165.7675)])
EngineHPTMap.PRmap    = np.array([np.float64(1.2095), np.float64(1.4739), np.float64(1.7384), np.float64(2.0029), np.float64(2.2673), np.float64(2.5318), np.float64(2.7963), np.float64(3.0607), np.float64(3.3252), np.float64(3.5896), np.float64(3.8541), np.float64(4.1186), np.float64(4.383), np.float64(4.6475), np.float64(4.912), np.float64(5.1764), np.float64(5.4409), np.float64(5.7054), np.float64(5.9698), np.float64(6.2343)])

EngineHPTMap.effMap = np.array([
    [  # alpha = area_frac = 1.00
        [0.7896, 0.8546, 0.8689, 0.8737, 0.8739, 0.8713, 0.8688, 0.8677, 0.8677, 0.8684, 0.8699, 0.8714, 0.8715, 0.8697, 0.8656, 0.8586, 0.8486, 0.8352, 0.8173, 0.7917]
    ]
])

EngineHPTMap.WpMap = np.array([
    [  # alpha = area_frac = 1.00
        [34.2232, 35.7508, 36.0879, 36.0848, 36.0848, 36.0848, 36.0848, 36.0848, 36.0848, 36.0848, 36.0848, 36.0848, 36.0848, 36.0848, 36.0848, 36.0848, 36.0848, 36.0848, 36.0848, 36.0848]
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
                                'default': 4.111, 'units': None})

EngineHPTMap.output_data.append({'name': 'WpMap', 'values': EngineHPTMap.WpMap,
                                 'default': np.mean(EngineHPTMap.WpMap), 'units': 'lbm/s'})
EngineHPTMap.output_data.append({'name': 'effMap', 'values': EngineHPTMap.effMap,
                                 'default': np.mean(EngineHPTMap.effMap), 'units': None})
