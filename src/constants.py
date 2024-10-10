# Inverse scaling functions
inverse_dfunc = {
    'r': lambda x: x,
    'g': lambda x: x,
    'b': lambda x: x,
    'rgb': lambda x: x,
    'ndvi': lambda x: 2 * (x - 0.5),
    'gndvi': lambda x: 2 * (x - 0.5),
    'ndwi': lambda x: 2 * (x - 0.5),
    'ndmi': lambda x: 2 * (x - 0.5),
    'nbr': lambda x: 2 * (x - 0.5),
    'ndre': lambda x: 2 * (x - 0.5),
    'evi': lambda x: 2 * (x - 0.5) / 2.5,  # For EVI, apply the inverse of the transformation
    'crswir': lambda x: x * 5,  # Scale back crswir
}