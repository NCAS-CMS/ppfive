`pyfive` structure
==================

This page summarizes how the `umfive` classes represent the `pyfive`
structure.

The examples use the TODO dataset (download 28 KB). Example datasets
in other formats can be found here.

Dataset
-------

A PP or fields file dataset is mapped to a `umfive.File` object
corresponds to `pyfive.File`


Variables
---------

The dataset contains data variable objetcs (`umfive.DataVariable`) and
and metadata variable objects (`umfive.DimensionScale` and
`umfive.Variable`), each of which corresponds to `pyfive.Dataset`.

Variable name
^^^^^^^^^^^^^

The name of a data or metdata variable instance is accessed with the
`!name` attribute.

.. code-block:: python
   :caption: Example

   >>> import umfive
   >>> u = umfive.File(FILE)  # Open the dataset
   >>> var = u['UM_m01s15i201_vn405']
   >>> var.name
   'UM_m01s15i201_vn405'
   
.. _Variable-data-and-indexing:

Variable data and indexing
^^^^^^^^^^^^^^^^^^^^^^^^^^

The data array of a data or metdata variable instance is accessed by
direct indexing, TODO

The requested subspace is always returned as a `numpy` array.

.. code-block:: python
   :caption: Example

   >>> import umfive
   >>> u = umfive.File(FILE)  # Open the dataset
   >>> var = u['air_pressure']  # Select a metadata variable
   >>> var
   <umfive.DimensionScale: air_pressure, shape=(5,)>
   >>> var[...]  # Get the entire data array
   array([850.00006 , 700.00006 , 500.00003 , 250.00002 ,  50.000004],
         dtype=float32)
   >>> var = u['UM_m01s15i201_vn405']  # Select a data variable
   >>> var
   <umfive.DataVariable: UM_m01s15i201_vn405, shape=(3, 5, 110, 106), dimensions=(time, air_pressure, grid_latitude, grid_longitude)>
   >>> var[::-1, :, 0, [1, 4, 5]]  # Get a subspace of the data array
   array([[[ 1.975026  ,  3.5579882 ,  4.1247582 ],
           [ 8.940813  ,  9.728019  ,  9.491571  ],
           [22.701277  , 26.253435  , 26.559647  ],
           [29.02668   , 33.359756  , 34.43601   ],
           [-0.82124394, -1.0969521 , -1.2123854 ]],
   
          [[ 3.3617647 ,  5.4127526 ,  6.1013527 ],
           [11.391631  , 12.237262  , 12.1365795 ],
           [25.576569  , 26.69227   , 26.317883  ],
           [40.734505  , 43.26319   , 43.388798  ],
           [ 1.1319677 ,  1.1232048 ,  1.0162908 ]],
   
          [[ 0.3053741 ,  2.4959655 ,  3.3548405 ],
           [ 9.405224  , 10.491347  , 10.658589  ],
           [24.7883    , 24.599092  , 24.115479  ],
           [50.26751   , 50.851723  , 50.074444  ],
           [ 1.9971831 ,  2.0961797 ,  2.018762  ]]], dtype=float32)

.. _Variable-attributes:

The attributes of a data or metdata variable instance is accessed by
the `!attrs` attribute.

.. code-block:: python
   :caption: Example

   >>> import umfive
   >>> u = umfive.File(FILE)  # Open the dataset
   >>> var = u['air_pressure']  # Select a metadata variable
   >>> var
   <umfive.DimensionScale: air_pressure, shape=(5,)>
   >>> var.attrs  # Get the attributes
   {'CLASS': b'DIMENSION_SCALE',
    'NAME': b'netCDF dimension coordinate variable',
    '_Netcdf4Dimid': np.int32(2),
    'axis': 'Z',
    'positive': 'down',
    'standard_name': 'air_pressure',
    'units': 'hPa'}
   >>> var = u['UM_m01s15i201_vn405']  # Select a meta variable
   >>> var
   <umfive.DataVariable: UM_m01s15i201_vn405, shape=(3, 5, 110, 106), dimensions=(time, air_pressure, grid_latitude, grid_longitude)>
   >>> var.attrs
   {'DIMENSION_LIST': (('time',),
                       ('air_pressure',),
                       ('grid_latitude',),
                       ('grid_longitude',)),
    '_FillValue': np.float32(-1.0737418e+09),
    'cell_methods': 'time: mean',
    'coordinates': 'time air_pressure grid_latitude grid_longitude',
    'grid_mapping': 'rotated_latitude_longitude',
    'lbcode': '101',
    'lbproc': '128',
    'lbtim': '121',
    'lbvc': '8',
    'long_name': 'U COMPNT OF WIND ON PRESSURE LEVELS',
    'missing_value': np.float32(-1.0737418e+09),
    'runid': 'aaacf',
    'source': 'UM',
    'standard_name': 'eastward_wind',
    'stash_code': '15201',
    'submodel': '1',
    'um_identity': 'UM_m01s15i201_vn405',
    'um_stash_source': 'm01s15i201',
    'um_version': '4.5',
    'units': 'm s-1'}
