.. _Quick-start:

Quick start
===========

The examples on this page use the ``TODO`` dataset (`download TODO KB
<https://raw.githubusercontent.com/NCAS-CMS/umfive/main/tests/data/TODO>`_).

----

Here is an example of how to use `umfive` to open a dataset
and inspect its contents:

.. code-block:: python

    import umfive

    # Open the dataset
    with umfive.File('test_file') as u:
        # A one-line summary of the dataset
        print(repr(u))

        # A longer summary of the dataset
        print(u)

        # Use the dump() method for an even more detailed view
        u.dump()

        # Use the dump() method with "data=True" for yet more detail
        u.dump(data=True)

        # Access a variable
	var = u['UM_m01s15i201_vn405']

        # A one-line summary of the variable
        print(repr(var))
	
        # Print the variable attributes
        print(var.attrs)

        # Print the data array from the variable
        print(var[...])
	    
.. rubric:: Import the library and open the dataset.

See `umfive.File`.

.. code-block:: python

   >>> import umfive
   >>> u = umfive.File(TODO)

.. rubric:: Display the `repr` description of the dataset

This one-line description includes the dataset name, and how many data
variables and metadata variables there are.

.. code-block:: python
		
   >>> u
   TODO: <umfive.File: 1 data variable, 9 metadata variables>

.. rubric:: Display the `str` description of the dataset

In addition to the `repr` output, this shows some details about each
data and metadata variable. The `umfive.DataVariable` and
`umfive.Variable` variable descriptions indicate which dimensions are
spanned by their data arrays.

.. code-block:: python
		
   >>> print(u)
   TODO: <umfive.File: 1 data variable, 9 metadata variables>
   Data variables:
       UM_m01s15i201_vn405: <umfive.DataVariable: UM_m01s15i201_vn405, shape=(3, 5, 110, 106), dimensions=(time, air_pressure, grid_latitude, grid_longitude)>
   Metadata variables:
       time: <umfive.DimensionScale: time, shape=(3,)>
       bounds2: <umfive.DimensionScale: bounds2, size=2>
       time_bounds: <umfive.Variable: time_bounds, shape=(3, 2), dimensions=(time, bounds2)>
       air_pressure: <umfive.DimensionScale: air_pressure, shape=(5,)>
       grid_latitude: <umfive.DimensionScale: grid_latitude, shape=(110,)>
       grid_latitude_bounds: <umfive.Variable: grid_latitude_bounds, shape=(110, 2), dimensions=(grid_latitude, bounds2)>
       grid_longitude: <umfive.DimensionScale: grid_longitude, shape=(106,)>
       grid_longitude_bounds: <umfive.Variable: grid_longitude_bounds, shape=(106, 2), dimensions=(grid_longitude, bounds2)>
       rotated_latitude_longitude: <umfive.Variable: rotated_latitude_longitude, shape=(), dimensions=()>
	
.. rubric:: Access variable attributes.

See :ref:`Variable-attributes` and :ref:`Variable-data-and-indexing`.

.. code-block:: python
		
   >>> var = u['UM_m01s15i201_vn405']
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
   >>> air_pressure = u['air_pressure']
   >>> air_pressure.attrs
   {'CLASS': b'DIMENSION_SCALE',
    'NAME': b'netCDF dimension coordinate variable',
    '_Netcdf4Dimid': np.int32(2),
    'axis': 'Z',
    'positive': 'down',
    'standard_name': 'air_pressure',
    'units': 'hPa'}
	
.. rubric:: Access variable data.

.. code-block:: python
	
   >>> var[...]
   array([[[[-1.28504544e-01,  3.05374086e-01,  8.93456340e-01, ...,
              4.12485838e+00,  4.27008438e+00,  4.22145319e+00],
            [ 4.74474072e-01,  9.77664232e-01,  1.43804467e+00, ...,
              3.98733783e+00,  4.09150171e+00,  4.04539061e+00],
            [ 8.05260956e-01,  1.24306083e+00,  1.74817479e+00, ...,
              3.80105686e+00,  3.98432851e+00,  3.94045901e+00],
            ...,
            [ 5.67281818e+00,  5.62275362e+00,  5.59070444e+00, ...,
             -1.69666624e+00, -1.95548487e+00, -1.96797502e+00],
            [ 5.14257669e+00,  5.06531000e+00,  5.00876808e+00, ...,
             -1.68950760e+00, -1.70027304e+00, -1.71349967e+00],
            [ 4.59499311e+00,  4.50737619e+00,  4.43036985e+00, ...,
             -1.15786588e+00, -1.21203947e+00, -1.22635651e+00]]]],
         shape=(3, 5, 110, 106), dtype=float32)
   >>> air_pressure[...]
   array([850.00006 , 700.00006 , 500.00003 , 250.00002 ,  50.000004],
         dtype=float32)

.. rubric:: Display a variable's attributes and data array using the
            variable's `~xnetcdf.Variable.dump` method

.. code-block:: python
		
   >>> var.dump(data=True)
   lon: <xnetcdf.Variable: /forecast/lon, shape=(8,), dimensions=(/forecast/lon,)>
       Attributes:
           bounds: '/forecast/lon_bnds'
           standard_name: 'longitude'
           units: 'degrees_east'
       Data float64:
           [ 22.5,  67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5]

.. rubric:: Display the `~xnetcdf.Dataset.dump` description of the
            dataset

In addition to the `~xnetcdf.Dataset.structure` description, this
shows the attributes of all variables and groups (note that the depth
of the group hierarchy shown can be limited with the ``depth`` keyword
argument to `~xnetcdf.Dataset.dump` method).
	     
.. code-block:: python

   >>> nc.dump()
   test.nc: <xnetcdf.Dataset: /, 1 dimension, 1 variable, 1 group>
       Attributes:
           Conventions: 'CF-1.13'
           global_attr_1: np.float64(3.14)
           global_attr_2: 'foo'
       Dimensions:
           bounds2: <xnetcdf.Dimension: /bounds2, size=2>
       Variables:
           time: <xnetcdf.Variable: /time, shape=(), dimensions=()>
               Attributes:
                   standard_name: 'time'
                   units: 'days since 2018-12-01'
       Groups:
           forecast: <xnetcdf.Group: /forecast, 1 dimension, 2 variables, 1 group>
               Dimensions:
                   lon: <xnetcdf.Dimension: /forecast/lon, size=8, unlimited>
               Variables:
                   lon_bnds: <xnetcdf.Variable: /forecast/lon_bnds, shape=(8, 2), dimensions=(/forecast/lon, /bounds2)>
                   lon: <xnetcdf.Variable: /forecast/lon, shape=(8,), dimensions=(/forecast/lon,)>
                       Attributes:
                           bounds: '/forecast/lon_bnds'
                           standard_name: 'longitude'
                           units: 'degrees_east'
               Groups:
                   model: <xnetcdf.Group: /forecast/model, 1 dimension, 3 variables, 0 groups>
                       Attributes:
                           group_attr_1: np.int64(12)
                           group_attr_2: 'bar'
                       Dimensions:
                           lat: <xnetcdf.Dimension: /forecast/model/lat, size=5>
                       Variables:
                           lat_bnds: <xnetcdf.Variable: /forecast/model/lat_bnds, shape=(5, 2), dimensions=(/forecast/model/lat, /bounds2)>
                           lat: <xnetcdf.Variable: /forecast/model/lat, shape=(5,), dimensions=(/forecast/model/lat,)>
                               Attributes:
                                   bounds: '/forecast/model/lat_bnds'
                                   standard_name: 'latitude'
                                   units: 'degrees_north'
                           q: <xnetcdf.Variable: /forecast/model/q, shape=(5, 8), dimensions=(/forecast/model/lat, /forecast/lon)>
                               Attributes:
                                   cell_methods: 'area: mean'
                                   coordinates: 'time'
                                   float: np.float64(49.0)
                                   float32: np.float32(49.0)
                                   float64: np.float64(49.0)
                                   int: np.int64(49)
                                   int16: np.int16(49)
                                   int32: np.int32(49)
                                   int64: np.int64(49)
                                   int8: np.int8(49)
                                   list1: array([2, 3], dtype=int8)
                                   list10: array([], dtype=float64)
                                   list11: ['a', 'bb', 'ccc']
                                   list12: ['a', '1', '2.5']
                                   list13: 'a'
                                   list14: ['a', 'bb']
                                   list2: array([2, 3], dtype=int16)
                                   list3: array([2, 3])
                                   list4: array([2, 3], dtype=int32)
                                   list5: array([2., 3.], dtype=float32)
                                   list6: array([2., 3.])
                                   list7: array([2, 3])
                                   list8: np.int32(2)
                                   list9: array([], dtype=int32)
                                   standard_name: 'specific_humidity'
                                   string1: '1'
                                   string2: 'a'
                                   string3: 'kg m-2'
                                   string4: ''
                                   string5: ' '
                                   string6: ''
                                   string7: ''
                                   string8: ''
                                   string9: ''
                                   uint16: np.uint16(49)
                                   uint32: np.uint32(49)
                                   uint64: np.uint64(49)
                                   uint8: np.uint8(49)
    
.. rubric:: Display the `~xnetcdf.Dataset.dump` description of the
            dataset with argument ``data=True``

In addition to the `~xnetcdf.Dataset.dump` description, this shows the
data array values (abbreviated if large, which is not the case here)
of all variables.
	     
.. code-block:: python

   >>> nc.dump(data=True)
   test.nc: <xnetcdf.Dataset: /, 1 dimension, 1 variable, 1 group>
       Attributes:
           Conventions: 'CF-1.13'
           global_attr_1: np.float64(3.14)
           global_attr_2: 'foo'
       Dimensions:
           bounds2: <xnetcdf.Dimension: /bounds2, size=2>
       Variables:
           time: <xnetcdf.Variable: /time, shape=(), dimensions=()>
               Attributes:
                   standard_name: 'time'
                   units: 'days since 2018-12-01'
               Data int32:
                   31
       Groups:
           forecast: <xnetcdf.Group: /forecast, 1 dimension, 2 variables, 1 group>
               Dimensions:
                   lon: <xnetcdf.Dimension: /forecast/lon, size=8, unlimited>
               Variables:
                   lon_bnds: <xnetcdf.Variable: /forecast/lon_bnds, shape=(8, 2), dimensions=(/forecast/lon, /bounds2)>
                       Data float64:
                           [[  0.,  45.],
                            [ 45.,  90.],
                            [ 90., 135.],
                            [135., 180.],
                            [180., 225.],
                            [225., 270.],
                            [270., 315.],
                            [315., 360.]]
                   lon: <xnetcdf.Variable: /forecast/lon, shape=(8,), dimensions=(/forecast/lon,)>
                       Attributes:
                           bounds: '/forecast/lon_bnds'
                           standard_name: 'longitude'
                           units: 'degrees_east'
                       Data float64:
                           [ 22.5,  67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5]
               Groups:
                   model: <xnetcdf.Group: /forecast/model, 1 dimension, 3 variables, 0 groups>
                       Attributes:
                           group_attr_1: np.int64(12)
                           group_attr_2: 'bar'
                       Dimensions:
                           lat: <xnetcdf.Dimension: /forecast/model/lat, size=5>
                       Variables:
                           lat_bnds: <xnetcdf.Variable: /forecast/model/lat_bnds, shape=(5, 2), dimensions=(/forecast/model/lat, /bounds2)>
                               Data float64:
                                   [[-90., -60.],
                                    [-60., -30.],
                                    [-30.,  30.],
                                    [ 30.,  60.],
                                    [ 60.,  90.]]
                           lat: <xnetcdf.Variable: /forecast/model/lat, shape=(5,), dimensions=(/forecast/model/lat,)>
                               Attributes:
                                   bounds: '/forecast/model/lat_bnds'
                                   standard_name: 'latitude'
                                   units: 'degrees_north'
                               Data float64:
                                   [-75., -45.,   0.,  45.,  75.]
                           q: <xnetcdf.Variable: /forecast/model/q, shape=(5, 8), dimensions=(/forecast/model/lat, /forecast/lon)>
                               Attributes:
                                   cell_methods: 'area: mean'
                                   coordinates: 'time'
                                   float: np.float64(49.0)
                                   float32: np.float32(49.0)
                                   float64: np.float64(49.0)
                                   int: np.int64(49)
                                   int16: np.int16(49)
                                   int32: np.int32(49)
                                   int64: np.int64(49)
                                   int8: np.int8(49)
                                   list1: array([2, 3], dtype=int8)
                                   list10: array([], dtype=float64)
                                   list11: ['a', 'bb', 'ccc']
                                   list12: ['a', '1', '2.5']
                                   list13: 'a'
                                   list14: ['a', 'bb']
                                   list2: array([2, 3], dtype=int16)
                                   list3: array([2, 3])
                                   list4: array([2, 3], dtype=int32)
                                   list5: array([2., 3.], dtype=float32)
                                   list6: array([2., 3.])
                                   list7: array([2, 3])
                                   list8: np.int32(2)
                                   list9: array([], dtype=int32)
                                   standard_name: 'specific_humidity'
                                   string1: '1'
                                   string2: 'a'
                                   string3: 'kg m-2'
                                   string4: ''
                                   string5: ' '
                                   string6: ''
                                   string7: ''
                                   string8: ''
                                   string9: ''
                                   uint16: np.uint16(49)
                                   uint32: np.uint32(49)
                                   uint64: np.uint64(49)
                                   uint8: np.uint8(49)
                               Data float32:
                                   [[0.007, 0.034, 0.003, 0.014, 0.018, 0.037, 0.024, 0.029],
                                    [0.023, 0.036, 0.045, 0.062, 0.046, 0.073, 0.006, 0.066],
                                    [0.11 , 0.131, 0.124, 0.146, 0.087, 0.103, 0.057, 0.011],
                                    [0.029, 0.059, 0.039, 0.07 , 0.058, 0.072, 0.009, 0.017],
                                    [0.006, 0.036, 0.019, 0.035, 0.018, 0.037, 0.034, 0.013]]
    
.. rubric:: Display the `~xnetcdf.Dataset.ncdump` description of the
            dataset

Represents the dataset in CDL, omitting the data arrays, reproducing
``$ ncdump -h test.nc`` output.
	     
.. code-block:: python
	     
   >>> nc.ncdump()
   netcdf test.nc {
   dimensions:
        bounds2 = 2 ;
   variables:
        int time ;
            time:standard_name = "time" ;
            time:units = "days since 2018-12-01" ;
   
   // global attributes:
            :Conventions = "CF-1.13" ;
            :global_attr_1 = 3.14 ;
            :global_attr_2 = "foo" ;
   
   group: forecast {
     dimensions:
          lon = UNLIMITED ; // (8 currently)
     variables:
          double lon_bnds(lon, bounds2) ;
          double lon(lon) ;
              lon:bounds = "/forecast/lon_bnds" ;
              lon:standard_name = "longitude" ;
              lon:units = "degrees_east" ;
   
     group: model {
       dimensions:
            lat = 5 ;
       variables:
            double lat_bnds(lat, bounds2) ;
            double lat(lat) ;
                lat:bounds = "/forecast/model/lat_bnds" ;
                lat:standard_name = "latitude" ;
                lat:units = "degrees_north" ;
            float q(lat, lon) ;
                q:cell_methods = "area: mean" ;
                q:coordinates = "time" ;
                q:float = 49. ;
                q:float32 = 49.f ;
                q:float64 = 49. ;
                q:int = 49LL ;
                q:int16 = 49s ;
                q:int32 = 49 ;
                q:int64 = 49LL ;
                q:int8 = 49b ;
                q:list1 = 2b, 3b ;
                q:list10 = "" ;
                string q:list11 = "a", "bb", "ccc" ;
                string q:list12 = "a", "1", "2.5" ;
                q:list13 = "a" ;
                string q:list14 = "a", "bb" ;
                q:list2 = 2s, 3s ;
                q:list3 = 2LL, 3LL ;
                q:list4 = 2, 3 ;
                q:list5 = 2.f, 3.f ;
                q:list6 = 2., 3. ;
                q:list7 = 2LL, 3LL ;
                q:list8 = 2 ;
                q:list9 = "" ;
                q:standard_name = "specific_humidity" ;
                q:string1 = "1" ;
                q:string2 = "a" ;
                q:string3 = "kg m-2" ;
                q:string4 = "" ;
                q:string5 = " " ;
                q:string6 = "" ;
                q:string7 = "" ;
                q:string8 = "" ;
                q:string9 = "" ;
                q:uint16 = 49US ;
                q:uint32 = 49U ;
                q:uint64 = 49ULL ;
                q:uint8 = 49UB ;
   
       // group attributes:
                :group_attr_1 = 12LL ;
                :group_attr_2 = "bar" ;
       } // group model
     } // group forecast
   }

