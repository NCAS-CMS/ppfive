Introduction
============

Overview
--------

`umfive` is a Python open source library for representing `UK Met
Office PP and UM fields file datasets
<https://artefacts.ceda.ac.uk/badc_datadocs/um/umdp_F3-UMDPF3.pdf>`_
with `CF-netCDF <https://cfconventions.org/>`_-like structures that
follow the API of `pyfive <https://pyfive.readthedocs.io>`_, an HDF5
reader.

The contents of a PP or UM fields file dataset are mapped to a
`umfive.File` object that follows the `CF conventions
<https://cfconventions.org/>`_ in that it contains data variables
(`umfive.DataVariable` objects); dimensions and coordinate variables
(`umfive.DimensionScale` objects); and auxiliary coordinate, domain
ancillary, bounds, and grid mapping variables (`umfive.Variable`
objects).

32-bit and 64-bit PP and fields files of any endian-ness can be read.

2-d "slices" defined by a single lookup headers are always combined,
where possible, into fields with 3-d or 4-d data.

A simple example
----------------
    
How to use `umfive` to open a dataset and inspect its contents:

.. code-block:: python
   :caption: Example

   >>> import umfive
   >>> u = umfive.File('path/to/dataset')  # Open the dataset
   >>> print(u)  # Inspect the dataset contents
   path/to/dataset: <umfive.File: 1 data variable, 9 metadata variables>
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
   >>> u['UM_m01s15i201_vn405']  # Get a variable's attributes
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
   >>> u['time'][...]  # Get a variable's data array
   array([ 510.,  870., 1230.])

See :ref:`Quick-start` and :ref:`CF-netCDF-structure` for more
examples.

Dataset definitions
-------------------

A dataset can be passed to `umfive.File` with one of the following
dataset definitions:

- A string-like path name to a local dataset (such as `str` or
  `pathlib.Path` instance).
  
- A file-like object that accesses a local or remote dataset (such as
  `io.BufferedReader` or the result of an `fsspec` file system open)
  
- A `umfive.LocalPosixReader` or `umfive.FileObjReader` object (or any
  subclass of `umfive.ByteReader`) that accesses a local or remote
  dataset.

`pyfive` compatibility
----------------------

`umfive` is designed to be `pyfive
<https://pyfive.readthedocs.io>`_-compatible meaning that, as far as
possible, code that manipulates a `pyfive.File` object will work
identically if that object is replaced by a `umfive.File` object.

In addition, `umfive.File` is registered as a virtual subclass of
`pyfive.File`; and `umfive.DataVariable`, `umfive.Variable`, and
`umfive.DimensionScale` are all registered as virtual subclasses of
`pyfive.Dataset`.

Performance
-----------

The read of a dataset is lazy in that only the metadata (i.e. the
lookup headers and any extra data) are accessed during the initial
read of the dataset. A data array in the dataset is then accessed on
demand, and then only for the part of the data array requested by the
indexing. Data array reads are parallelised over the 2-d slices
defined by each lookup header (see `umfive.File.set_parallelism` and
`umfive.File.get_parallelism`).
