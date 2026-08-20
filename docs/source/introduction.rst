Introduction
============

Overview
--------

`umfive` is a Python open source library for representing `UK Met
Office PP and UM fields file datasets
<https://artefacts.ceda.ac.uk/badc_datadocs/um/umdp_F3-UMDPF3.pdf>`_
with `CF-netCDF <https://cfconventions.org/>`_-like structures that follow the `pyfive
<https://pyfive.readthedocs.io>`_ API.

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
    
An example of how to use `umfive` to open a dataset and inspect its
contents:

.. code-block:: python

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
   >>> u['time'].attrs  # Get a variable's attributes
   {'CLASS': b'DIMENSION_SCALE',
    'NAME': b'netCDF dimension coordinate variable',
    '_Netcdf4Dimid': np.int32(0),
    'axis': 'T',
    'bounds': 'time_bounds',
    'calendar': '360_day',
    'standard_name': 'time',
    'units': 'days since 2159-1-1'}
   >>> u['time'][...]  # Get a variable's data array
   array([ 510.,  870., 1230.])

A netCDF (as opposed to HDF5) view is easily found via the `xnetcdf
<https://xnetcdf.readthedocs.io>`_ library:

.. code-block:: python

   >>> import xnetcdf
   >>> x = xnetcdf.Dataset(u)
   >>> print(x)
   path/to/dataset: <xnetcdf.Dataset: /, 5 dimensions, 9 variables, 0 groups>
        Dimensions:
            time: <xnetcdf.Dimension: /time, size=3>
            bounds2: <xnetcdf.Dimension: /bounds2, size=2>
            air_pressure: <xnetcdf.Dimension: /air_pressure, size=5>
            grid_latitude: <xnetcdf.Dimension: /grid_latitude, size=110>
            grid_longitude: <xnetcdf.Dimension: /grid_longitude, size=106>
        Variables:
            UM_m01s15i201_vn405: <xnetcdf.Variable: /UM_m01s15i201_vn405, shape=(3, 5, 110, 106), dimensions=(/time, /air_pressure, /grid_latitude, /grid_longitude)>
            time: <xnetcdf.Variable: /time, shape=(3,), dimensions=(/time,)>
            time_bounds: <xnetcdf.Variable: /time_bounds, shape=(3, 2), dimensions=(/time, /bounds2)>
            air_pressure: <xnetcdf.Variable: /air_pressure, shape=(5,), dimensions=(/air_pressure,)>
            grid_latitude: <xnetcdf.Variable: /grid_latitude, shape=(110,), dimensions=(/grid_latitude,)>
            grid_latitude_bounds: <xnetcdf.Variable: /grid_latitude_bounds, shape=(110, 2), dimensions=(/grid_latitude, /bounds2)>
            grid_longitude: <xnetcdf.Variable: /grid_longitude, shape=(106,), dimensions=(/grid_longitude,)>
            grid_longitude_bounds: <xnetcdf.Variable: /grid_longitude_bounds, shape=(106, 2), dimensions=(/grid_longitude, /bounds2)>
            rotated_latitude_longitude: <xnetcdf.Variable: /rotated_latitude_longitude, shape=(), dimensions=()>


See :ref:`Quick-start` for more examples.


`pyfive` compatibility
----------------------

`umfive` is designed to be *pyfive-compatible* meaning that, as far as
possible, code that manipulates a `pyfive.File` object will work
identically if that object is replaced by a `umfive.File` object.

In addition, `umfive.File` is registered as a virtual subclass of
`pyfive.File`; and `umfive.DataVariable`, `umfive.Variable`, and
`umfive.DimensionScale` are all registered as virtual subclasses of
`pyfive.Dataset`.

Dataset definitions
-------------------

A dataset can be passed to `umfive.File` with one of the following
dataset definitions:

- A string-like path name to the dataset (such as `str` or
  `pathlib.Path` instance).
  
- A file-like object that accesses the dataset (such as
  `io.BufferedReader` or the result of an `fsspec` file system open)
  
- A subclass of `umfive.ByteReader` (such as `umfive.LocalPosixReader`
  or `umfive.FileObjReader`).

Attributes
----------

The following attributes, derived from the lookup headers, are added
to data variables (i.e. added to `umfive.DataVariable` instances):

===================  =========================================================
Attribute            Description
===================  =========================================================
``lbcode``           The value of LBCODE (word 16, grid code)
``lbproc``           The value of LBPROC (word 25, pocessing code)
``lbtim``            The value of LBTIM (word 13, time indicator)
``lbvc``             The value of LBVC (word 26, vertical co-ordinate type)
``runid``            The runid decoded from LBEXP (word 28, experiment number)
``source``           The source decoded from LBSRCE (word 38)
``stash_code``       The value of LBUSER(4) (word 42, stash code)
``submodel``         The value of LBUSER(7) (word 45, model code)
``um_identity``      A definitive identifier for the field
``um_stash_source``  The stash code and source
``um_version``       The UM version
===================  =========================================================

A selection of :ref:`CF-attributes` are also set on the data and
metadata variables.

CF attributes
-------------

The following CF attributes are derived from the lookup headers and,
are added to the output variables, or as global attributes, where
possible and appropriate:

=================  =======================================
CF attribute       CF variable/global usage
=================  =======================================
``_FillValue``     Data
``add_offset``     Data
``axis``           Coordinate, Auxiliary coordinate
``bounds``         Coordinate, Domain ancillary
``calendar``       Coordinate
``climatology``    Coordinate
``Conventions``    Global
``coordinates``    Data
``cell_methods``   Data
``formula_terms``  Coordinate
``grid_mapping``   Data
``long_name``      Data, Coordinate, Auxiliary coordinate,
                   Domain ancillary
``missing_value``  Data
``positive``       Coordinate, Auxiliary coordinate
``scale_factor``   Data
``source``         Data
``standard_name``  Data, Coordinate, Auxiliary coordinate,
                   Domain ancillary
``units``          Data, Coordinate, Auxiliary coordinate,
                   Domain ancillary
=================  =======================================

Performance
-----------

The read of a dataset is lazy in that only the metadata (i.e. the
lookup headers and any extra data) are accessed during the initial
read. A data array in the dataset is then accessed on demand, and then
only for the part of the data array requested by the indexing. Data
reads are parallelised over the 2-d slices stored for each lookup
header (see `umfive.File.set_parallelism` and
`umfive.File.get_parallelism`).
