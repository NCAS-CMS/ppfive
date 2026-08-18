Introduction
============

Overview
--------

`umfive` is a Python open source library for representing `UK Met
Office PP and UM fields file datasets
<https://artefacts.ceda.ac.uk/badc_datadocs/um/umdp_F3-UMDPF3.pdf>`_
with the `pyfive` HDF5 API. It maps the data and metadata described by
the dataset's lookup headers onto HDF5 dataset structures.

The contents of a PP or UM fields file dataset are mapped to a
`umfive.File` object that contains CF dimensions and coordinate
variables (`umfive.DimensionScale` objects); auxiliary coordinate,
domain ancillary, bounds, and grid mapping variables
(`umfive.Variable` objects); and data variables (as
`umfive.DataVariable` objects).

32-bit and 64-bit PP and UM fields files of any endian-ness can be
read.

2-d "slices" within a single file are always combined, where possible,
into fields with 3-d or 4-d data.

A simple example
----------------
    
An example of how to use `umfive` to open a dataset and inspect its
contents:

.. code-block:: python

   >>> import umfive
   >>> u = umfive.File('path/to/your/dataset')  # Open the dataset
   >>> print(u)  # Inspect the dataset contents
   path/to/your/dataset <umfive.File: 1 data variable, 7 metadata variables>
   Data variables:
       UM_m01s00i001_vn405: <umfive.DataVariable: UM_m01s00i001_vn405, shape=(3, 73, 96), dimensions=(time, latitude, longitude)>
   Metadata variables:
       time: <umfive.DimensionScale: time, shape=(3,)>
       bounds2: <umfive.DimensionScale: bounds2, size=2>
       time_bounds: <umfive.Variable: time_bounds, shape=(3, 2), dimensions=(time, bounds2)>
       latitude: <umfive.DimensionScale: latitude, shape=(73,)>
       latitude_bounds: <umfive.Variable: latitude_bounds, shape=(73, 2), dimensions=(latitude, bounds2)>
       longitude: <umfive.DimensionScale: longitude, shape=(96,)>
       longitude_bounds: <umfive.Variable: longitude_bounds, shape=(96, 2), dimensions=(longitude, bounds2)>
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

UM Attributes
-------------

The following attributes, derived from the lookup headers, are added
to data variables (i.e. added to `umfive.DataVariable` instances):

===================  ================================================
Attribute            Description
===================  ================================================
``lbcode``           The value of LBCODE (grid code)
``lbproc``           The value of LBPROC (pocessing code)
``lbtim``            The value of LBTIM (time indicator)
``lbvc``             The value of LBVC (vertical co-ordinate type)
``runid``            The runid decoded from LBEXP (experiment number)
``source``           The source decoded from LBSRCE
``stash_code``       The value of LBUSER(4) (stash code)
``submodel``         The value of LBUSER(7) (model code)
``um_identity``      A definitive identifier for the field
``um_stash_source``  The stash code and source
``um_version``       The UM version
===================  ================================================

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
