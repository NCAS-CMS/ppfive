.. _CF-mappings:

CF mappings
===========

.. _CF-attributes:

CF attributes
-------------

The following `CF
<https://cfconventions.org/cf-conventions/cf-conventions.html>`_
attributes are derived from the lookup headers and, are added to the
output variables, or as global attributes, where possible and
appropriate:

=================  ========================================================
CF attribute       CF variable/global usage
=================  ========================================================
``_FillValue``     Data
``add_offset``     Data
``axis``           Coordinate, Auxiliary coordinate
``bounds``         Coordinate, Domain ancillary
``calendar``       Coordinate
``climatology``    Coordinate
``Conventions``    Global
``coordinates``    Data
``cell_methods``   Data
``formula_terms``  Coordinate (see :ref:`Orography`)
``grid_mapping``   Data
``long_name``      Data, Coordinate, Auxiliary coordinate, Domain ancillary
``missing_value``  Data
``positive``       Coordinate, Auxiliary coordinate
``scale_factor``   Data
``source``         Data
``standard_name``  Data, Coordinate, Auxiliary coordinate, Domain ancillary
``units``          Data, Coordinate, Auxiliary coordinate, Domain ancillary
=================  ========================================================

.. _Orography:
   
Orography
---------

When a data variable has vertical coordinates that are defined by a
2-d orography field (such as `atmosphere hybrid height coordinates
<https://cfconventions.org/cf-conventions/cf-conventions.html#atmosphere-hybrid-height-coordinate>`_),
if the orography field is present as a data variable in the same
dataset then it will be referenced by the ``formula_terms`` attribute
of any applicable vertical coordinate variables.
