.. _Installation:

Installation
============

The only dependencies required run the software, besides Python, are:

* `numpy`
* `pyfive`
* `xnetcdf`
* `cftime`

`umfive` can be installed using ``pip`` using the command:

.. code-block:: console

    $ pip install umfive

A ``conda`` package, which also installs all of the backend libraries
is available from conda-forge:

.. code-block:: console

    $ conda install -c conda-forge umfive

The library can also be imported directly from the `umfive` source
root directory:

.. code-block:: console

    $ cd umfive
    $ pip install -e . 

    
