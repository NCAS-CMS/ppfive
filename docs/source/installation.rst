.. _Installation:

Installation
============

The only dependexncies required run the software, besides Python, are
`numpy` (version 2.0.0 or later) and `cftime`.

`umfive` can be installed using ``pip`` using the command:

.. code-block:: console

    $ pip install umfive

To install with all of the backend libraries using ``pip``:

.. code-block:: console

    $ pip install umfive[all]

A ``conda`` package, which also installs all of the backend
libraries is available from conda-forge:

.. code-block:: console

    $ conda install -c conda-forge umfive

The library can also be imported directly from the `umfive` source
root directory:

.. code-block:: console

    $ cd umfive
    $ pip install -e . 
