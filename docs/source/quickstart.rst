QuickStart
==========

Installation
------------

To install the solution, first install poetry 

.. code-block:: console

    $ python -m pip install poetry
    $ poetry install

Quick configuration
-------------------

The default `config.yaml` provided at the root of repository contains an example of a simulation making use of most of the options available.

It is divided in 5 main parts :

- `general` : the general settings for the simulation, number of periods, distances between ports, conversion rate.
- `KPIS` : data to calculate the financial kpis at the end of the process 
- `factory` : the parameters describing the factory
- `storages` : the parameters describing the storages
- `ships` : the parameters describing the ships


Run the program
---------------

- To run the simulation, you must place the generated `.exe` file in the same directory as your `config.yaml` file.
- Navigate to the directory containing the `config.yaml` file and the executable (`.exe`).
- Run the `.exe` file:

.. code-block:: console

    $ ./simulate_co2_transportation.exe


- You can also specify the path of the config/configs you want to use by using --config:

.. code-block:: console

    $ ./simulate_co2_transportation.exe --config ../scenarios


If no ``--config`` is specified, the simulation will use the `config.yaml` in the same directory for its configuration and execute the transport model.
If the value of ``--config`` is a directory containing multiple .yaml configs, they will all be simulated, and a directory  `le_havre_capacity_evolution_comparison` will be created with a graph to compare the different scenarios.
