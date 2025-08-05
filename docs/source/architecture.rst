Architecture
============

Principles
----------

- Ports (Factory, and Storage) have a generic class : `Port` from which they inherit a few properties
- A ship is initialized with a list of ports, is responsible for picking its destination port and checking the weather
- We use stateSaver as a tool to save of every object at every period


KPI Generation
--------------

- KPIS are generated after the simulation is run in a separate class
- The class reads the result from the simulation and calculate each kpi and graph directly from the data returned from the simulation
- If a folder has been passed as the `--config` parameter, then the program also generates graphs that are aggregates of results from the different scenarios

