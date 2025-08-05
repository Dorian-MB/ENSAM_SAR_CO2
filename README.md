
# eco2_normandy - CO₂ Transport Simulation

## Quickstart / Run the Simulation

To quickly get started with the simulation, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd eco2_normandy
    ```

2. **Install dependencies**:
    - Install the required Python dependencies using `pip` or `poetry`:
    ```bash
    pip install -r requirements.txt
    ```
    or if you're using Poetry:
    ```bash
    poetry install
    ```

3. **Build the executable**:
    - After installing the dependencies, build the simulation executable using PyInstaller:
    ```bash
    make exe
    ```
    or in Poweshell
    ```powershell
    python -m poetry run python -m PyInstaller --onefile --add-data "KPIS\template.html;KPIS\" --hidden-import xlsxwriter configurable_main.py
    ```

    This will generate an `.exe` file inside the `dist/` folder.

4. **Run the simulation**:
    - To run the simulation, you must place the generated `.exe` file in the same directory as your `config.yaml` file.
    - Navigate to the directory containing the `config.yaml` file and the executable (`.exe`).
    - Run the `.exe` file:
    ```bash
    ./simulate_co2_transportation.exe
    ```

    - You can also specify the path of the config/configs you want to use by using --config:
    ```bash
    ./simulate_co2_transportation.exe --config ..\scenarios
    ```

    If no --config is specified, the simulation will use the `config.yaml` in the same directory for its configuration and execute the transport model.
    If the value of --config is a directory containing multiple .yaml configs, they will all be simulated, and a directory  `le_havre_capacity_evolution_comparison` will be created with a graph to compare the different scenarios.

---

## Configuration Syntax

### List Format for Values

All values must be enclosed in a list, even if it’s a single entry. This is to ensure consistent parsing by the script.

#### Incorrect Format:
```yaml
name: Ship 1
```

#### Correct Format:
```yaml
name: [Ship 1]
```

---

### Creating a Range of Values

You can specify a range of values using the `range` keyword. This is useful for generating a sequence of numbers or values in a specified range.

Example:
```yaml
ships:
  - name: [Ship 1]
    capacity_max:
      range: [20, 30000, 1000]
```

This will create a series of values for `capacity_max`, starting at `20`, ending at `30000`, with an increment of `1000`. The resulting values will be: `[20, 1020, 2020, 3020, ..., 29000]`.

---

### Defining Multiple Entities

- **Multiple Ships and Storages**: You can define multiple ships and storage ports.
- **One Factory**: Only one factory can be defined in the configuration. If more than one factory is specified, an error will occur.

---

### Respecting the `num_ships` Parameter

The `general.num_ships` parameter will be respected by the script. If fewer ships are defined than specified by `num_ships`, the script will automatically generate additional ships by duplicating the existing ones to meet the required number.

---

### Example Configuration

Here is an example of a valid configuration file:

```yaml
general:
  num_ships: 5

ships:
  - name: [Ship 1]
    capacity_max:
      range: [20, 30000, 1000]

storages:
  - type: [Storage A]
  - type: [Storage B]

factory:
  name: Factory 1
  location: [Port X]
```

In this example:
- The script will ensure there are **5 ships** in total (duplicating ships if necessary).
- The first ship will have a `capacity_max` ranging from 20 to 30000, with an increment of 1000.
- Two storage ports and a single factory are defined.

---

By adhering to the above configuration syntax, your file will be properly parsed, and the simulation will run as expected.

---

### Final Notes

- Make sure your configuration is structured correctly as described to avoid parsing errors.
- The simulation handles a variety of factors affecting the transport of CO₂, from weather conditions to maintenance issues, ensuring a realistic model of the logistics involved.

---

## Project Overview

This project models the transport of liquefied CO₂ between a liquefaction port (Factory) and one or more storage ports (Storage). The transportation is done by ship.

---

## Entities and Components

### Liquefaction Port (Factory)

- **Production Maintenance**: Maintenance phases may impact the production of liquefied CO₂, slowing down the rate at which the tanks are filled.
- **Pump Maintenance**: The pumps used to load a ship may undergo maintenance phases, reducing the speed at which a ship is filled.
- **Excess CO₂**: If the tanks are full while production is ongoing, extra CO₂ must be vented, which incurs a fee.

#### Key Elements at the Liquefaction Port:
- **Tanks**: Store the CO₂ before it’s loaded onto the ship.
- **Production Pumps**: Fill the tanks with liquefied CO₂.
- **Loading Pumps**: Transfer CO₂ from tanks to the ship.
- **Docks**: Locations where ships dock to load the CO₂.

### Storage Port (Storage)

- **Pump Maintenance**: Slows down unloading from the ship.
  
#### Key Elements at the Storage Port:
- **Unloading Pumps**: Offload CO₂ from the ship.

### Ship (Ship)

- **Weather Impact**: The ship's navigation can be affected by weather conditions:
    - **During Navigation**: The ship's speed will decrease depending on the severity of the weather.
    - **At Dock**: Ships may have to wait in port until bad weather passes.
  
- **Docks**: Ships require an available dock to be able to dock.
- **Pilot**: A pilot is required for a ship to enter or leave port. There is a 2-hour wait time for this, which can be buffered while docked.
- **Lock**: Liquefaction ports have locks for ship access. Ships can only enter or exit if the lock is available.

