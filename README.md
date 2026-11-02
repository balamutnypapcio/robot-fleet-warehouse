# robot-fleet-warehouse

Repository for group project - Event Based Control.

Supervisory control of a robot fleet in a grid warehouse: formal model, deadlock-avoiding
supervisor, discrete-event simulator, safety verification and (later) a ROS 2 / Gazebo demo.

## Repository layout

| Directory        | Contents                                                              |
|------------------|-----------------------------------------------------------------------|
| `fleet/`         | Model core - **pure functions only** (no time, no graphics, no ROS)  |
| `sim/`           | Discrete-event simulator and animation                                |
| `verification/`  | Runtime monitors and exhaustive reachability analysis                 |
| `experiments/`   | KPI logging and policy comparison experiments                         |
| `layouts/`       | Warehouse layout files                                                |
| `tests/`         | Pytest test suite                                                     |
| `docs/`          | Reports and formal model write-up                                     |
| `ros2_ws/`       | ROS 2 workspace (added later)                                         |

> **Architecture rule:** `fleet/` must not import anything time-, graphics- or ROS-related.
> The simulator and the ROS 2 `fleet_supervisor` node import it unchanged.

## Install

Requires Python 3.10+.

```bash
git clone https://github.com/balamutnypapcio/robot-fleet-warehouse.git
cd robot-fleet-warehouse
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"   # editable install + dev tools (pytest, ruff)
```

## Run tests and linter

```bash
pytest
ruff check .
ruff format --check .   # use `ruff format .` to auto-format
```

CI runs the same three commands on every push and pull request.

## Run the simulator

The simulator is not implemented yet (see issue #9). Once it is, it will be started with:

```bash
python -m sim
```

## Workflow

- Create a branch per issue, open a pull request to `main`.
- `main` is protected: a PR needs green CI and one approving review before merging.
