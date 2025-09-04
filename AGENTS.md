# Repository Guidelines

## Project Structure & Module Organization
- Workspace: ROS 2 sources live in `ros2_ws/src/`.
- Package layout (example `ros2_ws/src/<package_name>/`): `package.xml`, `CMakeLists.txt` (C++/ament_cmake) or `setup.py`/`setup.cfg` (Python/ament_python), `src/`, `include/<package_name>/`, `launch/`, `config/`, `msg/`/`srv/` (if used), `test/`.
- Create new packages: `ros2 pkg create --build-type ament_python <name>` or `ros2 pkg create --build-type ament_cmake <name>`.

## Build, Test, and Development Commands
- `colcon build --symlink-install`: Build all packages in the workspace.
- `source install/setup.bash`: Overlay the build before running tools.
- `colcon test && colcon test-result --all`: Run tests and show a summary.
- `ros2 run <pkg> <node>` / `ros2 launch <pkg> <file.launch.py>`: Execute nodes or launch files locally.

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indent; modules/files `snake_case.py`, classes `PascalCase`, constants `UPPER_SNAKE`.
- C++: Follow `ament_cpplint` and any repo `.clang-format`; files `snake_case.cpp|hpp`; prefer `#pragma once` for headers.
- ROS: package and node names `lower_snake_case`; topic/service/parameter names `lower_snake_case`; keep message fields explicit and documented.
- Linting: Enable standard ament linters (e.g., `ament_flake8`, `ament_pep257`, `ament_cpplint`) and fix warnings before merging.

## Testing Guidelines
- Python tests: `pytest` under `test/` with files named `test_*.py`.
- C++ tests: `gtest` targets added in CMake and registered with ament.
- Launch/integration: prefer `launch_testing` for multi-node scenarios.
- Aim to cover new/changed behavior; include regression tests for fixed bugs.

## Commit & Pull Request Guidelines
- Commits: small, focused, imperative mood. Use Conventional Commits where practical: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- PRs: include a clear description, linked issues, test instructions (build/run commands), and any configuration or launch examples. Update docs/config as needed.
- CI/lint/test must pass before review; avoid introducing new warnings.

## Security & Configuration Tips
- Do not commit secrets. Store runtime parameters in `config/*.yaml` and load via launch files.
- Use namespaces and, when relevant, `ROS_DOMAIN_ID` to avoid cross-talk in multi-robot setups.
