# C++ Source Synchronization

`Cpp/` is the single source of truth for the metaSDT C++ backend.

R and Python package directories keep synchronized copies so they can be
prepared for CRAN and PyPI package builds:

- `R/src/cpp`
- `R/inst/include/metaSDT`
- `Python/src/cpp`
- `Python/src/include/metaSDT`

Do not edit files in those synchronized directories directly. Edit
`Cpp/src` and `Cpp/include/metaSDT`, then run CMake from the repository root.

Synchronization runs automatically during:

```sh
cmake -S . -B build
cmake --build build --config Release
```

All project headers should be included with package-style paths such as:

```cpp
#include <metaSDT/model_sdt.hpp>
```
