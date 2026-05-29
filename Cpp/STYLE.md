# C++ Style Notes

This directory uses a conservative, explicit C++ style for model and
estimator code.

## `auto`

Use `auto` when the type is obvious or when spelling it out is noisy:

- Iterators, for example `auto it = values.find(key);`
- Range loops, for example `for (const auto& item : items)`
- Lambdas, for example `auto build_task = [](...) { ... };`
- Very long template-heavy implementation details

Prefer explicit types when they make model or estimator code easier to read:

- Numeric values, for example `double loss = ...`
- Eigen objects, for example `Eigen::VectorXd free_params(...)`
- Project result objects, for example `SubjectFitResult result`
- Model outputs, for example `MatrixProb<double> prob = ...`
- Parameter containers, for example
  `std::unordered_map<std::string, std::vector<double>> best_params`

## Local Values

Mark local variables `const` when they are not modified after initialization.
Do not add `const` when a variable is intentionally updated later.

Use references in loops to avoid copies:

- `for (const auto& item : items)` for read-only access
- `for (auto& item : items)` when elements are modified

## Casts

Use explicit C++ casts for narrowing and library index conversions:

- `static_cast<int>(values.size())`
- `static_cast<std::size_t>(index)`
- `static_cast<Eigen::Index>(index)`

Do not use C-style casts.

## Includes

Order includes as follows:

1. The corresponding project header.
2. Other project headers.
3. Third-party headers.
4. Standard library headers.
5. Conditional OpenMP includes.

Keep a blank line between include groups.

## Errors And Printing

Use `std::invalid_argument` for invalid user inputs or missing required fields.
Use `std::runtime_error` for runtime failures that are not simple input
validation.

Avoid casual backend printing. Progress output should stay in
`progress_bar.cpp` or in existing controlled progress paths. Preserve existing
estimator status and result semantics when handling optimizer failures.

## C++ Version

The C++ code targets C++17. Do not use C++20 designated initializers or other
features that would change the required language standard.
