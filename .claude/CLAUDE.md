# CLAUDE.md

## General writing rules

- Use ASCII only. Do not use non-ASCII characters in code, comments, strings, or documentation unless the file already requires them.
- Do not align code with extra spaces for visual formatting. Use normal single spacing only.
    - Example: write `y = x`, not `y =     x`.
- Keep comments short, direct, and useful.
- Do not write overly descriptive, theatrical, or AI-sounding comments.
- Do not add comments that restate obvious code.
- Prefer simple, readable code over clever code.

## Python conventions

- Follow PEP 8 style.
- Prefer explicit code over implicit behavior.
- Prefer small functions with one clear responsibility.
- Use descriptive variable and function names.
- Use type hints for public functions and non-trivial internal functions.
- Prefer f-strings for formatting.
- Prefer `pathlib` over `os.path`.
- Prefer `enumerate`, `zip`, comprehensions, and generators when they improve clarity.
- Avoid deeply nested logic; refactor into helper functions when needed.
- Avoid unnecessary abstraction.
- Avoid premature optimization.
- Keep imports organized:
  - standard library
  - third-party
  - local
- Remove unused imports and dead code.
- Avoid mutable default arguments.
- Use `None` checks explicitly when needed.
- Prefer `is` / `is not` for singleton checks.
- Use `snake_case` for functions, variables, and module names.
- Use `PascalCase` for classes.
- Use `UPPER_SNAKE_CASE` for constants.

## Formatting

- Do not manually space-justify code.
- Keep line length reasonable. Prefer 88 characters unless the project already uses a different standard.
- Use blank lines to separate logical blocks, not to pad formatting.
- Do not add trailing whitespace.
- Keep diffs minimal and localized.

## Comments and docstrings

- Only add comments when they explain non-obvious intent, constraints, or tradeoffs.
- Do not narrate what the code already makes clear.
- Do not write thought-process comments.
- Keep docstrings concise and factual.
- Prefer explaining "why" over "what" when a comment is needed.

## Editing behavior

- Preserve existing project style unless it conflicts with these rules.
- Prefer the minimal change that solves the problem.
- Do not introduce unrelated refactors.
- Do not rename symbols unless necessary.