# Project Guidelines

## Configurable defaults

Every tunable parameter (thresholds, sizes, model names, etc.) must be:
- A function/constructor parameter with a sensible default value
- Never hardcoded inline without being surfaced as a configurable knob

This keeps functions composable and CLI-friendly without forcing callers to always specify everything.

## Container runtime

Using **Podman**, not Docker. Use `podman` / `podman compose` commands.
