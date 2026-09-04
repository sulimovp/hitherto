# Hitherto — contributor notes

Architecture: `ARCHITECTURE.md`. Current implementation snapshot: `docs/STATUS.md`.

## Commands

```bash
pip install -e ".[dev,web]"
./scripts/verify.sh
hitherto ping
hitherto list-profiles
hitherto-web
```

## Verification

1. `./scripts/verify.sh` must exit 0 before claiming work is done.
2. Web/UI changes: walk `docs/VERIFICATION.md`.
3. Secrets stay in env. Never commit `.env`.

## Design

- Evidence over verdict; every synthesis claim needs a citation.
- Ecosystem specialisation lives in `profiles/*.yaml`.
