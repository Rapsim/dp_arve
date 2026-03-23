# Data Structure

This project keeps raw/large data out of Git. Place data locally under `data/` with this structure:

```text
data/
  hydrique_ml/
    Archive previsions Hydrique ML Arve-Bout du Monde.json
  observations/
    2170_Abfluss_10-Min-Mittel_1999-01-01_2024-12-31.csv
    ARVE_debit_20250101_20260216.csv
  ofev/
    control_member.csv
  sig_cnr/
```

## Rules for compatibility

- Keep folder names exactly the same as above.
- Keep file names unchanged unless code/config is updated accordingly.
- Use project-relative paths in code (for example, `data/observations/...`).
- If a file is missing, scripts should fail with a clear error message.

## Team workflow

1. Share data outside Git (cloud drive, secure transfer, institutional storage).
2. Each teammate copies files into the local `data/` folder using this layout.
3. Keep code and docs in Git; keep large datasets out of Git.
