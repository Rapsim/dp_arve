# Environment setup


## 1) Create the environment

```powershell
conda env create -f environment.yml
conda activate arve-flow
```

## 2) If you add packages

Install with conda:

```powershell
conda install -n arve-flow -c conda-forge <package1> <package2>
```

If the package is only available on pip:

```powershell
pip install <package>
```

If you installed with pip, add these entries by hand in `environment.yml` (under `dependencies`):

```yaml
  - pip
  - pip:
    - <package>
```

Then refresh `environment.yml`:

```powershell
conda env export --from-history -n arve-flow > environment.yml
```

Commit and push `environment.yml`.

## 3) Update an existing environment from latest file

```powershell
conda env update -n arve-flow -f environment.yml
```

