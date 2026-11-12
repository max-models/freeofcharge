# CoilStellaration sample

One benchmark instance of [CoilStellaration](https://huggingface.co/datasets/proxima-fusion/coilstellaration),
copied from plasma-plots' `scripts/data` (fetched there by `scripts/fetch_coilstellaration_sample.py`):
the first instance, by boundary ID, of the `eval` split in the `fixed_shape` track (3 field
periods, 5 coils per half period, `tight` stratum).

- `coilstellaration_coilset.json`: its DESC-optimized coilset (CoilStellaration's layout);
- `coilstellaration_boundary.json`: its plasma boundary from
  [ConStellaration](https://huggingface.co/datasets/proxima-fusion/constellaration);
- `coilstellaration_sample.json`: the IDs and the dataset's own metrics.

Both datasets are MIT-licensed, Copyright (c) Proxima Fusion GmbH.
