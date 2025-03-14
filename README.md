[![rtd](https://readthedocs.org/projects/fixms/badge/?version=latest)](https://fixms.readthedocs.io/)
[![Tests](https://github.com/AlecThomson/FixMS/actions/workflows/test.yml/badge.svg)](https://github.com/AlecThomson/FixMS/actions/workflows/test.yml)
[![PyPi](https://github.com/AlecThomson/FixMS/actions/workflows/publish.yml/badge.svg)](https://pypi.org/project/fixms/)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/AlecThomson/FixMS/main.svg)](https://results.pre-commit.ci/latest/github/AlecThomson/FixMS/main)
[![codecov](https://codecov.io/gh/AlecThomson/FixMS/graph/badge.svg?token=08GDOB2DM0)](https://codecov.io/gh/AlecThomson/FixMS)

# FixMS

WARNING FOR USERS: These scripts are still being verified for complete
correctness. Please use at your own risk.

ASKAP utilities for updating MeasurementSets for external imagers.

ASKAP MSs are produced in a way that breaks compatibility with most other
imagers (e.g. CASA, WSclean). Here we provide two modules (with CLI hooks) that
perform the fixes that need to be applied in order to produce astronomically
correct imagers with non-YandaSoft imagers:

1. `fix_ms_dir` : ASKAP MeasurementSets are phased towards the centre of field,
   but not the centre of its given beam. This utility reads the appropriate
   offsets to the beam centre from the `BEAM_OFFSET` and updates the `FIELD`
   table, as well as the phase and delay reference columns. An option is also
   available to restore the `FIELD` table directions to the original directions
   if `fix_ms_dir` has already been run. `BEAM_OFFSET` is also restored with
   this option.

2. `fix_ms_corrs` : ASKAP MeasurementSets, as calibrated by the obervatory,
   provide correlations in the instrument frame. ASKAP has a unique 'roll' axis
   which means, in principle, the instrument frame can be at any arbitrary
   rotation on the sky. This utility applies the appropriate rotation matrix to
   the visibilities such the 'X' is aligned North-South and 'Y' is aligned
   East-West (IAU convention). Further, ASKAPsoft defines Stokes I as $I=XX+YY$,
   whereas most other telescopes use $I=\frac{1}{2}(XX+YY)$ (note this also
   applies to all other Stokes parameters). This factor is also corrected for
   here at the same time as the rotation. If you have calibrated with
   non-ASKAPsoft tools, you may need to use the `--no-fix-stokes-factor` option,
   which will disable the factor of two correction and just do a rotation.

For convenience, we also provide `fix_ms` which does both of the above!

Full documentation on [Read The Docs](https://fixms.readthedocs.io/en/latest/).

## Installation

Obtain and install Python 3 (I recommend
[Miniforge](https://github.com/conda-forge/miniforge) or
[Miniconda](https://docs.conda.io/en/latest/miniconda.html)).

Install the Python scripts:

Latest:

```
pip install git+https://github.com/AlecThomson/FixMS.git
```

Stable:

```
pip install FixMS
```

## Usage

> **_NOTE:_** If running on large MeasurementSets pay attention to the
> 'chunksize' and 'max-chunks' arguments. These are used to balance memory usage
> and the disk I/O. You may need to adjust these parameters to achieve maximum
> speed on your hardware.

```
❯ fix_ms -h
usage: fix_ms [-h] [--chunksize CHUNKSIZE] [--max-chunks MAX_CHUNKS] [--data-column DATA_COLUMN] [--corrected-data-column CORRECTED_DATA_COLUMN] [--no-fix-stokes-factor] [ms]

Utility to correct the ASKAP beam positions and apply a rotation to apply a change of the reference frame of the visibilities

positional arguments:
  ms                    Measurement set to update (default: None)

optional arguments:
  -h, --help            show this help message and exit
  --chunksize CHUNKSIZE
                        The chunksize to use when reading the MS (default: 1000)
  --max-chunks MAX_CHUNKS
                        The maximum number of chunks to process at once (default: 1000)
  --data-column DATA_COLUMN
                        The column to fix (default: DATA)
  --corrected-data-column CORRECTED_DATA_COLUMN
                        The column to write the corrected data to (default: CORRECTED_DATA)
  --no-fix-stokes-factor
                        Don't fix the Stokes factor. Use this if you have *not* used ASKAPsoft. If you have used ASKAPsoft, you should leave this option alone. (default: False)
```

```
❯ fix_ms_corrs -h
usage: fix_ms_corrs [-h] [--chunksize CHUNKSIZE] [--max-chunks MAX_CHUNKS] [--data-column DATA_COLUMN] [--corrected-data-column CORRECTED_DATA_COLUMN] [--no-fix-stokes-factor] ms

Fix the correlation rotation of ASKAP MSs. Converts the ASKAP standard correlations to the 'standard' correlations This will make them compatible with most imagers (e.g. wsclean, CASA) The
new correlations are placed in a new column called 'CORRECTED_DATA'

positional arguments:
  ms                    The MS to fix

optional arguments:
  -h, --help            show this help message and exit
  --chunksize CHUNKSIZE
                        The chunksize to use when reading the MS (default: 1000)
  --max-chunks MAX_CHUNKS
                        The maximum number of chunks to process at once (default: 1000)
  --data-column DATA_COLUMN
                        The column to fix (default: DATA)
  --corrected-data-column CORRECTED_DATA_COLUMN
                        The column to write the corrected data to (default: CORRECTED_DATA)
  --no-fix-stokes-factor
                        Don't fix the Stokes factor. Use this if you have *not* used ASKAPsoft. If you have used ASKAPsoft, you should leave this option alone. (default: False)
```

```
❯ fix_ms_dir -h
usage: fix_ms_dir [-h] [ms]

ASKAP utility - update the pointing centre of a beam in an MS. - Allows imaging by CASA or wsclean.

positional arguments:
  ms             Measurement set to update (default: None)

options:
  -h, --help     show this help message and exit
  -r, --restore  Switch to restore direction to the original ASKAPsoft pipeline direction. (default: False)
```

## Contribution

Contributions are very welcome! Please open an issue first to discuss any bugs
or updates you might have.
