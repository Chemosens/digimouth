# Example data

## `in_vitro_isoamyl_acetate_5min.csv`

Release of isoamyl acetate from an aqueous solution in a closed vial, measured by PTR-MS.
Used by `digimouth_optim_in_vitro_y0_dependency.py` to fit the in vitro convection model.

| Column | Content |
|---|---|
| `time` | time (s), from the start of the recording |
| `intensity` | PTR-MS signal of isoamyl acetate (m/z 131.107) |
| `file` | repetition: `expe_1`, `expe_2`, `expe_3` |

Experimental conditions, identical for the three repetitions (runs `CSDA177-test5min_1` to
`_3`, 7 April 2025):

| | |
|---|---|
| Sample | 2 mL of an aqueous solution of isoamyl acetate at 2 ppm, no agitation |
| Vial | 50 mL |
| Temperature | 22 °C |
| Equilibrium time before measurement | 5 min |
| Sweep gas flow (PTR-MS inlet) | 158 mL/min |
| PTR-MS acquisition | 500 ms per cycle |

The file is an extract (isoamyl acetate only) of the in vitro measurements of the Digimouth
project.
