# attention lovely teammates

i recommend to structure the data like so (or adjust the config.py, but that will be annoying with git so please don't, lol):

```
.
├── data
│   └── cbis-ddsm
│       ├── test
│       │   ├── images
│       │   └── masks
│       └── train
│           ├── images
│           └── masks
├── meta
│   └── cbis-ddsm
│       ├── calc_case_description_test_set.csv
│       ├── calc_case_description_train_set.csv
│       ├── mass_case_description_test_set.csv
│       └── mass_case_description_train_set.csv
└── *.py and whatever else
```
