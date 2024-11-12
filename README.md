# tremor2024

1. Download this repo.

1. Download the data from https://dataverse.unc.edu/dataset.xhtml?persistentId=doi:10.15139/S3/KKUMU1

1. Unzip the `csv.zip` directory and put it in your module. Rename it `data` so that it is conformable with the file specification in the `file` column of the `labels.csv` file.

1. Your job is to fill in the empty functions in `process.py`.

1. The existing repo will use the features to evaluate the performance of an SVM with linear kernel, all data, rbf kernel, all data, linear kernel, easy data, and rbf kernel, easy data.

- All Data, Linear
  - F1 Score: 0.886

- All Data, Radial
  - F1 Score: 0.7951

- Easy Data, Linear
  - F1 Score: 0.9966

- Easy Data, Radial
  - F1 Score: 0.9846

