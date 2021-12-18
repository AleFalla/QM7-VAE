The main files are:
- 'main_testing_global.py' does testing of three models from randomly grouped 15 global properties to molecule
- 'main_testing_local.py' does testing of three models from 4 different local properties to molecule

In 'separate_models.py' you will find the single modules, while in VAE.py you will find the variational autoencoder implementation plus some different implementations (useless for now)

In 'Train_Test_utils.py' you will find different training and test functions

In 'Utils.py' you will find various functions used in most files

The main_testing folder contains various saved models, plots and logs of the testing results. Check the main_testing files to see where this stuff is saved inside the folders

The rest of the files are just for random tests or useless stuff

The correct way to proceed once you have you .db molecule dataset file is the following:
- use 'db_to_json.py' to turn it into a pandas dataset (check the paths and filenames used in the script)
- use 'splitting.py' to get your training and test datasets and save them to .json format (in order to use the same splitting for all tested models)
- run one of the main_testing files
- go to the jupyter notebook 'Analysis.ipynb' to analyse the results and the models

NOTE: if the file 'dataset41537.json' is already present and you want to use that dataset, then it is enough to start from step 2 of the aforementioned procedure. Moreover always check the paths used in the files as they may differ from the ones you are using