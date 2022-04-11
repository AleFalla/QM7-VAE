The main files are:
- 'main.py' trains and tests VAE + property->molecule prediction for 17 global properties
- 'cloob_train.py' does training and testing of CLOOB (contrastive) with atomic properties followed by training encoder/decoder on the contrastively obtained latent space (encoding global properties this time)

In CLIP_utils.py ignore the inner_gaussian function, is a work in progress

In 'separate_models.py' you will find the single modules, while in VAE.py you will find the variational autoencoder implementation plus some different implementations (useless for now)

In 'Train_Test_utils.py' you will find different training and test functions, should be undestadable but just in case look at how they are used in the main.py and cloob_train.py files

In 'Utils.py' you will find various functions used in most files

For now there are no saved models but it is straightforward to implement and will be done once the methodology is fixed

The rest of the files are just for random tests or useless stuff

The correct way to proceed once you have you .db molecule dataset file is the following:
- use 'db_to_json.py' to turn it into a pandas dataset (check the paths and filenames used in the script)
- use the functions in Utils.py to generate a molecular representation (see how it is done in main.py)
- use 'splitting.py' to get your training and test datasets. You can either save them to .json format (in order to use the same splitting for all tested models) or fix a seed and call it online
- run one main files

NOTE: if the file 'dataset41537.json' is already present and you want to use that dataset, then it is enough to start from step 2 of the aforementioned procedure. Moreover always check the paths used in the files as they may differ from the ones you are using