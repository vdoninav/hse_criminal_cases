import prepare_data
import params

if params.RENEW_SAVED_DATA_IN_PREPROCESSING:
    prepare_data.csv_preprocessing(params.TRAIN_PATH, params.SAVE_DIR+'train.pkl')
    prepare_data.csv_preprocessing(params.TEST_PATH, params.SAVE_DIR+'test.pkl')
    prepare_data.csv_preprocessing(params.VALIDATION_PATH, params.SAVE_DIR+'validation.pkl')