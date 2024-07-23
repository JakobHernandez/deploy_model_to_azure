import mlflow
import mlflow.sklearn
import os
import argparse
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, log_loss
from sklearn.model_selection import train_test_split

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--n_val", required=False, default=20, type=int)
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.25)
    parser.add_argument("--registered_model_name", type=str, help="model name")
    args = parser.parse_args()
   
    # Start Logging
    mlflow.start_run()

    # enable autologging
    mlflow.sklearn.autolog()

    ###################################################################################
    # prepare data

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.data)
    
    raw_data = pd.read_csv(args.data)

    X = raw_data[args.n_val:]

    train_df, test_df = train_test_split(
        X,
        test_size=args.test_train_ratio,
        random_state=112
    )

    y_train = train_df['Status'].copy()
    X_train = train_df.drop(['Status'], axis=1)

    y_test = test_df['Status'].copy()
    X_test = test_df.drop(['Status'], axis=1)

    encode = LabelEncoder()
    y_train = encode.fit_transform(y_train)

    print(f"Training with data of shape {X_train.shape}")

    ###################################################################################
    # train model

    numeric_cols = ['N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin']
    categorical_cols_OH = []
    categorical_cols_OE = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Stage']

    class ETL():
        def __init__(self):
            return None

        def transform(self,X,y=None):
            X['Alk_Phos'] = X['Alk_Phos'].apply(lambda x: min(x, 4000))
            X['Prothrombin'] = X['Prothrombin'].apply(lambda x: min(x, 14))
            X['Stage'] = X['Stage'].astype(object)
            X = X.drop(['id'], axis=1)
            return X

        def fit(self, X, y=None):
            return self
        
    transformer = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat_OE', OrdinalEncoder(), categorical_cols_OE),
            ('cat_OH', OneHotEncoder(), categorical_cols_OH)
        ]
    )

    ETL_pipline = Pipeline(steps=[
                            ('ETL', ETL()),
                            ('transformer', transformer)
                            ])
    
    xgb_params ={'max_depth': 16,
        'learning_rate': 0.07955658827201513,
        'n_estimators': 1318,
        'subsample': 0.7469188198194912,
        'colsample_bytree': 0.023847784550319412,
        'reg_alpha': 4.251195978998175,
        'reg_lambda': 9.143539708234597,
        'random_state': 112,
        'eval_metric': 'mlogloss',
    }

    lgb_params = {
        'metric': 'multi_logloss', 
        'max_depth': 18,
        'min_child_samples': 12,
        'learning_rate': 0.019265722887208857,
        'n_estimators': 561,
        'subsample': 0.58120988528067,
        'min_child_weight': 5.3032983255619115,
        'colsample_bytree': 0.15297945002747868,
        'reg_alpha': 0.513877346112968,
        'reg_lambda': 0.26119096957757604, 
        'random_state': 112,
        'verbosity' : -1
    }

    lgbm_model = LGBMClassifier(**lgb_params)

    xgb_model = XGBClassifier(**xgb_params)

    Ensemble = VotingClassifier(estimators = [('lgb', lgbm_model), ('xgb', xgb_model)], 
                                voting='soft',
                                weights = [
                                    0.5,
                                    0.5]
                                )
    
    pipeline = Pipeline(steps=[
                            ('ETL_pipline', ETL_pipline),
                            ('model', Ensemble)
                            ])
    
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_test = encode.transform(y_test)

    print(classification_report(y_test, y_pred))

    ###################################################################################
    # Register model
    
    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        registered_model_name=args.registered_model_name,
        artifact_path=args.registered_model_name,
    )

    # Saving the model to a file
    mlflow.sklearn.save_model(
        sk_model=pipeline,
        path=os.path.join(args.registered_model_name, "trained_model"),
    )
    
    # Stop Logging
    mlflow.end_run()

if __name__ == "__main__":
    main()
