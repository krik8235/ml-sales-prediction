# import numpy as np
# import pandas as pd
# import lightgbm as lgb # type: ignore
# import torch
# from sklearn.linear_model import ElasticNet
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# import joblib
# import pickle

# import src.model.torch_model as t
# import src.model.sklearn_model as sk
# from src._utils import main_logger, MODEL_SAVE_PATH


# df_base = pd.DataFrame({
#     'stockcode': ['85123A', '85123A', 'PROD002', '85123A', 'PROD002'],
#     'unitprice': [10.0, 11.0, 5.0, 10.5, 5.5],
#     'quantity': [100, 110, 50, 105, 55],
#     'invoicedate': pd.to_datetime(['2024-01-01', '2024-01-15', '2024-01-10', '2024-02-01', '2024-02-15']),
#     'country': ['USA', 'USA', 'UK', 'USA', 'UK'],
#     'customerid': [1, 2, 3, 4, 5]
# })
# num_cols = ['unitprice', 'quantity']
# cat_cols = ['stockcode', 'country']

# def structure_missing_values(df): return df
# def handle_feature_engineering(df):
#     df['year'] = df['invoicedate'].dt.year
#     df['month'] = df['invoicedate'].dt.month
#     return df
# def make_train_val_datasets(df):
#     X = df[num_cols + cat_cols].fillna(0)
#     y = np.log(df['quantity'])
#     return X, X, X, y, y, y

# numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
# categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, num_cols),
#         ('cat', categorical_transformer, cat_cols)
#     ])
# preprocessor.fit(df_base[num_cols + cat_cols])
# joblib.dump(preprocessor, MODEL_SAVE_PATH + 'preprocessor.pkl')
# s3_client.upload_file(MODEL_SAVE_PATH + 'preprocessor.pkl', S3_BUCKET_NAME, f"{S3_MODEL_PREFIX}preprocessor.pkl")
# with open(MODEL_SAVE_PATH + 'df_base.pkl', 'wb') as f:
#     pickle.dump(df_base, f)
# s3_client.upload_file(MODEL_SAVE_PATH + 'df_base.pkl', S3_BUCKET_NAME, f"{S3_MODEL_PREFIX}df_base.pkl")


# def transform_input(X_train, X_val, X_test, num_cols, cat_cols):
#     loaded_preprocessor = joblib.load(MODEL_SAVE_PATH + 'preprocessor.pkl')
#     return (loaded_preprocessor.transform(X_train[num_cols + cat_cols]),
#             loaded_preprocessor.transform(X_val[num_cols + cat_cols]),
#             loaded_preprocessor.transform(X_test[num_cols + cat_cols]))

# YOUR_EXPECTED_INPUT_DIM = preprocessor.transform(df_base[num_cols + cat_cols]).shape[1]


# def retrain_models(new_df_base: pd.DataFrame):
#     """Returns retrained models with new training dataset."""

#     main_logger.info("Starting model retraining process...")

#     df_base_new = structure_missing_values(df=new_df_base.copy())
#     df_new = handle_feature_engineering(df=df_base_new)

#     X_train, X_val, X_test, y_train, y_val, y_test = make_train_val_datasets(df=df_new)
#     X_train, X_val, X_test = transform_input(
#         X_train=X_train, X_val=X_val, X_test=X_test, num_cols=num_cols, cat_cols=cat_cols
#     )
#     X_train = np.nan_to_num(X_train, nan=0.0)
#     y_train = np.nan_to_num(y_train, nan=0.0)

#     processed_input_tensor = torch.tensor(X_train, dtype=torch.float32)
#     processed_target_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

#     main_logger.info("\n--- Retraining Deep Learning Model ---")

#     try:
#         best_dfn = t.main_script(X_train, X_val, X_test, y_train, y_val, y_test)
#         main_logger.info("DFN retraining complete.")

#     except:

#     # current_dl_model = t.load_model(input_dim=processed_input_final.shape[1], trig='best')
    
#     # optimizer = optim.Adam(current_dl_model.parameters(), lr=0.001)
#     # criterion = nn.MSELoss()

#     # num_epochs = 10
#     # for epoch in range(num_epochs):
#     #     current_dl_model.train()
#     #     optimizer.zero_grad()
#     #     outputs = current_dl_model(processed_input_tensor)
#     #     loss = criterion(outputs, processed_target_tensor)
#     #     loss.backward()
#     #     optimizer.step()
#     #     main_logger.info(f"DL Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

#     # t.save_model(model=current_dl_model, model_name='dfn')
    

#         main_logger.info("\n--- Retraining LightGBM Model ---")
#         try:
#             current_gbm, best_hparams = sk.load_model(model_name='gbm', trig='best')
#             if not current_gbm:
#                 main_logger.info("No LightGBM model loaded or not an LGBMRegressor. Initializing a new one.")
#                 best_hparams = best_hparams if best_hparams else dict(
#                     boosting_type='gbdt',
#                     num_leaves=500,
#                     max_depth=15,
#                     min_child_sample=5,
#                     min_child_weight=10.0,
#                     learning_rate=0.3,
#                     n_estimators=100,
#                     subsample=0.6,
#                     subsample_freq=1,
#                     colsample_bytree=1.0,
#                     reg_alpha=1e-05,
#                     reg_lambda=10.0,
#                     random_state=100,
#                     n_job=-1
#                 )
#                 current_gbm = lgb.LGBMRegressor(**best_hparams)

#             current_gbm.fit(X_train, y_train)
#             sk.save_model(model=current_gbm, model_name='gbm', hparams=best_hparams) # type: ignore
#             main_logger.info("LightGBM Model retraining complete.")
        
#         except:
#             main_logger.info("\n--- Retraining Elastic Net Model ---")
#             elastic_net_model = ElasticNet(
#                 alpha=0.00010076137476187376,
#                 l1_ratio=0.6001201735703293,
#                 max_iter=48953,
#                 tol=1.8200116325906476e-05,
#                 selection='random',
#                 fit_intercept=True,
#                 random_state=96
#             )
#             elastic_net_model.fit(X_train, y_train)
#             main_logger.info("Elastic Net Model retraining complete.")

#     main_logger.info("\nAll models retraining complete. New models saved.")

# if __name__ == "__main__":
#     main_logger.info("Running dummy retraining process...")

#     new_data = pd.DataFrame({
#         'stockcode': ['85123A', 'PROD003'],
#         'unitprice': [10.2, 7.5],
#         'quantity': [115, 80],
#         'invoicedate': pd.to_datetime(['2024-03-01', '2024-03-05']),
#         'country': ['USA', 'CAN'],
#         'customerid': [6, 7]
#     })
#     updated_df_base = pd.concat([df_base, new_data], ignore_index=True)

#     retrain_models(updated_df_base)
#     main_logger.info("\nDemonstration of retraining complete.")