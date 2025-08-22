import numpy as np
import pandas as pd

from src._utils import main_logger


def handle_feature_engineering(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:

    pd.set_option('future.no_silent_downcasting', True)

    if df is None or not isinstance(df, pd.DataFrame):
        main_logger.error("need pandas dataframe")
        return None

    #### stockcode
    df['stockcode'] = df['stockcode'].replace('m', 'M')
    stockcodes_to_drop = ['B', 'CRUK', 'C2']
    df = df[~df['stockcode'].isin(stockcodes_to_drop)]


    #### remove unnecessary features
    if 'description' in df.columns.tolist(): df = df.drop(columns='description')


    #### adds quantity momentum features
    df['invoicedate'] = pd.to_datetime(df['invoicedate'], errors='coerce')
    df['year'] = df['invoicedate'].dt.year
    df['year_month'] = df['invoicedate'].dt.to_period('M')
    df['month_name'] = df['invoicedate'].dt.strftime('%b')
    df['day_of_week'] = df['invoicedate'].dt.strftime('%a')
    # df['sales'] = df['quantity'] * df['unitprice']

    ### last month
    _temp_df_prod_month_agg = df.copy().groupby(['stockcode', 'year_month']).agg(
        prod_total_monthly_quantity=('quantity', 'sum'),
        prod_ave_monthly_price=('unitprice', 'mean')
    ).reset_index().sort_values(by=['stockcode', 'year_month'])
    _temp_df_prod_month_agg['product_avg_quantity_last_month'] = _temp_df_prod_month_agg.groupby('stockcode')['prod_total_monthly_quantity'].shift(1)
    df_fin = pd.merge(
        df, _temp_df_prod_month_agg[['stockcode', 'year_month', 'product_avg_quantity_last_month']], on=['stockcode', 'year_month'], how='left'
    )
    df_fin['product_avg_quantity_last_month'] = df_fin['product_avg_quantity_last_month'].fillna(value=0)


    #### add unitprice related features
    ### vs max
    _temp_df_max_price = df.groupby('stockcode')['unitprice'].max().reset_index()
    _temp_df_max_price.rename(columns={'unitprice': 'product_max_price_all_time'}, inplace=True)
    df_fin = pd.merge(left=df_fin, right=_temp_df_max_price, on='stockcode', how='left')

    df_fin['unitprice_vs_max'] = df_fin['unitprice'] / df_fin['product_max_price_all_time']
    df_fin.loc[df_fin['product_max_price_all_time'] == 0, 'unitprice_vs_max'] = 0

    ### vs ave
    df_fin['unitprice_to_avg'] = df_fin['unitprice'] / df_fin['product_avg_quantity_last_month']
    df_fin.loc[df_fin['product_avg_quantity_last_month'] == 0, 'unitprice_to_avg'] = 0

    ### squared
    df_fin['unitprice_squared'] = df_fin['unitprice'] ** 2

    ### log
    df_fin['unitprice_log'] = np.log1p(df_fin['unitprice'])


    ####  add customer related features
    df_fin['is_registered'] = np.where(df_fin['customerid'].isna(), 0, 1)
    df_fin['customerid'] = df_fin['customerid'].fillna('unknown').astype('str')


    #### final adjustment
    # drop unnecessary data
    stockcodes_to_drop = ['D', 'S']
    df_fin = df_fin[~df_fin['stockcode'].isin(stockcodes_to_drop)]
    df_fin = df_fin.drop(columns=['month_name'], axis='columns')


    # quantity (drop negative vals)
    df_fin['quantity'] = pd.to_numeric(df_fin['quantity'], errors='coerce')
    df_fin['quantity'] = df_fin['quantity'].fillna(0)
    df_fin = df_fin[df_fin['quantity'] > 0]


    # dtype transformation
    df_fin['year_month'] = df_fin['year_month'].dt.month
    df_fin['invoicedate'] = df_fin['invoicedate'].astype(int) / 10 ** 9


    # imputation
    df_fin['customerid'] = df_fin['customerid'].fillna(value='unknown')
    df_fin['stockcode'] = df_fin['stockcode'].fillna(value='unknown')
    df_fin['invoiceno'] = df_fin['invoiceno'].fillna(value='unknown')



    # imputation (values referred to stockcode)
    df_imputed = df_fin.copy().sort_values(by='stockcode').reset_index(drop=True)
    df_stockcode = df_imputed.groupby('stockcode', as_index=False).agg(
        imputed_country=('country', lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown'),
        imputed_unitprice=('unitprice', 'median')
    )
    df_fin = pd.merge(df_fin, df_stockcode, on='stockcode', how='left')
    df_fin['country'] = df_fin['country'].fillna(df_fin['imputed_country'])

    global_median = df_fin['unitprice'].median()
    df_fin['unitprice'] = df_fin['unitprice'].fillna(df_fin['imputed_unitprice'])
    df_fin['unitprice'] = df_fin['unitprice'].fillna(global_median)
    df_fin = df_fin.drop(columns=['imputed_country', 'imputed_unitprice'])


    # transform sales to logged values
    alpha = 1e-10
    df_fin['quantity'] = np.log1p(df_fin['quantity'] + alpha)

    if verbose: main_logger.info(df_fin.info())

    return df_fin
