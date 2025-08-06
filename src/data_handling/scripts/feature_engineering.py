import numpy as np
import pandas as pd

from src._utils import main_logger

obj_cols = ['invoiceno', 'stockcode', 'country', 'invoicedate', 'description']



def structure_missing_values(df: pd.DataFrame, target_cols: list = [], verbose: bool = False) -> pd.DataFrame:
    pd.set_option('future.no_silent_downcasting', True)

    if df is None or not isinstance(df, pd.DataFrame):
        main_logger.error("Need a Panda DataFrame") 
        return None
        
    target_cols = target_cols if target_cols else df.columns.tolist()

    df_processed = df.copy()
    null_vals = ['', 'nan', 'N/A', None, 'na', 'None', 'none']
    replaced_null = { item: np.nan for item in null_vals }

    for i, col in enumerate(target_cols):
        df_processed[col].replace(replaced_null)
        unique_num = df_processed[col].nunique()
        nan_num = df_processed[col].isna().sum()
        if verbose: main_logger.info(f'{i}. {col} - {unique_num:,} data points (missing data: {nan_num:,})')

    return df_processed




def handle_feature_engineering(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Handles various feature engineering (addition, deletion, replacement, dtype adjustment)
    """

    pd.set_option('future.no_silent_downcasting', True)

    if df is None or not isinstance(df, pd.DataFrame): 
        main_logger.error("need pandas dataframe") 
        return None


    df_processed = df.copy()

    # remove unnecessary features
    if 'description' in df_processed.columns.tolist(): df_processed = df_processed.drop(columns='description') 

    # add target col: sales (later takes log)
    df_processed['sales'] = df_processed['quantity'] * df_processed['unitprice']

    # adds sales momentum features
    df_processed['invoicedate'] = pd.to_datetime(df_processed['invoicedate'], errors='coerce')
    df_processed['year'] = df_processed['invoicedate'].dt.year
    df_processed['year_month'] = df_processed['invoicedate'].dt.to_period('M')
    df_processed['month_name'] = df_processed['invoicedate'].dt.strftime('%b')
    df_processed['day_of_week'] = df_processed['invoicedate'].dt.strftime('%a')

    _df_prod_month_agg = df_processed.copy().groupby(['stockcode', 'year_month']).agg(
        prod_total_monthly_quantity=('quantity', 'sum'),
        prod_ave_monthly_price=('unitprice', 'mean')
    ).reset_index().sort_values(by=['stockcode', 'year_month'])
    _df_prod_month_agg['product_avg_quantity_last_month'] = _df_prod_month_agg.groupby('stockcode')['prod_total_monthly_quantity'].shift(1)
    _df_prod_last_month_agg = _df_prod_month_agg.groupby('stockcode')['product_avg_quantity_last_month'].mean().reset_index()
    _df_prod_last_month_agg_renamed = _df_prod_last_month_agg.rename(
        columns={'product_avg_quantity_last_month': 'new_product_avg_quantity_last_month'}
    )

    df_fin = pd.merge(
        df_processed,
        _df_prod_last_month_agg_renamed[['stockcode', 'new_product_avg_quantity_last_month']],
        on='stockcode',
        how='left'
    )
    df_fin['product_avg_quantity_last_month'] = df_fin['new_product_avg_quantity_last_month']
    df_fin = df_fin.drop(columns='new_product_avg_quantity_last_month', axis=1)
    df_fin['product_avg_quantity_last_month'] = df_fin['product_avg_quantity_last_month'].fillna(value=0)

    # engineers customer based features
    ## 1. customer_recency_days
    df_processed['is_registered'] = np.where(df_processed['customerid'].isna(), 0, 1)
    df_processed['customerid'] = df_processed['customerid'].fillna('no_id').astype('str')

    _df_all_customers_year_month = pd.MultiIndex.from_product(
        [df_fin['customerid'].unique(), df_fin['year_month'].unique()], # type: ignore
        names=['customerid', 'year_month']
    ).to_frame(index=False).sort_values(by=['customerid', 'year_month']).reset_index(drop=True)
    _df_customer_monthly_agg = df_fin.copy().groupby(['customerid', 'year_month']).agg(
        monthly_sales=('sales', 'sum'),
        monthly_unique_invoices=('invoiceno', 'nunique'),
        monthly_last_purchase_date=('invoicedate', 'max')
    ).reset_index()
    _df_cus = _df_all_customers_year_month.merge(_df_customer_monthly_agg, on=['customerid', 'year_month'], how='left').sort_values(by=['customerid', 'year_month'])

    _df_cus['pfin_last_purchase_date'] = _df_cus.groupby('customerid')['monthly_last_purchase_date'].shift(1)
    _df_cus['invoice_timestamp_end'] = _df_cus['year_month'].dt.end_time
    _df_cus['customer_recency_days'] = (_df_cus['invoice_timestamp_end'] - _df_cus['pfin_last_purchase_date']).dt.days
    df_fin['customer_recency_days'] = _df_cus['customer_recency_days']
    max_recency = _df_cus['customer_recency_days'].max()
    df_fin['customer_recency_days'] = df_fin['customer_recency_days'].fillna(value=max_recency + 30)
    df_fin['customer_recency_days'] = df_fin['customer_recency_days'].fillna(365)

    ## 2. customer_total_spend_ltm
    if not _df_cus['customerid'].isna().all():
        _df_cus['customer_total_spend_ltm'] = _df_cus.groupby('customerid')['monthly_sales'].rolling(window=3, closed='left').sum().reset_index(level=0, drop=True)
        df_fin['customer_total_spend_ltm'] = _df_cus['customer_total_spend_ltm']
        df_fin['customer_total_spend_ltm'] = df_fin['customer_total_spend_ltm'].fillna(value=0)

        ## 3. customer_freq_ltm
        _df_cus['customer_freq_ltm'] = _df_cus.groupby('customerid')['monthly_unique_invoices'].rolling(window=3, closed='left').sum().reset_index(level=0, drop=True)
        df_fin['customer_freq_ltm'] = _df_cus['customer_freq_ltm']
        df_fin['customer_freq_ltm'] = df_fin['customer_freq_ltm'].fillna(value=0)
    else:
        df_fin['customer_freq_ltm'] = 0
        df_fin['customer_total_spend_ltm'] = 0


    # imputation & data type transformation
    df_fin['customerid'] = df_fin['customerid'].fillna(value='unknown')
    df_fin['stockcode'] = df_fin['stockcode'].fillna(value='unknown')
    df_fin['invoiceno'] = df_fin['invoiceno'].fillna(value='unknown')
    df_fin['quantity'] = df_fin['quantity'].fillna(value=np.floor(df_fin['sales'] / df_fin['unitprice'])) # type: ignore
    df_fin['country'] = df_fin['country'].fillna(value=df_fin['country'].mode().iloc[0])
    df_fin['year_month'] = df_fin['year_month'].dt.month
    df_fin['invoicedate'] = df_fin['invoicedate'].astype(int) / 10 ** 9
    df_fin = df_fin.drop(columns=['month_name'], axis='columns')

    # handle return
    df_fin_log = df_fin.copy()
    df_fin_log['is_return'] = (df_fin_log['sales'] < 0).astype(int)
    df_fin_log['quantity'] = np.where(df_fin_log['is_return'] == 1, 0, df_fin_log['quantity'])

    # transform quantity to logged values
    alpha = 1
    df_fin_log['quantity'] = np.log(df_fin_log['quantity'] + alpha)

    # drop sales column
    df_fin_log = df_fin.drop(columns=['sales'], axis='columns')
    

    # df_fin_log['sales'] = df_fin_log['sales'].apply(lambda x: max(x, 0))

    # # transform sales to logged sales
    # alpha = 1
    # df_fin_log['sales'] = np.log(df_fin_log['sales'] + alpha)
    
    if verbose: main_logger.info(df_fin.info())
    return df_fin_log