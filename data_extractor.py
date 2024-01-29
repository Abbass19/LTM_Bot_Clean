import pandas as pd
import psycopg2
import psycopg2.extras
from psycopg2.extensions import AsIs
import numpy as np

pd.set_option.max_columns = None

def data_extractor(asset,cols=['DCP','DNCP','OPCP','HPCP','LPCP','CPCP','ACPCP','VTCP','MPN5P']):
    # The credentials to conect to the database
    hostname = 'database-1.ctzm0hf7fhri.eu-central-1.rds.amazonaws.com'
    database = 'dyDATA_new'
    username = 'postgres'
    pwd = 'Proc2023awsrdspostgresql'
    port_id = 5432
    conn = None
    asset_script="SELECT * FROM "+'\"'+"ASSET_"+asset+'\"'+".features_targets_input_view WHERE features_targets_input_view."+'\"'+"cleaned_raw_features_environment_PK"+'\"'+ "= 4"
    # Here we select the active financial asset from the financial asset list table
    try:
        with psycopg2.connect(
            host = hostname,
            dbname = database,
            user = username,
            password = pwd,
            port = port_id
        ) as conn:
            dataframe = pd.read_sql(asset_script,conn)
    except Exception as error:
        conn.close()
        return error
    finally:
        if conn is not None:
            conn.close()
    dataframe = dataframe.filter(regex='|'.join(cols),axis=1)
    
    for i,j in zip(cols,dataframe.columns):
        dataframe.rename(columns={j:i},inplace=True)
    print(dataframe.tail())

    return dataframe


if __name__ == '__main__':
    result = data_extractor(asset='MSFT')