import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from datetime import datetime, timedelta, timezone
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.abspath('/Users/supriyasindigerekumaraswmamy/Desktop/Thesis/wind_Turbine'))
from utils.helper import signals_aggregation_rules, metamast_aggregation_rules


def get_signals_with_low_variance(df: pd.DataFrame, threshold=0) -> list:
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    cont_data = df.select_dtypes(include=numerics)
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(cont_data)
    inverted_list = ~np.array(selector.get_support())
    return cont_data.columns[inverted_list].tolist()

def signal_preprocess():
    signals_2016 = pd.read_csv('./data/init/signals-2016.csv', sep=';')
    signals_2017 = pd.read_csv('./data/init/signals-2017.csv', sep=';')
    signals = pd.concat([signals_2016, signals_2017], axis=0)
    signals = signals.set_index(pd.to_datetime(signals['Timestamp']))
    signals = signals.drop(columns = get_signals_with_low_variance(signals, 0))
    agg_signals = signals.groupby('Turbine_ID').resample('D').agg(signals_aggregation_rules)
    agg_signals['Turbine_ID'] = agg_signals.index.get_level_values('Turbine_ID')  
    agg_signals = agg_signals.reset_index('Timestamp')
    agg_signals['signal_missing_values'] = agg_signals.isnull().any(axis=1).astype(int)
    agg_signals = agg_signals.bfill().reset_index(drop=True)
    return agg_signals

def metacast_preprocess():
    metmast_2016 = pd.read_csv('./data/init/metmast-2016.csv',sep=';')
    metmast_2017 = pd.read_csv('./data/init/metmast-2017.csv',sep=';')
    metmast = pd.concat([metmast_2016, metmast_2017], axis=0)
    metmast = metmast.set_index(pd.to_datetime(metmast['Timestamp']))
    metmast = metmast.drop(columns = get_signals_with_low_variance(metmast, 0) + ["Min_Winddirection2", "Max_Winddirection2", "Avg_Winddirection2", "Var_Winddirection2"])
    agg_metmast=metmast.resample('D').agg(metamast_aggregation_rules).reset_index()
    agg_metmast['metacast_missing_values'] = agg_metmast.isnull().any(axis=1).astype(int)
    agg_metmast = agg_metmast.bfill().reset_index(drop=True)
    return agg_metmast

def merge_signals_metmast(signals: pd.DataFrame, metmast: pd.DataFrame) -> pd.DataFrame:
    merged_df = pd.merge(signals, metmast, on="Timestamp", how="left")
    merged_df = merged_df[merged_df["signal_missing_values"] == 0]
    merged_df = merged_df[merged_df["metacast_missing_values"] == 0 ]
    merged_df.drop(columns=["signal_missing_values", "metacast_missing_values"], inplace=True)
    column_description = pd.read_excel('./data/init/column description.xlsx')
    column_description['Expanded_column_names'] = column_description['Expanded_column_names'].str.replace(' ', '_').str.lower()
    merged_df.columns = column_description.set_index('Columns').loc[merged_df.columns, 'Expanded_column_names'].values
    return merged_df

def create_failure_list(failures: pd.DataFrame, days_lookback: int, value_function, target_name: str = "Target") -> pd.DataFrame:
    failure_list = []

    for i in range(len(failures)):
        turbine_id = str(failures.iloc[i]["turbine_id"])
        failure_datetime = failures.iloc[i]["timestamp"]
        components = failures.iloc[i]["component"]
        rounded_datetime = failure_datetime.replace(hour=0, minute=0, second=0, microsecond=0)  # Round to the start of the day

        for j in range(days_lookback):
            delta = timedelta(days=j)
            new_datetime = rounded_datetime - delta
            datetime_formated = new_datetime.replace(tzinfo=timezone.utc)
            # Calculate the target value using the provided value_function
            target_value = value_function(j, days_lookback)
            failure_list.append([turbine_id, datetime_formated.isoformat(), components,target_value])
    
    failure_df = pd.DataFrame(failure_list, columns=["turbine_id", "timestamp", "component",target_name])
    return failure_df

def failure_preprocess():
    failure_2016 = pd.read_csv('./data/init/failures-2016.csv',sep=';')
    failure_2017 = pd.read_csv('./data/init/failures-2017.csv',sep=';')
    failures=pd.concat([failure_2016, failure_2017], axis=0)
    failures['Timestamp'] = pd.to_datetime(failures['Timestamp'])
    failures['Timestamp'] = failures['Timestamp'].dt.floor('d')
    failures.columns = ['turbine_id', 'component', 'timestamp','remarks']
    class_target_name = "target_class"
    classif_function = lambda i, j: 1
    days_lookback = 60
    components = failures["component"].unique()
    for component in components:
        globals()[f"failure_df_{component}"] = create_failure_list(failures[failures["component"] == component], days_lookback, classif_function, class_target_name)
        globals()[f"failure_df_{component}"]['timestamp'] = pd.to_datetime(globals()[f"failure_df_{component}"]['timestamp'])
        globals()[f"labeled_df_{component}"] = pd.merge(merged_df.reset_index(drop=True), globals()[f"failure_df_{component}"].reset_index(drop=True), on=["turbine_id", "timestamp"], how="left")
        globals()[f"labeled_df_{component}"][class_target_name] = globals()[f"labeled_df_{component}"][class_target_name].fillna(0).astype(int)
        globals()[f"labeled_df_{component}"].drop_duplicates(inplace=True)

    return {f"labeled_df_{component}": globals()[f"labeled_df_{component}"] for component in components}

if __name__ == "__main__":
    signal = signal_preprocess()
    
    metacast = metacast_preprocess()
    merged_df = merge_signals_metmast(signal, metacast)
    failure = failure_preprocess()
    os.makedirs('./data/model_data', exist_ok=True)
    for key, value in failure.items():
        value.to_csv(f'./data/model_data/{key}.csv', index=False)
        
    
    
