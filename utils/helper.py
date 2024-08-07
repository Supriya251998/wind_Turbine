import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
from sklearn.feature_selection import SelectFromModel

# load the failures data
signals_aggregation_rules = {
    'Gen_RPM_Max': 'max',
    'Gen_RPM_Min': 'min',
    'Gen_RPM_Avg': 'mean',
    'Gen_RPM_Std': 'mean',
    'Gen_Bear_Temp_Avg': 'mean',
    'Gen_Phase1_Temp_Avg': 'mean',
    'Gen_Phase2_Temp_Avg': 'mean',
    'Gen_Phase3_Temp_Avg': 'mean',
    'Hyd_Oil_Temp_Avg': 'mean',
    'Gear_Oil_Temp_Avg': 'mean',
    'Gear_Bear_Temp_Avg': 'mean',
    'Nac_Temp_Avg': 'mean',
    'Rtr_RPM_Max': 'max',
    'Rtr_RPM_Min': 'min',
    'Rtr_RPM_Avg': 'mean',
    'Amb_WindSpeed_Max': 'max',
    'Amb_WindSpeed_Min': 'min',
    'Amb_WindSpeed_Avg': 'mean',
    'Amb_WindSpeed_Std': 'mean',
    'Amb_WindDir_Relative_Avg': 'mean',
    'Amb_WindDir_Abs_Avg': 'mean',
    'Amb_Temp_Avg': 'mean',
    'Prod_LatestAvg_ActPwrGen0': 'mean',
    'Prod_LatestAvg_ActPwrGen1': 'mean',
    #'Prod_LatestAvg_ActPwrGen2': 'mean',
    'Prod_LatestAvg_TotActPwr': 'sum',
    'Prod_LatestAvg_ReactPwrGen0': 'mean',
    'Prod_LatestAvg_TotReactPwr': 'sum',
    'HVTrafo_Phase1_Temp_Avg': 'mean',
    'HVTrafo_Phase2_Temp_Avg': 'mean',
    'HVTrafo_Phase3_Temp_Avg': 'mean',
    'Grd_InverterPhase1_Temp_Avg': 'mean',
    'Cont_Top_Temp_Avg': 'mean',
    'Cont_Hub_Temp_Avg': 'mean',
    'Cont_VCP_Temp_Avg': 'mean',
    'Gen_SlipRing_Temp_Avg': 'mean',
    'Spin_Temp_Avg': 'mean',
    'Blds_PitchAngle_Min': 'min',
    'Blds_PitchAngle_Max': 'max',
    'Blds_PitchAngle_Avg': 'mean',
    'Blds_PitchAngle_Std': 'mean',
    'Cont_VCP_ChokcoilTemp_Avg': 'mean',
    'Grd_RtrInvPhase1_Temp_Avg': 'mean',
    'Grd_RtrInvPhase2_Temp_Avg': 'mean',
    'Grd_RtrInvPhase3_Temp_Avg': 'mean',
    'Cont_VCP_WtrTemp_Avg': 'mean',
    'Grd_Prod_Pwr_Avg': 'mean',
    'Grd_Prod_CosPhi_Avg': 'mean',
    'Grd_Prod_Freq_Avg': 'mean',
    'Grd_Prod_VoltPhse1_Avg': 'mean',
    'Grd_Prod_VoltPhse2_Avg': 'mean',
    'Grd_Prod_VoltPhse3_Avg': 'mean',
    'Grd_Prod_CurPhse1_Avg': 'mean',
    'Grd_Prod_CurPhse2_Avg': 'mean',
    'Grd_Prod_CurPhse3_Avg': 'mean',
    'Grd_Prod_Pwr_Max': 'max',
    'Grd_Prod_Pwr_Min': 'min',
    'Grd_Busbar_Temp_Avg': 'mean',
    'Rtr_RPM_Std': 'mean',
    'Amb_WindSpeed_Est_Avg': 'mean',
    'Grd_Prod_Pwr_Std': 'mean',
    'Grd_Prod_ReactPwr_Avg': 'mean',
    'Grd_Prod_ReactPwr_Max': 'max',
    'Grd_Prod_ReactPwr_Min': 'min',
    'Grd_Prod_ReactPwr_Std': 'mean',
    'Grd_Prod_PsblePwr_Avg': 'mean',
    'Grd_Prod_PsblePwr_Max': 'max',
    'Grd_Prod_PsblePwr_Min': 'min',
    'Grd_Prod_PsblePwr_Std': 'mean',
    'Grd_Prod_PsbleInd_Avg': 'mean',
    'Grd_Prod_PsbleInd_Max': 'max',
    'Grd_Prod_PsbleInd_Min': 'min',
    'Grd_Prod_PsbleInd_Std': 'mean',
    'Grd_Prod_PsbleCap_Avg': 'mean',
    'Grd_Prod_PsbleCap_Max': 'max',
    'Grd_Prod_PsbleCap_Min': 'min',
    'Grd_Prod_PsbleCap_Std': 'mean',
    'Gen_Bear2_Temp_Avg': 'mean',
    'Nac_Direction_Avg': 'mean'
}

metamast_aggregation_rules = {
    'Min_Windspeed1': 'min',
    'Max_Windspeed1': 'max',
    'Avg_Windspeed1': 'mean',
    'Var_Windspeed1': 'mean',
    'Min_Windspeed2': 'min',
    'Max_Windspeed2': 'max',
    'Avg_Windspeed2': 'mean',
    'Var_Windspeed2': 'mean',
    'Min_AmbientTemp': 'min',
    'Max_AmbientTemp': 'max',
    'Avg_AmbientTemp': 'mean',
    'Min_Pressure': 'min',
    'Max_Pressure': 'max',
    'Avg_Pressure': 'mean',
    'Min_Humidity': 'min',
    'Max_Humidity': 'max',
    'Avg_Humidity': 'mean',
    'Min_Precipitation': 'min',
    'Max_Precipitation': 'max',
    'Avg_Precipitation': 'mean',
    'Max_Raindetection': 'max',
    'Anemometer1_Avg_Freq': 'mean',
    'Anemometer2_Avg_Freq': 'mean',
    'Pressure_Avg_Freq': 'mean',
}
def load_failures_data(failures_path):
    return pd.read_csv(failures_path, sep=',')

def load_all_component_data(components):
    encoder = LabelEncoder()
    component_data = {}
    for component in components:
        df = pd.read_csv(f'../data/model_data/labelled_data_{component}.csv', sep=',')
        df['turbine_id'] = encoder.fit_transform(['turbine_id'] * df.shape[0])
        df = df.set_index('timestamp')
        component_data[component] = df
    return component_data

def prepare_all_data_for_training(component_data, class_target_name):
    data_splits = {}
    for component, df in component_data.items():
        X = df.drop(columns=['component', class_target_name])
        y = df[class_target_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        # return all X_train, X_test, y_train, y_test
        data_splits[component] = (X_train, X_test, y_train, y_test)
    return data_splits

def load_all_models(components, model_name):
    models = {}
    for component in components:
        with open(f'../models/selected-{component}.pickle', 'rb') as file:
            models[component] = pickle.load(file)
    return models

def fit_and_select_features(models, data_splits):
    selected_features_data = {}
    for component, model in models.items():
        X_train, X_test, y_train, y_test = data_splits[component]
        model.fit(X_train, y_train)
        params = model.get_params()
        selector = SelectFromModel(model, threshold=-np.inf, prefit=True, max_features=params.get('max_features', None))
        selected_features = X_train.columns[selector.get_support()]
        selected_features_train = X_train[selected_features]
        selected_features_train.reset_index(drop=True, inplace=True)
        selected_features_test = X_test[selected_features]
        selected_features_test.reset_index(drop=True, inplace=True)
        selected_features_data[component] = (selected_features_train, selected_features_test, selected_features)
    return selected_features_data

def retrain_models_on_selected_features(models, selected_features_data, data_splits):
    for component, model in models.items():
        selected_features_train, _, _ = selected_features_data[component]
        _, _, y_train, _ = data_splits[component]
        model.fit(selected_features_train, y_train)
    return models

