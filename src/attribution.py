import numpy as np


X_dict = {
    'XMEAS_1':'A_feed_stream',
    'XMEAS_2':'D_feed_stream',
    'XMEAS_3':'E_feed_stream',
    'XMEAS_4':'Total_fresh_feed_stripper',
    'XMEAS_5':'Recycle_flow_into_rxtr',
    'XMEAS_6':'Reactor_feed_rate',
    'XMEAS_7':'Reactor_pressure',
    'XMEAS_8':'Reactor_level',
    'XMEAS_9':'Reactor_temp',
    'XMEAS_10':'Purge_rate',
    'XMEAS_11':'Separator_temp',
    'XMEAS_12':'Separator_level',
    'XMEAS_13':'Separator_pressure',
    'XMEAS_14':'Separator_underflow',
    'XMEAS_15':'Stripper_level',
    'XMEAS_16':'Stripper_pressure',
    'XMEAS_17':'Stripper_underflow',
    'XMEAS_18':'Stripper_temperature',
    'XMEAS_19':'Stripper_steam_flow',
    'XMEAS_20':'Compressor_work',
    'XMEAS_21':'Reactor_cooling_water_outlet_temp',
    'XMEAS_22':'Condenser_cooling_water_outlet_temp',
    'XMEAS_23':'Composition_of_A_rxtr_feed',
    'XMEAS_24':'Composition_of_B_rxtr_feed',
    'XMEAS_25':'Composition_of_C_rxtr_feed',
    'XMEAS_26':'Composition_of_D_rxtr_feed',
    'XMEAS_27':'Composition_of_E_rxtr_feed',
    'XMEAS_28':'Composition_of_F_rxtr_feed',
    'XMEAS_29':'Composition_of_A_purge',
    'XMEAS_30':'Composition_of_B_purge',
    'XMEAS_31':'Composition_of_C_purge',
    'XMEAS_32':'Composition_of_D_purge',
    'XMEAS_33':'Composition_of_E_purge',
    'XMEAS_34':'Composition_of_F_purge',
    'XMEAS_35':'Composition_of_G_purge',
    'XMEAS_36':'Composition_of_H_purge',
    'XMEAS_37':'Composition_of_D_product',
    'XMEAS_38':'Composition_of_E_product',
    'XMEAS_39':'Composition_of_F_product',
    'XMEAS_40':'Composition_of_G_product',
    'XMEAS_41':'Composition_of_H_product',
    'XMV_1':'D_feed_flow_valve',
    'XMV_2':'E_feed_flow_valve',
    'XMV_3':'A_feed_flow_valve',
    'XMV_4':'Total_feed_flow_stripper_valve',
    'XMV_5':'Compressor_recycle_valve',
    'XMV_6':'Purge_valve',
    'XMV_7':'Separator_pot_liquid_flow_valve',
    'XMV_8':'Stripper_liquid_product_flow_valve',
    'XMV_9':'Stripper_steam_valve',
    'XMV_10':'Reactor_cooling_water_flow_valve',
    'XMV_11':'Condenser_cooling_water_flow_valve'
}

x_dict = {k.lower(): v for k, v in X_dict.items()}

feature_name_map = {}

for k, v in x_dict.items():
    feature_name_map[k] = v
    feature_name_map[f"{k}_rollmean"] = f"{v}_rolling_mean"
    feature_name_map[f"{k}_diff"] = f"{v}_rate_of_change"


def map_feature_name(col):
    return feature_name_map.get(col.lower(), col)


def get_top_contributing_variables(X_seq, X_pred, feature_columns, top_n=5):
    error_matrix = np.square(X_seq[0] - X_pred[0])
    feature_error = np.mean(error_matrix, axis=0)

    latest_values = X_seq[0][-1]

    candidates = []

    for i, col in enumerate(feature_columns):
        if "fault" in col.lower():
            continue

        candidates.append((i, col, float(feature_error[i])))

    candidates = sorted(candidates, key=lambda x: x[2], reverse=True)[:top_n]

    top_features = []

    for idx, col, err in candidates:
        top_features.append({
            "raw_feature": col,
            "mapped_feature": map_feature_name(col),
            "current_scaled_value": float(latest_values[idx]),
            "reconstruction_error_contribution": err
        })

    return top_features