import pandas as pd
import numpy as np
    
def set_sample_size(path_to_powerflow_data, data_to_change, n_samples, n_original_samples, seed=None):
    '''

    Parameters
    ----------
    data_to_change: list of strings.
        ex: ["loads-p_set", "generators-p_max_pu", "snapshots"] 
    '''

    data = {}
    for datatype in data_to_change:
        data[datatype] = pd.read_csv(path_to_powerflow_data + datatype + ".csv")
        

    def increase_data(dataframe, n_samples, seed=None):
        addon = {}
        new_df_list = []
        for idx, column in enumerate(dataframe):
            if dataframe[column].dtype == np.float64 or dataframe[column].dtype == np.int64:  
                addon[column] = np.abs(
                    np.random.RandomState(seed=seed).normal(loc=dataframe[column][0:n_original_samples].mean(), 
                                                            scale=dataframe[column][0:n_original_samples].std(), 
                                                            size=n_samples))
            elif dataframe[column].dtype == object:
                # assuming object is datetime column
                latest_datetime = pd.to_datetime(dataframe[column][n_original_samples-1])
                addon[column] = []
                for sample in range(n_samples):
                    addon[column].append(latest_datetime + pd.Timedelta(hours=(1+sample)))
            else:
                raise TypeError("dataframe[column] type: {} should be object or float64/int64".format(
                    type(dataframe[column].dtype)))
        addon_dataframe = pd.DataFrame(addon)
        return dataframe.head(n_original_samples).append(addon_dataframe)            

            
    def cap_data(dataframe, min_value, max_value):
        for column in dataframe:
            if dataframe[column].dtype == np.float64 or dataframe[column].dtype == np.int64:
                for i, val in enumerate(dataframe[column]):
                    if val > max_value:
                        dataframe[column][i] = max_value
                    elif val < min_value:
                        dataframe[column][i] = min_value
        return dataframe
    
    
    
                
    if n_samples > n_original_samples:
        for datatype in data:
            data[datatype] = increase_data(data[datatype], n_samples-n_original_samples, seed)
            if datatype == "generators-p_max_pu":
                data[datatype] = cap_data(data[datatype], 0, 1)
        
    for datatype in data:
        data[datatype].to_csv(path_to_powerflow_data + datatype + ".csv", index=False)
    