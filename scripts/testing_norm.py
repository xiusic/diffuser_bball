import numpy as np
import pandas as pd

def load_and_inspect_npy(file_path):
    # Load the .npy file
    data = np.load(file_path, allow_pickle=True)
    
    # Inspect the loaded data
    print(f"Data type: {type(data)}")
    print(f"Data shape: {data.shape}")
    print(f"Data preview: {data[:2]}")  # Print the first 2 elements for inspection
    
    return data

def convert_to_dataframe(data):
    # Assuming data is a list of dictionaries or list of lists
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Display DataFrame info
    print("DataFrame info:")
    print(df.info())
    
    return df

def check_and_handle_nan_inf(df):
    # Check for NaN values
    if df.isnull().values.any():
        print("DataFrame contains NaN values.")
    else:
        print("No NaN values found in DataFrame.")
    
    # Check for infinite values
    if np.isinf(df.values).any():
        print("DataFrame contains infinite values.")
    else:
        print("No infinite values found in DataFrame.")
    
    # Handle NaN and infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    return df

def main(file_path):
    # Load and inspect .npy file
    data = load_and_inspect_npy(file_path)
    
    # Convert to DataFrame
    df = convert_to_dataframe(data)
    
    # Check and handle NaN and infinite values
    df = check_and_handle_nan_inf(df)
    
    # Attempt to convert to JSON
    try:
        json_str = df.to_json()
        print("Conversion to JSON successful.")
    except OverflowError as e:
        print(f"OverflowError during JSON conversion: {e}")

if __name__ == "__main__":
    file_path = '/local2/dmreynos/diffuser_bball/logs/guided_samplesnew_test2_cond100_0.1/2016.NBA.Raw.SportVU.Game.Logs12.05.2015.POR.at.MIN_dir-1-guided-245K.npy'
    main(file_path)