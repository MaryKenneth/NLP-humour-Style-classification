import numpy as np

def train_test_spliting(df,train_ratio=0.8,rand_seed=42):

    """
    Split the input DataFrame into training and testing sets.
    
    Args:
        df (DataFrame): Input DataFrame.
        train_ratio (float): Ratio of data to be used for training. Default is 0.8 (80%).
    
    Returns:
        x_train, x_test, y_train, y_test: Numpy arrays representing the features and labels for training and testing.
    """

    df_array  = np.array(df)        # Convert to array
    np.random.seed(rand_seed)               # Use a fixed random seed (5) for reproducibility
    np.random.shuffle(df_array) 

    split_ratio = int(len(df_array) * train_ratio)  # Train Data: 80% and Test data: 20%

    #Train data
    train_data = df_array[:split_ratio]  # Get all training features and labels
    x_train = train_data[:,0]            # Extract Train features only
    y_train = train_data[:,-1]           # Extract Train Labels only

    #Test Data
    test_data = df_array[split_ratio:] # Get all testing features and labels
    x_test = test_data[:,0]            # Extract Test features only
    y_test = test_data[:,-1]           # Extract Test Labels only

    return x_train, x_test, y_train, y_test
