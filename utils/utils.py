import numpy as np
import pandas as pd
import torch
import os
from absl import logging
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from utils.data_utils import latent_to_index, calculate_scale, function_y, function_x, function_a
from sklearn.model_selection import train_test_split

def setup_directories(FLAGS):
    """Create and prepare output directories."""
    if FLAGS.output_name == "":
        dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        dir_name = FLAGS.output_name
    out_dir = os.path.join(os.path.abspath(FLAGS.output_dir), dir_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir

def save_results(out_dir, results_dict):
    """Save the results to a file."""
    result_path = os.path.join(out_dir, "results.npz")
    np.savez(result_path, **results_dict)
    logging.info("Results saved successfully.")

def cf_output(model, loader_counterfactual, img_example = False):
    """
    Evaluate the trained model on counterfactual samples from different types of loaders and compile the results into a DataFrame.

    Parameters:
        model (torch.nn.Module): The trained neural network model.
        loader_counterfactual (torch.utils.data.DataLoader): DataLoader containing the counterfactual samples for tabular data.

    Returns:
        pd.DataFrame: DataFrame containing the predicted outcomes.
    """
    results = []

    if not img_example:
        for data in loader_counterfactual:
            inputs, _ = torch.split(data[0], (data[0].shape[1] - 1), 1)
            inputs = inputs.float()
            with torch.no_grad():
                predictions = model(inputs).squeeze()
            batch_results = {'values': [predictions.item()]}  
            results.append(batch_results)

    else:
        tot_values = []
        for _, data in enumerate(loader_counterfactual):
            with torch.no_grad():
                y = model(data)
            dist_values = [y[0][0].detach().item()]
            tot_values.extend(dist_values)

        img_df = pd.DataFrame(tot_values, columns=['values'])
        results.append({'values': img_df['values'].values})

    final_df = pd.concat([pd.DataFrame(r) for r in results], ignore_index=True)
    return final_df

#############################
# Data processing
#############################

def get_loaders(df_tabular, imgs, batch_size):  
    
    """
    Splits data into training and testing datasets, then creates loaders for images,
    tabular data, and labels.

    Parameters:
    - df_tabular (DataFrame): Contains latent variables and outcome variable 'output'.
    - imgs (np.array): Array of images corresponding to the entries in df_tabular.
    - batch_size (int): The size of the batches for data loading.

    Returns:
    - loader_train (DataLoader): DataLoader for training images, tabular data and outcome
    - loader_test (DataLoader): DataLoader for testing images, tabular data and outcome
    """

    # 80/20 train/test data split
    rnd = np.random.uniform(0, 1, len(df_tabular))
    train_idx = np.where(rnd < 0.8)[0]
    test_idx = np.where(rnd >= 0.8)[0]

    # Split in train/test for tabular data
    d_tab = np.array(df_tabular.drop(['output'], axis=1)).astype(np.float32)
    tabular_tensor_train = torch.tensor(d_tab[train_idx])
    tabular_tensor_test = torch.tensor(d_tab[test_idx])

    # Split in train/test for images
    img_tensor_train = torch.tensor(imgs[train_idx]).float()
    img_tensor_test = torch.tensor(imgs[test_idx]).float()

    # Split in train/test for labels
    labels_train = torch.tensor(df_tabular.loc[train_idx, 'output'].values).float()
    labels_test = torch.tensor(df_tabular.loc[test_idx, 'output'].values).float()

    # Creating TensorDatasets
    train_dataset = TensorDataset(img_tensor_train, tabular_tensor_train, labels_train)
    test_dataset = TensorDataset(img_tensor_test, tabular_tensor_test, labels_test)

    # Get loaders
    loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return loader_train, loader_test

def standardize(data, stats):
    mean, std = stats
    return (data - mean) / std 

def data_processing(df, batch_size):
    """
    standardize data and splits it into training and testing datasets.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing 'A', 'X', 'Z', 'Y' columns.
        batch_size (int): The size of each batch for the DataLoader.
    
    Returns:
        tuple: Tuple containing DataLoader for train and test datasets, and normalization parameters.
    """
    # standardize features
    norm_factors = {col: (df[col].mean(), df[col].std()) for col in ['X', 'Z', 'A','Y']}
    for col, stats in norm_factors.items():
        df[col] = standardize(df[col], stats)

    # Split data into training and testing sets
    train_df, test_df = train_test_split(df[['X', 'Z', 'A', 'Y']], test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    train_tensor = torch.tensor(train_df.values, dtype=torch.float32)
    test_tensor = torch.tensor(test_df.values, dtype=torch.float32)

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_tensor), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, norm_factors



def process_image_data(df):
    df = df.reset_index(drop=True)
    df_no_unobs = df.drop(['noise_scale'], axis=1)  
    stats = (df_no_unobs.mean(), df_no_unobs.std())
    df_no_unobs = standardize(df_no_unobs, stats) 
    df_no_unobs['color'] = 0  # color in dsprites dataset is always white
    
    return df, df_no_unobs, stats


#############################
# Generating counterfactuals
#############################

def generate_counterfactuals(n_count, stats, data):
    """
    Generates counterfactual samples based on a selected data point from the observed data.
    
    Parameters:
        n_count (int): Number of counterfactual samples to generate.
        stats (dict): Dictionary containing mean and std for A, X, Z, Y.
        data (pd.DataFrame): Observed data set.    
    Returns:
        tuple: Normalized counterfactuals of A, (X, Z), and Y.
    """
    # Select a random sample from observed data
    sample = data.sample(n=1)
    Z = sample['Z'].values[0]
    noise_x = sample['u_x'].values[0]
    noise_y = sample['u_y'].values[0]

    # Generate random values for Z in counterfactuals
    count_Z = np.random.normal(0, 1, n_count)

    # Generate corresponding A values
    count_A = np.exp(-0.5 * count_Z ** 2) * np.sin(2 * count_Z) + np.random.normal(0, 1, n_count)

    # Calculate counterfactual values for X and Y
    count_X = function_x(count_A, Z) + 0.1 * noise_x
    count_Y = function_y(count_A, Z, count_X) + 0.1 * noise_y

    # standardize counterfactuals
    norm_count_A = standardize(count_A, (stats['meanA'], stats['stdA']))
    norm_count_X = np.stack((count_X, np.full(n_count, Z)), axis=1)
    norm_count_X = standardize(norm_count_X, (np.array(stats['meanX']), np.array(stats['stdX'])))
    norm_count_Y = standardize(count_Y, (stats['meanY'], stats['stdY']))

    return norm_count_A, norm_count_X, norm_count_Y

def counterfactual_simulations(n_samples, n_count, df):
    """
    For a given number of samples, generate n_count counterfactuals and store them in data loaders.

    Parameters:
        n_samples (int): Number of samples data points.
        n_count (int): Number of counterfactual samples per data point.
        df (pd.DataFrame): Data from which statistical values are derived and sample is chosen.
        n_scenario (int): Scenario identifier to define different generation behaviors.

    Returns:
        dict: Dictionary of DataLoaders, each containing counterfactual samples for a data point.
    """
    stats = {
        'meanA': np.mean(df['A']), 'stdA': np.std(df['A']),
        'meanX': np.mean(df[['X', 'Z']], axis=0), 'stdX': np.std(df[['X', 'Z']], axis=0),
        'meanY': np.mean(df['Y']), 'stdY': np.std(df['Y'])
    }
    count_loader_tot = {}
    for i in range(n_samples):
        count_A, count_X, count_Y = generate_counterfactuals(n_count, stats, df)
        vec_count = np.concatenate((count_X, count_A[:, np.newaxis], count_Y[:, np.newaxis]), axis=1)
        count_tensor = torch.tensor(vec_count, dtype=torch.float32)
        count_loader_tot[i] = DataLoader(TensorDataset(count_tensor), batch_size=1, shuffle=False)

    return count_loader_tot

def generate_counterfactuals_img(n_count, stats, imgs, df_latent, latents_bases):
    """generate n_count counterfactual for a given sample"""
    
    # select random data point from df_latent
    index = np.random.randint(low=0, high=len(df_latent), size=1)
    shape = df_latent.loc[index, 'shape']
    posY = df_latent.loc[index, 'posY']
    color = 0
    orientation = df_latent.loc[index, 'orientation']
    noise_scale = df_latent.loc[index, 'noise_scale']

    # generate counterfactuals of posX and scale
    count_posx = df_latent.loc[np.random.randint(low=0, high=len(df_latent), size=n_count), 'posX']
    
    count_values_scale = calculate_scale(count_posx, np.full(n_count, posY).flatten(), np.full(n_count, shape).flatten(), np.full(n_count, noise_scale).flatten())

    # store counterfactual samples in df_count
    df_count = pd.DataFrame(np.vstack((np.full(n_count, color).flatten(), np.full(n_count, shape).flatten(),
                                       count_values_scale, np.full(n_count, orientation).flatten(), count_posx,
                                       np.full(n_count, posY).flatten())).T,
                            columns=['color', 'shape', 'scale', 'orientation', 'posX', 'posY'])
    df_count = df_count.astype('int32')

    indices_sampled_count = latent_to_index(df_count, latents_bases)
    imgs_sampled_causal_count = imgs[indices_sampled_count]

    df_count['output'] = np.full((n_count, 1), np.nan) # not needed
    
    norm_df_count = standardize(df_count, stats)  
    norm_df_count = norm_df_count.drop(['output'], axis=1)
    norm_df_count['color'] = np.full(n_count, 0).reshape(-1, 1)  

    return norm_df_count, imgs_sampled_causal_count


def counterfactual_simulations_img(n_samples, n_count, stats, imgs, df_total_final, latents_bases):
    """Generate n_count counterfactuals for n_samples datapoints with function generate_counterfactuals"""
    loader_trainer_images = {}
    loader_trainer_tabular = {}

    for i in range(n_samples):
        transform = transforms.ToTensor()

        df_count, imgs_sampled_causal_count = \
            generate_counterfactuals_img(n_count, stats, imgs, df_total_final, latents_bases)

        d_tab = np.array(df_count).astype(np.float32)
        tabular_tensor_count = transform(d_tab).reshape(-1, 6)
        tensor_imgs_count = transform(imgs_sampled_causal_count).reshape(-1, 64, 64)
        loader_trainer_images[i] = DataLoader(tensor_imgs_count, batch_size=1, num_workers=0)
        loader_trainer_tabular[i] = DataLoader(tabular_tensor_count, batch_size=1, num_workers=0)

    return loader_trainer_images, loader_trainer_tabular


