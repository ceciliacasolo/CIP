import numpy as np
import pandas as pd


def function_a(Z):
    """
    Compute the value of A (np.ndarray) from Z (np.ndarray).
    """
    return Z * Z

def function_x(A, Z):
    """
    Compute the value of X (np.ndarray) from A (np.ndarray) and Z (np.ndarray).
    """
    return np.exp(-0.5 * A * A) * np.sin(2 * A) + 2 * Z

def function_y(A, Z, X):
    """
    Compute the value of Y (np.ndarray) from A, Z, and X (np.ndarray) .
    """
    return 0.5 * np.sin(2 * X * Z) * np.exp(-Z * X) + 5 * A


def simulation_data(n_samples, scenario=1):
    """
    Generate simulated data for given number of samples and a specified scenario.
    
    Parameters:
        n_samples (int): The number of samples to generate.
        scenario (int, optional): Scenario identifier for different simulation methods. Defaults to 1.
    
    Returns:
        pd.DataFrame: A DataFrame containing the simulated data.
    """
    Z = np.random.normal(0, 1, size=n_samples)
    A = function_a(Z) + np.random.normal(0, 1, size=n_samples)

    noise_x = np.random.normal(0, 0.1, size=n_samples)
    noise_y = np.random.normal(0, 0.1, size=n_samples)
    X = function_x(A, Z) + 0.2 * noise_x
    Y = function_y(A, Z, X) + 0.2 * noise_y
    
    #XZ = np.vstack((X, Z)).T
    df = pd.DataFrame({
        'A': A,
        'X': X,
        'Z': Z,
        'Y': Y,
        'u_x': noise_x,
        'u_y': noise_y
    })
    return df


def latent_to_index(latents, latents_bases):
    return np.dot(latents, latents_bases).astype(int)

def sample_latent_dataframe(n_samples, latent_sizes, column_names):
    """Generates a dataframe of sampled latent variables."""
    latent_values = np.random.randint(0, latent_sizes, size=(n_samples, len(column_names)))
    df = pd.DataFrame(latent_values, columns=column_names)
    return df.astype('Int64')

def calculate_position_x(df):
    """Calculates modified posX based on shape, posY and noise."""
    values_posX = np.round(np.random.normal(df['shape'] + df['posY'], 1))
    values_posX = np.clip(values_posX, 0, 32)
    return values_posX.astype(int)

def calculate_scale(posX, posY, shape, noise):
    """Calculates scale based on posX, posY, shape and noise."""
    scale = (posX/24 + posY/24) * shape + noise
    scale = np.clip(scale, 0, 5)
    return scale.astype(int)

def generate_output(df, imgs):
    """Calculate output based on posX, posY, shape, scale, orientation, color and noise."""
    noise = np.random.normal(0, 0.01, size=df.shape[0])
    output = np.exp(df['shape']) * df['posX'] + df['scale']**2 * np.sin(df['posY']) + 0.2 * df['orientation'] + df['color'] + noise
    return output

def find_causal_dataset(n_small, n_large, latent_sizes, latents_bases, imgs):
    """Generates a new dataset via matching a defined SCM."""
    column_names = ['color', 'shape', 'scale', 'orientation', 'posX', 'posY']
    
    # Sample latent variables for two datasets
    df_total = sample_latent_dataframe(n_large, latent_sizes, column_names)
    df_small = sample_latent_dataframe(n_small, latent_sizes, column_names)

    # Calculate position X and scale with causal relations
    df_small['posX'] = calculate_position_x(df_small)
    
    df_small['noise_scale'] = np.random.normal(0, 1, size=len(df_small))
    df_small['scale'] = calculate_scale(df_small['posX'], df_small['posY'], df_small['shape'], df_small['noise_scale'])

    # Merge the small dataframe modifications into the large one based on causal keys
    df_merged = df_total.merge(df_small[['shape', 'posY', 'posX', 'scale', 'noise_scale']],
                               on=['shape', 'posX', 'posY', 'scale'], how='left', indicator=True)
    
    df_causal = df_merged[df_merged['_merge'] == 'both'].drop('_merge', axis=1)

    # Map latent variables to indices and fetch corresponding images
    indices = latent_to_index(df_causal.drop('noise_scale', axis=1), latents_bases)

    # Apply complex causal relationships to generate outputs
    df_causal['output'] = generate_output(df_causal, imgs)

    return df_causal