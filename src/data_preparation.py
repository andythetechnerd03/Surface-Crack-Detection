import kagglehub

DATA_URL = 'arunrk7/surface-crack-detection'

# Load the data
def download_data():
    # Download the data, and get the path to the downloaded file
    surface_crack_detection_path = kagglehub.dataset_download(DATA_URL)
    import os
    # copy the data to the data/ directory
    os.system(f'cp {surface_crack_detection_path} data/')
    print(f'Data downloaded to {surface_crack_detection_path}')

def train_test_split(path: str, test_size: float):
    # Get the two image folders "Neagtive" and "Positive" and split them
    pass