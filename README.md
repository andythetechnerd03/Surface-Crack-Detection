# Surface Crack Detection

## Motivation
Surface crack detection is essential in various industries, such as construction and infrastructure maintenance, to ensure safety and longevity. Manual inspection is time-consuming and prone to human error. This project leverages deep learning to automate the detection process, providing a faster and more accurate solution.

## Libraries Used
- [Streamlit](https://streamlit.io/): Used to create the web application interface for uploading images and displaying predictions.
- [PyTorch](https://pytorch.org/): Utilized for building and training the Convolutional Neural Network (CNN) model.
- [OpenCV](https://opencv.org/): Employed for image processing tasks such as reading and preprocessing images.
- [Pillow](https://python-pillow.org/): Another library for image processing, used for handling image file formats and transformations.
- [Pandas](https://pandas.pydata.org/): Used for data manipulation and analysis, particularly in preparing the dataset.
- [KaggleHub](https://pypi.org/project/kagglehub/): Facilitates downloading datasets directly from Kaggle.

## Repository Structure
- `models/cnn.pth`: The trained CNN model file.
- `app.py`: The main file for the Streamlit web application.
- `src/model.py`: Contains the definition of the `SimpleCNN` model.
- `src/data_preparation.py`: Functions for downloading and preparing the dataset.
- `requirements.txt`: List of dependencies required to run the project.
- `.gitignore`: Specifies files and directories to be ignored by Git (mostly notebooks and actual data not necessary for upload).

## Project Definition
The goal of this project is to create a web application that can detect surface cracks in images using a trained Convolutional Neural Network (CNN). Users can upload an image, and the application will predict whether the image contains a crack or not, displaying the result along with the probability.

## Analysis
The analysis involves training a CNN model on a dataset of surface crack images. The model architecture is defined in `src/model.py` and includes convolutional layers followed by fully connected layers. The dataset is downloaded and prepared using functions in `src/data_preparation.py`.

## Conclusion
The web application successfully detects surface cracks in uploaded images with a high degree of accuracy. The use of deep learning and a user-friendly interface makes the application a valuable tool for automated surface crack detection.

## How to Run the App
1. **Clone the Repository**:
    ```sh
    git clone https://github.com/andythetechnerd03/Surface-Crack-Detection.git
    cd Surface-Crack-Detection
    ```

2. **Install Dependencies**:
    Ensure you have Python installed. Then, install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. **Download the Model**:
    Ensure you have the trained model file `cnn.pth` in the project directory. If not, download it from the provided source or train your own model.

4. **Run the Application**:
    Start the Streamlit application:
    ```sh
    streamlit run app.py
    ```

5. **Use the Application**:
    - Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).
    - Upload an image in JPG format.
    - Click the "Predict" button to see the prediction and probability. The web app also includes a segmented image showing the detected crack (this is rule-based, not ML).

## Acknowledgements
- **Contributors**: 
    - [Dinh Ngoc An](https://github.com/andythetechnerd03)
- **Libraries and Tools**:
    - [Streamlit](https://streamlit.io/)
    - [PyTorch](https://pytorch.org/)
    - [OpenCV](https://opencv.org/)
    - [Pillow](https://python-pillow.org/)
    - [Pandas](https://pandas.pydata.org/)
    - [KaggleHub](https://pypi.org/project/kagglehub/)
- **Data Source**:
    - The dataset used for training the model is sourced from [Kaggle](https://www.kaggle.com/arunrk7/surface-crack-detection).

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.