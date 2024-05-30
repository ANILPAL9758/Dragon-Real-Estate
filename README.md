# Dragon Real Estate
Tools and Technologies Used
1.Python Libraries:

pandas: For data manipulation and analysis.
numpy: For numerical operations.
scikit-learn: For machine learning, including preprocessing, model selection, and evaluation.
joblib: For model serialization.

2.Jupyter Notebook: An interactive computing environment for creating and sharing documents that contain live code, equations, visualizations, and narrative text.

Project Structure and Workflow
Loading Data:

The project starts by loading the dataset using pandas. The dataset appears to be related to real estate prices, likely similar to the Boston housing dataset.
python
code:-
import pandas as pd
housing = pd.read_csv("data.csv")
Data Splitting:

The dataset is split into training and test sets using train_test_split from scikit-learn.
python
code:-
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
Preprocessing:

Imputation: Missing values are handled using SimpleImputer to fill them with the median value.
Transformation Pipeline: A pipeline is created using Pipeline from scikit-learn to streamline the preprocessing steps.
python
code:-
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing)
X = imputer.transform(housing)
housing_tr = pd.DataFrame(X, columns=housing.columns)
Feature Scaling:

Features are scaled to ensure they are on a similar scale, improving the performance of machine learning algorithms.
Model Training:

The notebook trains a machine learning model using RandomForestRegressor, a powerful ensemble method.
python
code:-
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(housing_prepared, housing_labels)
Model Evaluation:

The trained model is evaluated using metrics such as Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) on the test set.
python
 code
from sklearn.metrics import mean_squared_error
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
Model Persistence:

The trained model is saved to a file using joblib for future use.
python
 code
from joblib import dump
dump(model, 'Dragon.joblib')
Model Inference:

The saved model is loaded and used to make predictions on new data.
python
 code
from joblib import load
model = load('Dragon.joblib')
features = np.array([[-5.43942006, 4.12628155, -1.6165014, -0.67288841, -1.42262747,
                      -11.44443979304, -49.31238772,  7.61111401, -26.0016879 , -0.5778192 ,
                      -0.97491834,  0.41164221, -66.86091034]])
model.predict(features)


Summary of the Project:-
Objective: To build a machine learning model to predict real estate prices based on various features.
Data Preprocessing: Handling missing values and scaling features.
Model Training: Using RandomForestRegressor to train the model.
Evaluation: Using RMSE to evaluate the model's performance.
Persistence and Inference: Saving the model with joblib and making predictions on new data.
