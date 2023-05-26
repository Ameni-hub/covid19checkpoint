# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
import streamlit as st
import requests
api_url = 'https://api.coronavirus.data.gov.uk/v1/data'
response = requests.get(api_url)
data = response.json()
#print (data)
#response.status_code
#response.headers["Content-Type"]
import pandas as pd
df = pd.DataFrame(data, columns=['length', 'maxPageLimit', 'totalRecords', 'data'])
#print(df.head())
my_dict_1 = df['data']
my_list_1 = list(my_dict_1)
#print (my_list_1)
df2 = pd.DataFrame(my_list_1)
#print (df2)
df3 =df2
# Convert the 'date' column to datetime format and set it as the index
df2['date'] = pd.to_datetime(df2['date'])
df2.set_index('date', inplace=True)

# Print the DataFrame
#print(df2)
#df2.info()
#df2.shape
df2.isnull().sum()
df2['deathNew'].fillna((df2['deathNew'].mean()),inplace=True)
df2['confirmedRate'].fillna((df2['confirmedRate'].mean()),inplace=True)
df2['death'].fillna((df2['death'].mean()),inplace=True)
df2['deathRate'].fillna((df2['deathRate'].mean()),inplace=True)
import matplotlib.pyplot as plt

# Assuming df1 is your DataFrame

data = [df2['confirmedRate'], df2['latestBy'], df2['confirmed'], df2['deathNew'], df2['death'], df2['deathRate']]
labels=['Confirmed Rate', 'Latest By', 'Confirmed', 'Death New', 'Death', 'Death Rate']
plt.boxplot(data,labels )

plt.xlabel('Columns')
plt.ylabel('Values')
plt.title('Box Plot')
#plt.show()


import numpy as np
Q3 =  df3['confirmed'].quantile(0.75)
Q1 = df3['confirmed'].quantile(0.25)
IQR = Q3 - Q1
df3['confirmed'] = np.where(df3['confirmed']> Q3 + 1.5 * IQR, Q3, df3['confirmed'])
plt.boxplot(df2['confirmed'])
Q3 =  df3['latestBy'].quantile(0.75)
Q1 =  df3['latestBy'].quantile(0.25)
IQR = Q3 - Q1
df3['latestBy'] = np.where(df3['latestBy']> Q3 + 1.5 * IQR, Q3, df3['latestBy'])
plt.boxplot(df2['latestBy'])
Q3 =  df3['confirmedRate'].quantile(0.75)
Q1 =  df3['confirmedRate'].quantile(0.25)
IQR = Q3 - Q1
df3['confirmedRate'] = np.where(df3['confirmedRate']> Q3 + 1.5 * IQR, Q3, df3['confirmedRate'])
plt.boxplot(df3['confirmedRate'])
Q3 =  df3['deathNew'].quantile(0.75)
Q1 =  df3['deathNew'].quantile(0.25)
IQR = Q3 - Q1
df3['deathNew'] = np.where(df3['deathNew']> Q3 + 1.5 * IQR, Q3, df3['deathNew'])
plt.boxplot(df3['deathNew'])
Q3 =  df3['death'].quantile(0.75)
Q1 =  df3['death'].quantile(0.25)
IQR = Q3 - Q1
df3['death'] = np.where(df3['death']> Q3 + 1.5 * IQR, Q3, df3['death'])
plt.boxplot(df3['death'])
import pandas as pd

# Assuming you have a DataFrame named 'data' with the specified columns and date index
data = df2[['latestBy', 'deathNew', 'death', 'deathRate','confirmed','confirmedRate']]
corr_matrix = data.corr()

print(corr_matrix)

# Print the correlation matrix
print("Correlation Matrix:")
print(corr_matrix)

# Select the best features based on correlation
# Assuming 'target_variable' is the column you are interested in
target_variable = 'deathRate'
corr_threshold = 0.6  # Set the correlation threshold

# Get the absolute correlation values for the target variable or features of interest
target_corr = corr_matrix[target_variable].abs()

# Sort the correlation values in descending order
sorted_corr = target_corr.sort_values(ascending=False)

# Select the features with correlation above the threshold
best_features = sorted_corr[sorted_corr > corr_threshold].index.tolist()

# Print the best features
print("Best Features:")
for feature in best_features:
    print(feature)
df2= df2.drop(['latestBy','deathNew','areaName', 'areaCode'], axis =1)
#df2
from matplotlib import pyplot
pyplot.plot(df2) # creates a time series plot with the number of daily births on the y-axis and time in days along the x-axis.
pyplot.show()
x = df2['confirmed']
y = df2['deathRate']

# Plotting the data
plt.plot(x, y)
plt.xlabel('confirmed')
plt.ylabel('deathRate')
plt.title('Data Plot')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics
#extract x and y from our data
X=df2[['death','confirmed','confirmedRate']]
Y=df2['deathRate'].values
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.35,random_state=40) #splitting data with test size of 35%
model=LinearRegression()   #build linear regression model
model.fit(X_train,Y_train)  #fitting the training data
predicted=model.predict(X_test) #testing our model’s performance
print("MSE", mean_squared_error(Y_test,predicted))
print("R squared", metrics.r2_score(Y_test,predicted))
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures # create matrix and vectors
x=df2[['death','confirmed','confirmedRate']]
y=df2['deathRate']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=40)  #splitting data
lg=LinearRegression()
poly=PolynomialFeatures(degree=3)
x_train_fit = poly.fit_transform(x_train) #transforming our input data
lg.fit(x_train_fit, y_train)
x_test_ = poly.fit_transform(x_test)
predicted = lg.predict(x_test_)
print("MSE: ", metrics.mean_squared_error(y_test, predicted))
print("R squared: ", metrics.r2_score(y_test,predicted))
from sklearn.model_selection import cross_val_score
clf = LinearRegression()

# Perform cross-validation
scores = cross_val_score(clf, x, y, cv=10, scoring='neg_mean_squared_error')

# Calculate the average score
avg_score = -scores.mean()

# Print the scores
print("Mean squared error on each fold:")
print(scores)
print("Average mean squared error:", avg_score)
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Create the pipeline with polynomial features and linear regression
model = make_pipeline(PolynomialFeatures(), LinearRegression())
x=df2[['death','confirmed','confirmedRate']]
y=df2['deathRate']
# Define the hyperparameter grid
param_grid = {
    'polynomialfeatures__degree': [2, 3, 4],
    'linearregression__fit_intercept': [True, False],
}

# Perform grid search
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(x, y)

# Print the best hyperparameters and score
print("Best Hyperparameters: ", grid_search.best_params_)
print("Best Score: ", -grid_search.best_score_)
x=df3[['death','confirmed','confirmedRate']]
y=df3['deathRate']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=40)  #splitting data
lg=LinearRegression()
poly=PolynomialFeatures(degree=3)
x_train_fit = poly.fit_transform(x_train) #transforming our input data
lg.fit(x_train_fit, y_train)
x_test_ = poly.fit_transform(x_test)
predicted = lg.predict(x_test_)
print("MSE: ", metrics.mean_squared_error(y_test, predicted))
print("R squared: ", metrics.r2_score(y_test,predicted))
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import streamlit as st
    from plotly import graph_objs as go
    from datetime import datetime
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    import locale

    df2.reset_index(inplace=True)
    # Rename the column
    df2.rename(columns={'index': 'date'}, inplace=True)
    # Convert the column to datetime format (optional)
    df2['date'] = pd.to_datetime(df2['date'])

    st.title('Covid 19 dataset')
    choice = st.sidebar.radio('Navigation',['Home','Prediction'])
    if choice == 'Home':
        if st.checkbox('Show table'):
            st.table(df2)
    graph = st.selectbox("What kind of Graph ? ", ["Non-Interactive", "Interactive"])
    if graph == "Non-Interactive":
        fig, ax = plt.subplots()
        plt.scatter(df2['date'], df2['deathRate'])
        plt.xlabel("Years")
        plt.ylabel("Death Rate")
        plt.tight_layout()
        st.pyplot(fig)
    if graph == "Interactive":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df2['date'], y=df2['deathRate'], mode='lines'))
        fig.update_layout(
            xaxis=dict(range=['2020-01-01', '2023-12-31']),
            yaxis=dict(range=[0, 300])
        )
        st.plotly_chart(fig)
    if choice == "Prediction":
        val = st.number_input("Enter death number", 0.00, 1000000.00, step=500.0)

        features = df2[['confirmedRate', 'confirmed', 'death']].values
        labels = df2['deathRate'].values

        # Create polynomial features
        poly_features = PolynomialFeatures(degree=3)

        # Transform features
        x_poly_train = poly_features.fit_transform(features)
        x_poly_val = poly_features.transform(np.array([[val, val, val]]))

        # Train the polynomial regression model
        poly_regression = LinearRegression()
        poly_regression.fit(x_poly_train, labels)

        # Make a prediction
        pred = poly_regression.predict(x_poly_val)[0]

        # Set the locale to format the prediction value
        locale.setlocale(locale.LC_ALL, 'en_US')

        # Transform the prediction value with thousands separators
        transformed_value = locale.format_string("%d", round(pred), grouping=True)

        if st.button("Predict"):
            st.success(f"The rate of death is {transformed_value}")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

