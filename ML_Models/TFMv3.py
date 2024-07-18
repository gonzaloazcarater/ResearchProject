#!/usr/bin/env python
# coding: utf-8

# ## 1- Imports

# In[18]:


# imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
import requests
from io import BytesIO


# ## 2 - Load Dataset

# In[19]:


df=pd.read_excel('Data/Dataset_v3.xlsx')
# strip column names
df=df.rename(columns=lambda x: x.strip())
cols=df.columns
df.head(5)

#includes al collums from transfermarket


# ## 3 - Data Preprocessing

# In[20]:


## #check for missign values
print('ColumnName, DataType, MissingValues')
for i in cols:
        if  df[i].isnull().any():
            print(i, ',', df[i].dtype,',',df[i].isnull().any())


# In[21]:


#rename colums
df = df.rename(columns={
    'Season Ticket Avg Price(€)': 'Seasonticket',
    'Avg Attendance': 'Avgatt',
    'Avg Annual Salary (USD)': 'Avgsalr',
    'Popuation(millions)': 'Pop',
    'Capacity': 'Cap',
    'Number of Spectators': 'Nspec'
})
df.head(10)


# ## 4 - Graphs

# In[22]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14, 10))

# Plotting Season Ticket Avg Price(€) vs Avg Attendance
plt.subplot(2, 2, 1)
sns.scatterplot(data=df, x='Seasonticket', y='Avgatt')
plt.title('Season Ticket Avg Price vs Avg Attendance')
plt.xlabel('Season Ticket Avg Price(€)')
plt.ylabel('Avg Attendance')

# Plotting Avg Annual Salary (USD) vs Avg Attendance
plt.subplot(2, 2, 2)
sns.scatterplot(data=df, x='Avgsalr', y='Avgatt')
plt.title('Avg Annual Salary vs Avg Attendance')
plt.xlabel('Avg Annual Salary (USD)')
plt.ylabel('Avg Attendance')

# Plotting Season Ticket Avg Price(€) vs Population(millions)
plt.subplot(2, 2, 3)
sns.scatterplot(data=df, x='Seasonticket', y='Pop')
plt.title('Season Ticket Avg Price vs Population')
plt.xlabel('Season Ticket Avg Price(€)')
plt.ylabel('Population (millions)')

# Plotting Avg Annual Salary (USD) vs Population(millions)
plt.subplot(2, 2, 4)
sns.scatterplot(data=df, x='Avgsalr', y='Pop')
plt.title('Avg Annual Salary vs Population')
plt.xlabel('Avg Annual Salary (USD)')
plt.ylabel('Population (millions)')


plt.tight_layout()
# Save the figure as an image
plt.savefig('scatter.png')
# Show the figure in a pop-up window
plt.show()


# In[6]:


import pandas as pd
import matplotlib.pyplot as plt

df_graphs = df.drop(columns=['Team'])

# Agrupar los datos por país y calcular la media de las columnas numéricas
aggregated_df = df_graphs.groupby('Country').mean().reset_index()

# Plotting
plt.figure(figsize=(12, 8))

# Bar plot of 'Country' vs 'Avg Attendance'
plt.subplot(2, 2, 1)
sns.barplot(data=aggregated_df, x='Country', y='Avgatt')
plt.title('Average Attendance by Country')
plt.xlabel('Country')
plt.ylabel('Average Attendance')
plt.xticks(rotation=45)

# Bar plot of 'Country' vs 'Season Ticket Avg Price(€)'
plt.subplot(2, 2, 2)
sns.barplot(data=aggregated_df, x='Country', y='Seasonticket')
plt.title('Average Season Ticket Price by Country')
plt.xlabel('Country')
plt.ylabel('Average Season Ticket Price')
plt.xticks(rotation=45)

# Bar plot of 'Country' vs 'Avg Annual Salary (USD)'
plt.subplot(2, 2, 3)
sns.barplot(data=aggregated_df, x='Country', y='Avgsalr')
plt.title('Average Annual Salary by Country')
plt.xlabel('Country')
plt.ylabel('Average Annual Salary')
plt.xticks(rotation=45)

# Bar plot of 'Country' vs 'Population(millions)'
plt.subplot(2, 2, 4)
sns.barplot(data=aggregated_df, x='Country', y='Pop')
plt.title('Population (millions) by Country')
plt.xlabel('Country')
plt.ylabel('Population (millions)')
plt.xticks(rotation=45)

plt.tight_layout()



# Save the figure as an image
plt.savefig('bar_plot.png')
# Show the figure in a pop-up window
plt.show()



# In[11]:


import matplotlib.pyplot as plt
from PIL import Image

# Define paths to flag images
flag_paths = {
    'Spain': 'Imgs/Spain.png',
    'Italy': 'Imgs/Italy.png',
    'Mexico': 'Imgs/Mexico.png',
    'England': 'Imgs/England.png',
    'France': 'Imgs/France.png',
    'Germany': 'Imgs/Germany.png',
    'Usa': 'Imgs/Usa.png',

    # Add paths for other countries
}

# Load images
flag_images = {country: Image.open(path) for country, path in flag_paths.items()}

# Create a list with bar positions
x_positions = range(len(aggregated_df))

# Create the bars
plt.figure(figsize=(12, 8))
bars = plt.bar(x_positions, aggregated_df['Avgatt'])

# Create a secondary x-axis for the images
ax2 = plt.gca().twinx()

# Add the images below the bars
for i, country in enumerate(aggregated_df['Country']):
    flag_image = flag_images.get(country)  # Get the image corresponding to the country
    if flag_image:
        # Set the position of the image
        img_width = bars[i].get_width()
        img_height = img_width * flag_image.height / flag_image.width
        img_x = bars[i].get_x() + (bars[i].get_width() - img_width) / 2
        img_y = -img_height * 0.8
        ax2.imshow(flag_image, extent=(img_x, img_x + img_width, img_y, img_y + img_height), aspect='auto')

# Customize the plot
plt.title('Asistencia Promedio por País')
plt.xlabel('País')
plt.ylabel('Asistencia Promedio')
plt.xticks(x_positions, aggregated_df['Country'], rotation=45)
plt.tight_layout()


# Save the figure as an image
plt.savefig('country.png')
# Show the figure in a pop-up window
plt.show()


# ## Variable = Seasonticket.

# ## 5 - Predictive Analysis - Decision Tree.

# In[51]:


# train and test separation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder 
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
df_dt=df.copy(True)

# Discretize the variable Seasonticket into three classes: 'Low', 'Medium', and 'High'
bin_edges = [0, 1000, 2000, np.inf]  # Ranges: 0-1000, 1000-2000, over 2000
labels = ['Low', 'Medium', 'High']
df_dt['Seasonticket_class'] = pd.cut(df_dt['Seasonticket'], bins=bin_edges, labels=labels)

# Define features (X) and target (y)
X = df_dt[['Avgatt', 'Avgsalr', 'Pop','Cap','Nspec']]
y = df_dt['Seasonticket_class']  # Discrete target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df_dt.head(5)


# In[52]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder 
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score, make_scorer


# Define decision tree parameters for grid search
param_grid = {
    'criterion': ['entropy', 'gini'],  # Split criterion
    'max_depth': [None, 5, 10, 15],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10, 20],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4, 8]  # Minimum number of samples required at a leaf node
}

# Create decision tree classifier
clf_DT = DecisionTreeClassifier()

# Shuffle training sets
X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train)

# Perform grid search with cross-validation
grid_search = GridSearchCV(clf_DT, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_shuffled, y_train_shuffled)

# Get the best model found through grid search
best_model = grid_search.best_estimator_

# Print best parameters and accuracy
print("Best Parameters:", grid_search.best_params_)
print("Accuracy Score:", grid_search.best_score_)

# Select only relevant features from the test set
X_test_selected = X_test[['Avgatt', 'Avgsalr', 'Pop','Cap','Nspec']]
y_pred = best_model.predict(X_test_selected)

# Evaluate on the test set
y_pred = best_model.predict(X_test_selected)
# Calculate metrics on the test set
acc_test = accuracy_score(y_test, y_pred)
prec_test = precision_score(y_test, y_pred, average='micro', zero_division=0)
rec_test = recall_score(y_test, y_pred, average='micro', zero_division=0)
f1_test = f1_score(y_test, y_pred, average='micro')

print("Metrics for the test set")
print('Accuracy on test set:', acc_test)
print('Precision on test set:', prec_test)
print('Recall on test set:', rec_test)
print('F1 on test set:', f1_test)


# In[53]:


## 4.3 - Visualization of the tree
# DOT data
dot_data = export_graphviz(
best_model,
out_file=None,
feature_names=X_train_shuffled.columns.to_list(),
class_names=list(map(str, labels)),
filled=True)
# Draw graph
graph = graphviz.Source(dot_data, format="png")
# Save graph to MyDecisionTree.png
graph.render("MyDecisionTree1")
graph


# ## 5 - KNN

# In[58]:


from sklearn import neighbors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection
selector = SelectKBest(score_func=f_classif, k=3)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Define distance metrics to try
metrics = ['manhattan', 'euclidean', 'minkowski']

best_metric = None
best_k = None
best_accuracy = -1
best_prec = -1
best_rec = -1
best_f1 = -1
accuracies = []
k_values = range(1, 20)

for metric in metrics:
    print("Using metric:", metric)
    accuracies = []
    for k in range(1, 20, 1):  
        classifier = neighbors.KNeighborsClassifier(k, metric=metric)
        classifier.fit(X_train_selected, y_train)
        y_pred_test = classifier.predict(X_test_selected)

        # Calculates the metrics in the test set
        acc = accuracy_score(y_test, y_pred_test)
        prec = precision_score(y_test, y_pred_test, average='micro', zero_division=0)
        rec = recall_score(y_test, y_pred_test, average='micro', zero_division=0)
        f1 = f1_score(y_test, y_pred_test, average='micro')

        accuracies.append(acc)  # Append accuracy score for the current k

        # Check if these metrics are the best so far.
        if acc > best_accuracy:
            best_accuracy = acc
            best_metric = metric
            best_k = k
            best_prec = prec
            best_rec = rec
            best_f1 = f1

print("Best parameters found:")
print("Metric:", best_metric)
print("k:", best_k)
print("Accuracy:", best_accuracy)
print("Precision:", best_prec)
print("Recall:", best_rec)
print("F1-score:", best_f1)
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-')
plt.title('Accuracy vs. k in k-Nearest Neighbors')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.xticks(np.arange(1, 20, step=1))
plt.grid(True)
plt.savefig('knn_diagram_1.png')
plt.show()


# ## 6.1 - Linear Regression - No feature selection

# In[60]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

df_lr=df.copy(True)

# Discretize the variable Seasonticket into three classes: 'Low', 'Medium', and 'High'
bin_edges = [0, 1000, 2000, np.inf]  # Ranges: 0-1000, 1000-2000, over 2000
labels=['Low', 'Medium', 'High']
df_lr['Seasonticket_class'] = pd.cut(df_lr['Seasonticket'], bins=bin_edges, labels=labels)

# Define features (X) and target (y)
X = df_lr[['Avgatt', 'Avgsalr', 'Pop','Cap','Nspec']]
y = df_lr['Seasonticket_class']  # Discrete target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf_lr=LogisticRegression(penalty='l2',solver='lbfgs')
# Train the classifier and evaluate on the test set
clf_lr.fit(X_train, y_train)
y_pred = clf_lr.predict(X_test)
# Calculate metrics on the test set
acc_test = accuracy_score(y_test, y_pred)
prec_test = precision_score(y_test, y_pred, average='micro', zero_division=0)
rec_test = recall_score(y_test, y_pred, average='micro', zero_division=0)
f1_test = f1_score(y_test, y_pred, average='micro')
print("Metrics for the test set")
print('Accuracy on test set:', acc_test)
print('Precision on test set:', prec_test)
print('Recall on test set:', rec_test)
print('F1 on test set:', f1_test)


# In[65]:


# Get feature importance
coef = clf_lr.coef_[0]  # For logistic regression with one-vs-rest approach, use coef_[i] for each class
feature_names = X.columns

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_names, coef, color='blue')
plt.xlabel('Coefficient value')
plt.title('Feature Importance in Logistic Regression')
plt.savefig('linear_diagram_1.png')
plt.show()


# ## 6.2 - Linear Regression with feature selection method

# In[63]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector, RFE, SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Número total de características en tu conjunto de datos
num_features = X_train.shape[1]

# Define los métodos de selección de características
feature_selection_methods = {
    'Sequential Forward Selection': SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1),
                                                              n_features_to_select=None,
                                                              direction='forward',
                                                              scoring='accuracy',
                                                              cv=5),
    'Sequential Backward Selection': SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1),
                                                               n_features_to_select=None,
                                                               direction='backward',
                                                               scoring='accuracy',
                                                               cv=5),
    'RFE': RFE(RandomForestClassifier(n_jobs=-1), n_features_to_select=min(5, num_features)),
    'SelectKBest': SelectKBest(mutual_info_classif, k=min(5, num_features))  # Ajusta 'k' dinámicamente
}

# Inicializa un diccionario para almacenar los resultados
results = {}

# Itera a través de los métodos de selección de características
for name, selector in feature_selection_methods.items():
    print(f"Running feature selection: {name}")

    # Ajusta el selector en los datos de entrenamiento
    selector.fit(X_train, y_train)

    # Obtiene las características seleccionadas
    selected_features = selector.get_support()
    selected_features_names = X_train.columns[selected_features]

    # Transforma los conjuntos de entrenamiento y prueba con las características seleccionadas
    x_train_selected = X_train[selected_features_names]
    x_test_selected = X_test[selected_features_names]

    # Inicializa el clasificador
    clf_lr = LogisticRegression(penalty='l2', solver='lbfgs')

    # Ajusta el clasificador en las características seleccionadas del conjunto de entrenamiento
    clf_lr.fit(x_train_selected, y_train)

    # Realiza predicciones en el conjunto de prueba
    y_pred = clf_lr.predict(x_test_selected)

    # Calcula las métricas en el conjunto de prueba
    acc_test = accuracy_score(y_test, y_pred)
    prec_test = precision_score(y_test, y_pred, average='micro', zero_division=0)
    rec_test = recall_score(y_test, y_pred, average='micro', zero_division=0)
    f1_test = f1_score(y_test, y_pred, average='micro')

    # Almacena los resultados
    results[name] = {
        'Selected Features': selected_features_names,
        'Accuracy': acc_test,
        'Precision': prec_test,
        'Recall': rec_test,
        'F1 Score': f1_test
    }

# Imprime los resultados
for method, result in results.items():
    print(f"\nMetrics for {method}")
    print('Selected Features:', list(result['Selected Features']))
    print('Accuracy:', result['Accuracy'])
    print('Precision:', result['Precision'])
    print('Recall:', result['Recall'])
    print('F1 Score:', result['F1 Score'])


# In[64]:


# Get feature importance
coef = clf_lr.coef_[0]  # For logistic regression with one-vs-rest approach, use coef_[i] for each class
feature_names = X.columns

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_names, coef, color='blue')
plt.xlabel('Coefficient value')
plt.title('Feature Importance in Logistic Regression')
plt.show()


# ## SVM

# In[71]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

# 6.2 -SVM linear
from sklearn.svm import SVC
df_svm=df.copy(True)

# Discretize the variable Seasonticket into three classes: 'Low', 'Medium', and 'High'
bin_edges = [0, 1000, 2000, np.inf]  # Ranges: 0-1000, 1000-2000, over 2000
labels=['Low', 'Medium', 'High']
df_svm['Seasonticket_class'] = pd.cut(df_svm['Seasonticket'], bins=bin_edges, labels=labels)

# Define features (X) and target (y)
X = df_svm[['Avgatt', 'Avgsalr', 'Pop']]
y = df_svm['Seasonticket_class']  # Discrete target variable

labels = ['Low', 'Medium', 'High']
df_svm['Seasonticket_class'] = pd.cut(df_svm['Seasonticket'], bins=bin_edges, labels=labels)

# Define features (X) and target (y)
X = df_svm[['Avgatt', 'Avgsalr', 'Pop']]  # Adjust features as per your dataset
y = df_svm['Seasonticket_class']

# Convert categorical labels to integer labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning using cross-validation
param_grid = [
    {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear']},
    {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['poly'], 'degree': [2, 3, 4]},
    {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['rbf']}
]

cv = StratifiedKFold(n_splits=5)
grid_search = GridSearchCV(SVC(max_iter=10000), param_grid, cv=cv, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best classifier
clf_svm = grid_search.best_estimator_

# Print the best kernel and C parameter
print('Best kernel:', grid_search.best_params_['kernel'])
print('Best C parameter:', grid_search.best_params_['C'])

# Evaluate on test set
y_pred = clf_svm.predict(X_test)

# Calculate metrics on the test set
acc_test = accuracy_score(y_test, y_pred)
prec_test = precision_score(y_test, y_pred, average='micro', zero_division=0)
rec_test = recall_score(y_test, y_pred, average='micro', zero_division=0)
f1_test = f1_score(y_test, y_pred, average='micro')

print("\nMetrics for the test set")
print('Accuracy on test set:', acc_test)
print('Precision on test set:', prec_test)
print('Recall on test set:', rec_test)
print('F1 on test set:', f1_test)


# In[73]:


# Define plot decision regions using all features
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

# Generate meshgrid points with step size h
h = 0.02  # Meshgrid step size
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict each point on the meshgrid to visualize decision regions
# Here we need to use the meshgrid (xx, yy) with all three features
mesh_points = np.c_[xx.ravel(), yy.ravel()]  # Use only 'Avgatt' and 'Avgsalr' for visualization
if X_train.shape[1] > 2:
    mesh_points = np.hstack([mesh_points, np.zeros((mesh_points.shape[0], X_train.shape[1] - 2))])

Z = clf_svm.predict(mesh_points)
Z = Z.reshape(xx.shape)

# Plot decision regions and data points
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3)  # Color decision regions
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, cmap='viridis', edgecolor='k')  # Data points
plt.xlabel('Avgatt')
plt.ylabel('Avgsalr')
plt.title('Decision Regions of SVM')
plt.colorbar()
plt.savefig('svm_diagram1.png')
plt.show()


# In[ ]:




