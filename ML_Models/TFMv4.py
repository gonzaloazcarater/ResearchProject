#!/usr/bin/env python
# coding: utf-8

# ## 1- Imports

# In[79]:


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

# In[80]:


df=pd.read_excel('Data/Dataset_v3.xlsx')
# strip column names
df=df.rename(columns=lambda x: x.strip())
cols=df.columns
df.head(5)
#includes al collums from transfermarket


# ## 3 - Data Preprocessing

# In[81]:


## #check for missign values
print('ColumnName, DataType, MissingValues')
for i in cols:
        if  df[i].isnull().any():
            print(i, ',', df[i].dtype,',',df[i].isnull().any())


# In[82]:


#rename colums
df = df.rename(columns={
    'Season Ticket Avg Price(€)': 'Seasonticket',
    'Avg Attendance': 'Avgatt',
    'Avg Annual Salary (USD)': 'Avgsalr',
    'Popuation(millions)': 'Pop',
    'Capacity': 'Cap',
    'Number of Spectators': 'Nspec'
})
df.head(5)


# # AVG ATT

# ## Decision Tree

# In[83]:


import numpy as np

# Datos de Avg Attendance
avg_attendance = [
    24301, 61459, 60236, 62440, 53288, 73534, 39576, 24881, 31543, 55809, 
    41921, 31037, 11098, 52153, 39042, 29386, 17082, 21153, 11244, 29962, 
    81305, 75000, 17718, 15000, 51371, 56959, 49829, 55121, 45175, 41721, 
    34196, 30731, 29984, 29301, 25917, 25393, 24559, 21841, 46780, 61535, 
    27300, 37195, 25429, 15159, 42055, 14555, 7505, 25377, 29619, 14421, 
    22889, 25358, 9491, 21636, 24608, 5121, 6953, 1635, 46112, 59121, 
    19703, 18016, 20039, 17391, 39843, 11456, 12520, 16350, 12758, 17767, 
    51259, 72061, 31667, 34984, 12893, 25041, 43420, 17957, 72000, 12179, 
    29001, 62925, 14725, 25914, 10861, 21363, 72838, 39345, 22641, 5943, 
    43716, 46977, 22753, 21547, 9373, 26644, 18162, 14578, 22524, 19244, 
    31197, 39003, 39243, 12413, 22354, 17480, 24483, 35214, 24025, 12827, 
    11982, 11529, 16264, 15578, 38627, 20853, 50961, 20738, 17552, 39785, 
    19520, 15409, 20349, 18801, 25327, 19096, 17112, 21326, 22195, 25184, 
    19644, 28772, 33093, 20251, 18246, 23030, 18943, 20816, 19671, 30887, 
    30052, 29259, 22423, 25498, 26728
]

# Calcular estadísticas básicas
avg_min = np.min(avg_attendance)
avg_max = np.max(avg_attendance)
avg_mean = np.mean(avg_attendance)
avg_std = np.std(avg_attendance)

# Mostrar estadísticas
print(f"Mínimo: {avg_min}")
print(f"Máximo: {avg_max}")
print(f"Media: {avg_mean}")
print(f"Desviación estándar: {avg_std}")


# In[84]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Copy the original DataFrame
df_dt2 = df.copy(True)

# Discretize the variable 'Avgatt' into three classes: 'Low', 'Medium', and 'High'
bin_edges_avgatt = [0, 13000, 45000, np.inf]  # Ranges: 0-13000, 13000-45000, over 45000
labels_avgatt = ['Low', 'Medium', 'High']

df_dt2['Avgatt_class'] = pd.cut(df_dt2['Avgatt'], bins=bin_edges_avgatt, labels=labels_avgatt, include_lowest=True)

# Features without 'Avgatt', because this is what we want to predict
X = df_dt2[['Seasonticket', 'Avgsalr', 'Pop', 'Cap', 'Nspec']]
y = df_dt2['Avgatt_class']  # Target variable is the class of Avgatt

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Show the first 5 rows of the modified DataFrame
df_dt2.head(5)


# In[85]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder 
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
X_test_selected = X_test[['Seasonticket', 'Avgsalr', 'Pop','Cap','Nspec']]

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


# In[47]:


## 4.3 - Visualization of the tree
# DOT data
dot_data = export_graphviz(
best_model,
out_file=None,
feature_names=X_train_shuffled.columns.to_list(),
class_names=list(map(str, labels_avgatt)),
filled=True)
# Draw graph
graph = graphviz.Source(dot_data, format="png")
# Save graph to MyDecisionTree.png
graph.render("MyDecisionTree2")
graph


# ## KNN - Avg Att

# In[86]:


from sklearn import neighbors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Split the data set into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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


# In[87]:


plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-')
plt.title('Accuracy vs. k in k-Nearest Neighbors')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.xticks(np.arange(1, 20, step=1))
plt.grid(True)
plt.savefig('knn_diagram_2.png')
plt.show()


# ## Logistic regression

# In[88]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# Copia del DataFrame original
df_lr = df.copy()

# Discretize the variable Avg att into three classes: 'Low', 'Medium', and 'High'
bin_edges_avgatt = [0, 13000, 45000, np.inf]  # Ranges: 0-1000, 1000-2000, over 2000
labels_avgatt = ['Low', 'Medium', 'High']

df_dt2['Avgatt_class'] = pd.cut(df_dt2['Avgatt'], bins=bin_edges_avgatt, labels=labels_avgatt, include_lowest=True)

# Características sin Avgatt, porque es lo que queremos predecir
X = df_dt2[['Seasonticket', 'Avgsalr', 'Pop', 'Cap', 'Nspec']]
y = df_dt2['Avgatt_class']  # Variable objetivo es la clase de cuartil de Avgatt

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mostrar las primeras 5 filas del DataFrame modificado
df_dt2.head(5)

# Dividir el conjunto de datos en entrenamiento y prueba con estratificación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)

# Define y entrena el clasificador de regresión logística
clf_lr = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)
clf_lr.fit(X_train, y_train)

# Predecir en el conjunto de prueba y calcular métricas
y_pred = clf_lr.predict(X_test)
acc_test = accuracy_score(y_test, y_pred)
prec_test = precision_score(y_test, y_pred, average='micro', zero_division=0)
rec_test = recall_score(y_test, y_pred, average='micro', zero_division=0)
f1_test = f1_score(y_test, y_pred, average='micro')

print("Metrics for the test set:")
print('Accuracy on test set:', acc_test)
print('Precision on test set:', prec_test)
print('Recall on test set:', rec_test)
print('F1-score on test set:', f1_test)


# In[63]:


# Get feature importance
coef = clf_lr.coef_[0]  # For logistic regression with one-vs-rest approach, use coef_[i] for each class
feature_names = X.columns

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_names, coef, color='blue')
plt.xlabel('Coefficient value')
plt.title('Feature Importance in Logistic Regression')
plt.savefig('linear_diagram_2.png')
plt.show()


# In[89]:


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
    prec_test = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_test = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_test = f1_score(y_test, y_pred, average='macro')

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


# ## SVM Avgatt

# In[126]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
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
df_svm2=df.copy(True)

# Discretize the variable Avg att into three classes: 'Low', 'Medium', and 'High'
bin_edges_avgatt = [0, 13000, 45000, np.inf]  # Ranges: 0-1000, 1000-2000, over 2000
labels_avgatt = ['Low', 'Medium', 'High']

df_svm2['Avgatt_class'] = pd.cut(df_svm2['Avgatt'], bins=bin_edges_avgatt, labels=labels_avgatt, include_lowest=True)

# Características sin Avgatt, porque es lo que queremos predecir
X = df_svm2[['Seasonticket', 'Avgsalr', 'Pop', 'Cap', 'Nspec']]
y = df_svm2['Avgatt_class']  # Variable objetivo es la clase de cuartil de Avgatt

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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


# In[128]:


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
plt.savefig('svm_diagram2.png')
plt.show()


# In[ ]:




