# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"data/Social_Network_Ads.csv")
    return df

st.title("Decision Tree Classifier")

# Load data
df = load_data()
  
# Features and Target
X = df.iloc[:, [2, 3]].values   # Age, EstimatedSalary
y = df.iloc[:, 4].values        # Purchased

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Standardization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Sidebar hyperparameters
st.sidebar.header("Hyperparameter Tuning")

criterion = st.sidebar.selectbox(
    "Criterion (measure of split quality)", 
    ["gini", "entropy", "log_loss"]
)

splitter = st.sidebar.selectbox(
    "Splitter (how the split is chosen at each node)", 
    ["best", "random"]
)

# Depth control (important for small datasets)
max_depth = st.sidebar.number_input(
    "Maximum Depth of Tree", 
    1, 10, 3, 
    help="How deep the tree can grow. Smaller values prevent overfitting."
)

# Minimum samples to split a node
min_samples_split = st.sidebar.slider(
    "Minimum Samples Required to Split a Node", 
    2, len(df), 4, 
    help="Higher values prevent splits on very small subsets."
)

# Minimum samples required at a leaf
min_samples_leaf = st.sidebar.slider(
    "Minimum Samples Required at a Leaf Node", 
    1, 15, 2, 
    help="Larger values smooth the model (avoid leaves with too few samples)."
)

# Max number of leaf nodes
max_leaf_nodes = st.sidebar.slider(
    "Maximum Leaf Nodes", 
    2, 50, None, 
    help="Limits the number of leaf nodes. None means unlimited."
)

# Max features for split
max_features = st.sidebar.selectbox(
    "Maximum Features Considered for Splitting", 
    [None, "sqrt", "log2"], 
    help="Restricting features reduces variance. None = all features."
)

# Class weight balancing
class_weight = st.sidebar.selectbox(
    "Class Weight", 
    [None, "balanced"], 
    help="Balanced adjusts weights inversely proportional to class frequencies."
)

# Model
classifier = DecisionTreeClassifier(
    criterion=criterion,
    splitter=splitter,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    max_features=max_features,
    max_leaf_nodes=max_leaf_nodes if max_leaf_nodes != 0 else None,
    class_weight=class_weight,
    random_state=42
)


classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"### Model Accuracy: {acc:.2f}")

# Decision Boundary Plot
def plot_decision_boundary(X_set, y_set, title):
    X1, X2 = np.meshgrid(
        np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
        np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01),
    )
    plt.contourf(
        X1,
        X2,
        classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
        alpha=0.75,
        cmap=plt.cm.coolwarm,
    )
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(
            X_set[y_set == j, 0],
            X_set[y_set == j, 1],
            c=["red", "green"][i],
            label=j,
        )
    plt.title(title)
    plt.xlabel("Age (scaled)")
    plt.ylabel("Estimated Salary (scaled)")
    plt.legend()

# Plot for Training set
st.write("### Decision Boundary (Training set)")
fig, ax = plt.subplots()
plot_decision_boundary(X_train, y_train, "Decision Tree (Training set)")
st.pyplot(fig)

# # Plot for Test set
# st.write("### Decision Boundary (Test set)")
# fig, ax = plt.subplots()
# plot_decision_boundary(X_test, y_test, "Decision Tree (Test set)")
# st.pyplot(fig)