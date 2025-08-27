import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn import tree

st.title("Decision Tree Regressor")

# Generate non-linear dataset
@st.cache_data
def generate_data(n_samples=200):
    np.random.seed(42)
    X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    y = X.ravel()**3 + np.random.randn(n_samples) * 3  # cubic + noise
    return X, y

X, y = generate_data()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Sidebar hyperparameters
st.sidebar.header("Hyperparameter Tuning")

criterion = st.sidebar.selectbox(
    "Criterion (measure of split quality)", 
    ["squared_error", "friedman_mse", "absolute_error", "poisson"]
)

splitter = st.sidebar.selectbox(
    "Splitter (how the split is chosen at each node)", 
    ["best", "random"]
)

max_depth = st.sidebar.number_input(
    "Maximum Depth of Tree", 1, 20, 3
)

min_samples_split = st.sidebar.slider(
    "Minimum Samples Required to Split a Node", 2, 50, 2
)

min_samples_leaf = st.sidebar.slider(
    "Minimum Samples Required at a Leaf Node", 1, 50, 1
)

max_leaf_nodes = st.sidebar.slider(
    "Maximum Leaf Nodes", 2, 100, 20
)

max_features = st.sidebar.selectbox(
    "Maximum Features Considered for Splitting", 
    [None, "sqrt", "log2"]
)

min_impurity_decrease = st.sidebar.number_input(
    "Minimum Impurity Decrease",
    0.0, 1.0, 0.0, 0.01,
    help="A node will be split if this split induces a decrease of the impurity greater than or equal to this value."
)

# Create figure for scatter plot only once
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(X, y, color="blue", label="Data")
ax.set_title("Decision Tree Regressor")
ax.set_xlabel("X")
ax.set_ylabel("y")
ax.grid(True)
ax.legend()

# Display the original scatter plot
st.pyplot(fig)

# Run algorithm button
run_algorithm = st.sidebar.button("Run Algorithm")

if run_algorithm:
    # Model
    model = DecisionTreeRegressor(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=max_leaf_nodes if max_leaf_nodes != 0 else None,
        max_features=max_features,
        min_impurity_decrease=min_impurity_decrease,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    st.write(f"### R² Score: {r2:.2f}")

    # Regression curve
    X_plot = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
    y_plot = model.predict(X_plot)

    # Draw regression line on the same plot
    ax.plot(X_plot, y_plot, color="red", label="Decision Tree Prediction")
    ax.legend()
    st.pyplot(fig)

    # Display the tree structure
    st.write("### Decision Tree Structure")
    plt.figure(figsize=(20, 10))
    tree.plot_tree(
        model,
        feature_names=["X"],
        filled=True,
        rounded=True,
        fontsize=10
    )
    st.pyplot(plt)
