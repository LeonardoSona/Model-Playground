import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, mean_squared_error, silhouette_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64

# Streamlit app title
st.title("ML Model Playground")

# Sidebar for dataset upload
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

# Provide an example file to download
st.sidebar.markdown("### Example Dataset")
example_data = pd.DataFrame({
    "Feature1": [1, 2, 3, 4, 5],
    "Feature2": [10, 20, 30, 40, 50],
    "Target": [0, 1, 0, 1, 0]
})
csv = example_data.to_csv(index=False).encode('utf-8')
b64 = base64.b64encode(csv).decode('utf-8')
st.sidebar.markdown(f"[Download Example CSV](data:file/csv;base64,{b64})")

# Sidebar for model selection
st.sidebar.title("Choose Algorithm")
model_type = st.sidebar.selectbox(
    "Select Model Type:", ["Classification", "Regression", "Clustering"]
)

# Data Handling Capabilities
st.subheader("What is Possible in Data Handling")
st.markdown(
    """
    - **Automatic Scaling:** Features are automatically standardized for better model performance.
    - **Encoding Categorical Variables:** Non-numeric columns can be converted to numeric.
    - **Handling Missing Values:** Drop or fill missing values (to be enhanced).
    - **Feature Selection:** Choose which features and target column to use.
    """
)

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Dataset")
    st.write(data.head())

    # Feature selection
    st.sidebar.title("Feature Selection")
    target_column = st.sidebar.selectbox("Select target column:", data.columns)
    features = st.sidebar.multiselect(
        "Select feature columns:", [col for col in data.columns if col != target_column]
    )

    if target_column and features:
        X = data[features]
        y = data[target_column]

        # Preprocessing
        for col in X.select_dtypes(include=[object]).columns:
            X[col] = LabelEncoder().fit_transform(X[col])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data if applicable
        if model_type in ["Classification", "Regression"]:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42
            )

        # Model selection
        if model_type == "Classification":
            st.write("### When to Use Classification")
            st.markdown(
                "Use classification when your target variable is categorical. For example, predicting whether an email is spam (0 or 1)."
            )

            classifier = st.sidebar.selectbox(
                "Select Classifier:", ["Logistic Regression", "Random Forest"]
            )
            
            if classifier == "Logistic Regression":
                C = st.sidebar.slider("Regularization (C):", 0.01, 10.0, 1.0)
                model = LogisticRegression(C=C, random_state=42)
            elif classifier == "Random Forest":
                n_estimators = st.sidebar.slider("Number of Trees:", 10, 200, 100)
                max_depth = st.sidebar.slider("Max Depth:", 1, 20, 5)
                model = RandomForestClassifier(
                    n_estimators=n_estimators, max_depth=max_depth, random_state=42
                )
            
            # Train and evaluate
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write("### Classification Report")
            st.text(classification_report(y_test, y_pred))

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            st.write("### Confusion Matrix")
            fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual", color="Count"))
            st.plotly_chart(fig)

        elif model_type == "Regression":
            st.write("### When to Use Regression")
            st.markdown(
                "Use regression when your target variable is continuous. For example, predicting house prices."
            )

            regressor = st.sidebar.selectbox(
                "Select Regressor:", ["Linear Regression", "Random Forest"]
            )

            if regressor == "Linear Regression":
                model = LinearRegression()
            elif regressor == "Random Forest":
                n_estimators = st.sidebar.slider("Number of Trees:", 10, 200, 100)
                max_depth = st.sidebar.slider("Max Depth:", 1, 20, 5)
                model = RandomForestRegressor(
                    n_estimators=n_estimators, max_depth=max_depth, random_state=42
                )
            
            # Train and evaluate
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write("### Mean Squared Error")
            st.write(mean_squared_error(y_test, y_pred))

            # Prediction vs. Ground Truth
            st.write("### Prediction vs. Ground Truth")
            fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual", "y": "Predicted"}, title="Prediction vs. Ground Truth")
            st.plotly_chart(fig)

        elif model_type == "Clustering":
            st.write("### When to Use Clustering")
            st.markdown(
                "Use clustering to group data points into clusters based on similarity. For example, customer segmentation."
            )

            n_clusters = st.sidebar.slider("Number of Clusters:", 2, 10, 3)
            model = KMeans(n_clusters=n_clusters, random_state=42)
            model.fit(X_scaled)

            labels = model.labels_
            st.write("### Silhouette Score")
            st.write(silhouette_score(X_scaled, labels))

            # Visualize clustering
            fig = px.scatter(x=X_scaled[:, 0], y=X_scaled[:, 1], color=labels, labels={"x": features[0], "y": features[1]}, title="Clustering Visualization")
            st.plotly_chart(fig)

        # Download results
        st.subheader("Download Results")
        results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred}) if model_type != "Clustering" else pd.DataFrame({"Labels": labels})
        csv = results.to_csv(index=False).encode('utf-8')
        b64 = base64.b64encode(csv).decode('utf-8')
        st.markdown(f"[Download Results CSV](data:file/csv;base64,{b64})")

else:
    st.write("Please upload a CSV file to start.")
