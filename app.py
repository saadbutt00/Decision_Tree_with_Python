import streamlit as st
import numpy as np
import collections

st.set_page_config(page_title="Decision Tree", layout="centered")

# --- HEADER PART ---
st.title("Decision Tree Classifier Model")

st.write("Decision Tree is a supervised machine learning algorithm that splits data into branches based on feature values to make decisions.")
st.write("**Real-world Example:** Used in medical diagnosis to predict disease based on symptoms.")

st.subheader("**Here, How you can predict your values**")
st.markdown("""
**Step 1** - Enter number of features.

**Step 2** - Provide names for each feature and target.

**Step 3** - Enter feature and target values.

**Step 4** - Click 'Submit' to train the model.

**Step 5** - After training, input new data to make predictions.
""")

# --- USER INPUT SECTION ---
st.header("Enter Dataset")

# Get number of features
num_feature = st.number_input("Enter Number of Features:", min_value=1, step=1)

if num_feature > 0:
    feature_names = []
    for i in range(num_feature):
        feature_name = st.text_input(f"Enter name for Feature X{i}:", key=f"f{i}")
        feature_names.append(feature_name)
    target_name = st.text_input("Enter name for Target (Y):", key="target")

    data_entered = False
    if all(feature_names) and target_name:
        st.subheader("Enter Values for Features")

        X = []
        for i in range(num_feature):
            values = st.text_input(f"Enter values for {feature_names[i]} (space-separated):", key=f"values{i}")
            if values:
                X.append(values.split())

        Y_input = st.text_input(f"Enter values for {target_name} (space-separated):")
        if Y_input:
            Y = Y_input.split()
            if len(X) == num_feature and all(len(col) == len(Y) for col in X):
                data_entered = True
            else:
                st.warning("Please ensure all features have same number of samples as target Y.")

        if data_entered:
            if st.button("Submit"):
                st.success("✅ Model Trained!")

                # --- TRAINING LOGIC (your manual ID3 style) ---
                n = len(Y)
                X_rows = list(zip(*X))
                data = list(zip(X_rows, Y))

                def entropy(counter, total):
                    ent = 0
                    for count in counter.values():
                        p = count / total
                        if p != 0:
                            ent += -p * np.log2(p)
                    return ent

                f_idx = 0
                ig_list = []
                st.subheader("Information Gain (Training Result)")
                for f_idx in range(num_feature):
                    value_count = collections.defaultdict(list)
                    
                    for f, t in data:
                        F = f[f_idx]
                        value_count[F].append(t)

                    conditional_entropy = 0
                    for feature, target in value_count.items():
                        counter = collections.Counter(target)
                        size = sum(counter.values())
                        x_entropy = entropy(counter, size)
                        conditional_entropy += (size / n) * x_entropy

                    cnt_Y = collections.Counter(Y)
                    Y_entropy = entropy(cnt_Y, n)
                    ig = Y_entropy - conditional_entropy
                    ig_list.append(ig)
                    st.write(f"Information Gain for {feature_names[f_idx]}: **{ig:.4f}**")

                root_idx = ig_list.index(max(ig_list))
                st.write(f"🟢 **Root Node:** {feature_names[root_idx]}")

                # --- STORE TRAINED DATA ---
                st.session_state.trained = {
                    'feature_names': feature_names,
                    'root_idx': root_idx,
                    'X': X,
                    'Y': Y,
                    'target_name': target_name
                }

# --- PREDICTION PART ---
if 'trained' in st.session_state:
    st.header("Prediction")

    trained = st.session_state.trained
    feature_names = trained['feature_names']
    root_idx = trained['root_idx']
    X = trained['X']
    Y = trained['Y']
    target_name = trained['target_name']

    predict_input = st.text_input(f"Enter value for Root Node ({feature_names[root_idx]}):")

    if st.button("Predict"):
        if predict_input:
            value_count = collections.defaultdict(list)
            for i, value in enumerate(X[root_idx]):
                value_count[value].append(Y[i])

            prediction = None
            if predict_input in value_count:
                pred_counter = collections.Counter(value_count[predict_input])
                prediction = pred_counter.most_common(1)[0][0]
                st.success(f"🔮 Prediction: The model predicts **{target_name} = {prediction}** for {feature_names[root_idx]} = '{predict_input}'")
            else:
                st.warning("The input value not found in training data for root node.")