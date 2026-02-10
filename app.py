import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Linear Regression Learning Lab",
    layout="wide"
)

st.title("📊 Linear Regression Learning Lab")
st.caption("From intuition ➜ math ➜ machine learning")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    x_col = st.selectbox("Select X column", df.columns)
    y_col = st.selectbox("Select Y column", df.columns)

    X = df[x_col].values
    Y = df[y_col].values
    n = len(X)

    tabs = st.tabs([
    "🔍 Explore Data",
    "🎛 Play with Line",
    "📉 Loss Explained",
    "⭐ Best Fit Line",
    "🤖 Gradient Descent (Step)"
    ])


    # ---------------- TAB 1 ----------------
    with tabs[0]:
        st.subheader("Explore the Data")

        st.write("""
        Before regression, we **only observe the data**.
        Try to imagine where a straight line *might* go.
        """)

        fig, ax = plt.subplots()
        ax.scatter(X, Y)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title("Scatter Plot of Data")
        st.pyplot(fig)

    # ---------------- TAB 2 ----------------
    with tabs[1]:
        st.subheader("Play with the Line")

        col1, col2 = st.columns(2)

        with col1:
            m = st.slider("Slope (m)", -10.0, 10.0, 1.0, 0.1)
            b = st.slider("Intercept (b)", -20.0, 20.0, 0.0, 0.5)

            st.latex(r"y = mx + b")

        Y_pred = m * X + b
        errors = Y - Y_pred
        mse = np.mean(errors ** 2)

        with col2:
            st.metric("Mean Squared Error", round(mse, 4))

        fig, ax = plt.subplots()
        ax.scatter(X, Y, label="Actual Data")
        ax.plot(X, Y_pred, label="Your Line")

        # error lines
        for i in range(len(X)):
            ax.plot([X[i], X[i]], [Y[i], Y_pred[i]], linestyle="dotted")

        ax.legend()
        ax.set_title("Your Line & Errors")
        st.pyplot(fig)

        st.info("Vertical dotted lines represent **errors** (actual − predicted)")

    # ---------------- TAB 3 ----------------
    with tabs[2]:
        st.subheader("How Loss is Calculated")

        st.latex(r"""
        \text{Error} = y - \hat{y}
        """)
        st.latex(r"""
        \text{Loss (MSE)} = \frac{1}{n}\sum (y - \hat{y})^2
        """)

        loss_df = pd.DataFrame({
            "x": X,
            "Actual y": Y,
            "Predicted ŷ": Y_pred,
            "Error (y-ŷ)": errors,
            "Squared Error": errors ** 2
        })

        st.dataframe(loss_df)

        st.success(f"Final MSE = {round(mse, 4)}")

        fig, ax = plt.subplots()
        ax.bar(range(len(errors)), errors ** 2)
        ax.set_title("Squared Errors for Each Data Point")
        ax.set_ylabel("Error²")
        st.pyplot(fig)

    # ---------------- TAB 4 ----------------
    with tabs[3]:
        st.subheader("Best Fit Line (Closed-Form Solution)")

        st.write("""
        Instead of guessing, mathematics gives us the **best fit line**
        that minimizes Mean Squared Error.
        """)

        X_mean = np.mean(X)
        Y_mean = np.mean(Y)

        m_best = np.sum((X - X_mean) * (Y - Y_mean)) / np.sum((X - X_mean) ** 2)
        b_best = Y_mean - m_best * X_mean

        Y_best = m_best * X + b_best
        best_mse = np.mean((Y - Y_best) ** 2)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Best m", round(m_best, 4))
            st.metric("Best b", round(b_best, 4))

        with col2:
            st.metric("Your Line MSE", round(mse, 4))
            st.metric("Best Line MSE", round(best_mse, 4))

        fig, ax = plt.subplots()
        ax.scatter(X, Y, label="Data")
        ax.plot(X, Y_pred, label="Your Line", linestyle="--")
        ax.plot(X, Y_best, label="Best Fit Line")
        ax.legend()
        ax.set_title("Comparison: Your Line vs Best Fit")
        st.pyplot(fig)

        st.info("The best-fit line is what ML algorithms try to learn automatically.")

    # ---------------- TAB 5 ----------------
    with tabs[4]:
        st.subheader("Step-by-Step Gradient Descent")
    
        st.write("""
        Each click performs **one gradient descent update**.
        Try to *predict* what will happen before clicking.
        """)
    
        # Initialize session state
        if "m_gd" not in st.session_state:
            st.session_state.m_gd = 0.0
            st.session_state.b_gd = 0.0
            st.session_state.losses = []
            st.session_state.epoch = 0
    
        lr = st.slider("Learning Rate", 0.001, 0.1, 0.01)
    
        col1, col2 = st.columns(2)
    
        # STEP BUTTON
        if col1.button("➡ Step Gradient Descent"):
            Y_hat = st.session_state.m_gd * X + st.session_state.b_gd
    
            dm = (-2/n) * np.sum(X * (Y - Y_hat))
            db = (-2/n) * np.sum(Y - Y_hat)
    
            st.session_state.m_gd -= lr * dm
            st.session_state.b_gd -= lr * db
    
            loss = np.mean((Y - Y_hat) ** 2)
            st.session_state.losses.append(loss)
            st.session_state.epoch += 1
    
        # RESET BUTTON
        if col2.button("🔄 Reset"):
            st.session_state.m_gd = 0.0
            st.session_state.b_gd = 0.0
            st.session_state.losses = []
            st.session_state.epoch = 0
    
        # Current metrics
        st.metric("Epoch", st.session_state.epoch)
    
        current_loss = (
            st.session_state.losses[-1]
            if st.session_state.losses else None
        )
    
        if current_loss is not None:
            st.metric("Current MSE", round(current_loss, 4))
    
        # Plot regression line
        fig, ax = plt.subplots()
        ax.scatter(X, Y, label="Data")
        ax.plot(
            X,
            st.session_state.m_gd * X + st.session_state.b_gd,
            label="Gradient Descent Line"
        )
        ax.legend()
        ax.set_title("Line Updating After Each Step")
        st.pyplot(fig)
    
        # Loss curve
        if st.session_state.losses:
            fig2, ax2 = plt.subplots()
            ax2.plot(st.session_state.losses)
            ax2.set_xlabel("Step")
            ax2.set_ylabel("MSE")
            ax2.set_title("Loss Reduction Over Steps")
            st.pyplot(fig2)
    
        st.info("""
        Gradient Descent keeps updating **m** and **b**  
        until the loss stops decreasing.
        """)


else:
    st.info("👆 Upload a CSV file to start learning")
