import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


st.header("Interactive Linear Regression Project")
try:
    input_csv = st.text_input("Input Your csv Dataset","Example : xxx.csv")
    df = pd.read_csv("{}".format(input_csv))
    cols = df.columns
    st.dataframe(df.head())
    st.markdown("Columns")
    st.write(list(df.columns))

    st.markdown("Missing Values?")
    st.write(df.isnull().sum())
    st.markdown("Basic Stats")
    st.write(df.describe())
except:
    st.error("Please Input a Dataset")
st.write("---")

st.subheader("Exploratory Data Analysis")
x = st.radio("Show Pairplot?",["No","Yes"])
st.warning("Not Recomended when dataset has too many columns")
if x == "No":
    pass
elif x =="Yes":
    st.markdown("Correlation")
    sns.pairplot(df)
    plt.tight_layout()
    st.pyplot()

st.markdown("Jointplot")
try:
    col_1 = 0
    col_2 = 0
    col_1,col_2  = st.multiselect("Choose 2 Desired Columns to plot",[x for x in cols])
    sns.jointplot(x = col_1, y = col_2,data = df)
    plt.tight_layout()
    st.pyplot()
except:
    pass

st.header("Train Test Split")
with st.echo():
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

try:
    y = df[st.selectbox("Choose Dependent Variable (Y)",[col for col in cols])]
    independents = st.multiselect("Choose Independent Variables",[col for col in cols])
    x = df[[str for str in independents]]
    x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.3,random_state = 101)
    lm = LinearRegression()
    lm.fit(x_train,y_train)
    st.success("Fit successfully")
    st.write('Coefficients: \n', pd.DataFrame(lm.coef_,index=[str for str in independents],columns=["Independent Variables"]))
    st.write("Intercept:",lm.intercept_)
except:
    st.error("Fitting Error")
    pass

st.write("---")
st.header("Predictions + Evaluations")
try:
    with st.echo():
        pred = lm.predict(x_test)
    st.markdown("Actual vs Predicitions")
    plt.scatter(y_test,pred)
    plt.xlabel('Y Test')
    plt.ylabel('Predicted Y')
    st.pyplot()

    st.subheader("Evaluation Scores")

except:
    pass
try:
    with st.echo():
        from sklearn.metrics import mean_absolute_error,mean_squared_error

    st.write('MAE:', mean_absolute_error(y_test, pred))
    st.write('MSE:', mean_squared_error(y_test, pred))
    st.write('RMSE:', np.sqrt(mean_squared_error(y_test, pred)))

    st.subheader("Residuals")
    plt.title("Prediction Error")
    sns.distplot((y_test-pred),bins=50)
    plt.tight_layout()
    st.pyplot()
except:
    pass
