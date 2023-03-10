import streamlit  as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from PIL import Image
key_image=Image.open('lasso.png')
st.set_page_config(
    layout='wide'
)
st.header('Lasso回归和岭回归')
with st.expander('模型释义'):
    st.image(key_image)
st.subheader('Lasso回归和岭回归实践')
result=st.file_uploader('请上传已清洗完成的数据，包含因变量和自变量，csv格式',type=['csv'])
if result is not None:
    data = pd.read_csv(result)
    st.write('请选择各项参数')
    y = st.selectbox('自变量y', list(data.columns))
    x= st.multiselect('因变量x',list(data.columns))
    test_size = st.number_input('训练数据与测试数据分割比例')
    alphas = st.number_input('设置超参数')
    model=st.selectbox('选择模型',['Lasso回归','岭回归'])
    if model=='Lasso回归':
       if st.button('确认参数并开始训练模型'):
            X = pd.DataFrame(data[x])
            Y = data[y]
            X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=100, test_size=test_size)
            # 数据标准化处理
            ss_X = StandardScaler()
            X_train = ss_X.fit_transform(X_train)
            X_test = ss_X.transform(X_test)
            lasso=Lasso(alpha=alphas)
            lasso.fit(X_train, y_train)
            y_pre = lasso.predict(X_test)
            st.write('模型结果')
            st.write('常数项：' + str(round(lasso.intercept_, 2)))
            ds = pd.DataFrame({'各项特征': list(X), '权重系数': list(lasso.coef_)})
            st.dataframe(ds)
            c = round(r2_score(y_test, y_pre), 2)
            st.write(f'模型评分：R2={c}')

    if model == '岭回归':
        if st.button('确认参数并开始训练模型'):
            X = pd.DataFrame(data[x])
            Y = data[y]
            X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=100, test_size=test_size)
            # 数据标准化处理
            ss_X = StandardScaler()
            X_train = ss_X.fit_transform(X_train)
            X_test = ss_X.transform(X_test)
            ridge=Ridge(alpha=alphas)
            ridge.fit(X_train, y_train)
            y_pre = ridge.predict(X_test)
            st.write('模型结果')
            st.write('常数项：' + str(round(ridge.intercept_, 2)))
            ds = pd.DataFrame({'各项特征': list(X), '权重系数': list(ridge.coef_)})
            st.dataframe(ds)
            c = round(r2_score(y_test, y_pre), 2)
            st.write(f'模型评分：R2={c}')
