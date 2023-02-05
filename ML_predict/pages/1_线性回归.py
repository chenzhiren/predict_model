import streamlit  as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from PIL import Image
key_image=Image.open('.\logit.png')
st.set_page_config(
    layout='wide'
)
st.header('线性回归模型')
with st.expander('模型释义'):
    st.image(key_image)
lanmu=st.sidebar.radio('类型',('一元线性回归','多元线性回归'))
if lanmu=='一元线性回归':
    st.subheader('一元线性回归实践')
    result=st.file_uploader('请上传已清洗完成的数据，csv格式',type=['csv'])
    if result is not None:
        data=pd.read_csv(result,skiprows=1)
        st.write('请选择各项参数')
        y=st.selectbox('自变量y',list(data.columns))
        x=st.selectbox('因变量x',list(data.columns))
        test_size=st.number_input('训练数据与测试数据分割比例')
        if st.button('确认参数并开始训练模型'):
           X=pd.DataFrame(data[x])
           Y=data[y]
           X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=100, test_size=test_size)
           # 数据标准化处理
           ss_X = StandardScaler()
           X_train = ss_X.fit_transform(X_train)
           X_test = ss_X.transform(X_test)
           lr = LinearRegression()
           lr.fit(X_train, y_train)
           b=round(lr.intercept_,2)
           a=round(lr.coef_[0],2)
           st.write('模型结果：'+f'y={a}x+{b}')
           y_pre=lr.predict(X_test)
           c=round(r2_score(y_test,y_pre),2)
           st.write(f'模型评分：R2={c}')

if lanmu=='多元线性回归':
    st.subheader('多元线性回归实践')
    result = st.file_uploader('请上传已清洗完成的数据，csv格式', type=['csv'])
    if result is not None:
        data=pd.read_csv(result,skiprows=1)
        st.write('请选择各项参数')
        y=st.selectbox('自变量y',list(data.columns))
        x= st.multiselect('因变量x',list(data.columns))
        test_size = st.number_input('训练数据与测试数据分割比例')
        if st.button('确认参数并开始训练模型'):
           X=pd.DataFrame(data[x])
           Y = data[y]
           X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=100, test_size=test_size)
           # 数据标准化处理
           ss_X = StandardScaler()
           X_train = ss_X.fit_transform(X_train)
           X_test = ss_X.transform(X_test)
           lr = LinearRegression()
           lr.fit(X_train, y_train)
           st.write('模型结果')
           st.write('常数项：'+str(round(lr.intercept_,2)))
           ds=pd.DataFrame({'各项特征':list(X),'权重系数':list(lr.coef_)})
           st.dataframe(ds)
           y_pre = lr.predict(X_test)
           c = round(r2_score(y_test, y_pre), 2)
           st.write(f'模型评分：R2={c}')
