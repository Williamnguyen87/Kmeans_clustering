import numpy as np
import pandas as pd
import io
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import squarify
from datetime import datetime

from sklearn.feature_extraction.text import CountVectorizer 
from sklearn import metrics
from io import StringIO
from sklearn.cluster import KMeans

#------------------
# Hiệu ứng
st.balloons()

#------------------
# Functions: 
# Function to load data
@st.cache_data
def load_data(file_name):
    data = pd.read_csv(file_name)
    # Convert order_date to datetime type
    data['Date'] = data['Date'].apply(lambda x: pd.to_datetime(x,format='%Y%m%d', errors='coerce'))
    return data

# Function to calculate Recency, Frequency, Monetary
@st.cache_data
def calculate_RFM(dataframe):
    # Convert string to date, get max date of dataframe
    max_date = dataframe['Date'].max().date()
    Recency = lambda x : (max_date - x.max().date()).days
    Frequency  = lambda x: len(x.unique())
    Monetary = lambda x : round(sum(x), 2)
    dataframe_RFM = dataframe.groupby('customer_ID').agg({'Date': Recency,
                                            'Amount': Frequency,  
                                            'Price': Monetary }).reset_index()
    # Rename the columns of dataframe
    dataframe_RFM.columns = ['customer_ID', 'Recency', 'Frequency', 'Monetary']
    return dataframe_RFM

# Function get info of dataframe for streamlit
@st.cache_data
def info_dataframe(dataframe):
    buffer = io.StringIO()
    dataframe.info(buf = buffer)
    s = buffer.getvalue()
    return s

# Function to apply Kmeans clustering
@st.cache_data
def Kmean_model(data):
    sse = {}
    for k in range(1, 10):
        Kmean_model = KMeans(n_clusters=k, random_state=42)
        Kmean_model.fit(data)
        sse[k] = Kmean_model.inertia_
    return Kmean_model

def model(data): # k = 5
    model = KMeans(n_clusters=5, random_state=42)
    model.fit(data)
    model.labels_.shape
    return model

# Function to calculate average values and return the size for each segment
@st.cache_data
def calculate_segment(dataframe, col_cluster):
    rfm_agg = dataframe.groupby(col_cluster).agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count', 'sum']}).round(0)

    rfm_agg.columns = rfm_agg.columns.droplevel()
    rfm_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Quantity','TotalAmount']
    rfm_agg['Percent_Quantity'] = round((rfm_agg['Quantity']/rfm_agg.Quantity.sum())*100, 2)
    rfm_agg['Percent_Amount'] = round((rfm_agg['TotalAmount']/rfm_agg.TotalAmount.sum())*100, 2)
    # Reset the index
    rfm_agg = rfm_agg.reset_index()
    rfm_agg = rfm_agg.sort_values(['MonetaryMean','FrequencyMean', 'RecencyMean'], 
                                    ascending = [False,False,False])
    return rfm_agg

# Function to save results to csv
@st.cache_data
def convert_df(df):
    return df.to_csv(index = False).encode('utf-8')

# Load models 
# Đọc model
#with open('model/KNNmodel.pkl', 'rb') as file:  
#    K_model = pickle.load(file)

#------------------------------------------------
# GUI
menu = ["Business Objective Overview", "Build RFM Analysis Project", "Predict new customers"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Business Objective Overview':
    st.markdown("<h1 style='text-align: center; color: black;'>Customer Segmentation</h1>", unsafe_allow_html=True)   
    st.subheader("Objective Overview")
    st.write(""" From major customers to regular customers who want to exit corporate,
They have different needs and wants. But company want customers to spend more,
From marketing campaigns to programs, new products to customers in different ways.
However, the question is how to come up with the right marketing campaigns for these customer groups?
customers are in demand, thereby increasing the response rate from customers and thereby increasing sales.
The problem is how to accurately segment customers based on historical transaction behavior.
of customers, RFM algorithm will help us solve this problem quickly and efficiently. """)  
    st.write("""###### => Segment/group/cluster of customers (market segmentation is also known as market segmentation)
     is the process of grouping or collecting customers together based on common characteristics. It divides and groups
     customers into subgroups according to psychological, behavioral.
             """)
    st.image("RFM_Model_1.png")
    st.image("RFM_Model_2.png")
    st.write("""#### Phân tích RFM (Recency, Frequency, Monetary): là một kĩ thuật phân khúc 
             khách hàng dựa trên hành vi giao dịch của khách hàng trong quá khứ để nhóm thành các phân khúc.
   
**Dựa trên 3 thông số:**  
- Recency (R): Thời gian giao dịch cuối cùng.  
- Frequency (F): Tổng số lần giao dịch.
- Monetary (M): Tổng số tiền KH đã chi.  

**Lợi ích của phân tích RFM:**
- Tăng tỷ lệ giữ chân khách hàng vì có chiến lược cho từng nhóm KH.
- Tăng doanh thu từ các nhóm KH với chiến dịch Marketing phù hợp. 
             """)

elif choice == 'Build RFM Analysis Project':
    st.subheader("Build Project")
    st.markdown("<h1 style='text-align: center; color: black;'>Capstone Project</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'>Customer Segmentation</h2>", unsafe_allow_html=True)
    
    # Upload file
    st.write("""## Read data""")
    st.write(""" Tải lên dữ liệu transaction data theo định dạng như hình sau:\n
    ['customer_ID', 'Date', 'Amount', 'Price'] """)
    st.image("data_head(1).png")
    uploaded_file = st.file_uploader("Choose a file", type=['csv'])
    if uploaded_file is not None:
        data = load_data('data/CDNOW_master1.csv')
        st.dataframe(data.head(5))
    
    # Check and EDA Data     
    st.write("""## Information of data""")
    st.code('Transactions timeframe from {} to {}'.format(data['Date'].min(), data['Date'].max()))
    st.code('{:,} transactions don\'t have a customer id'.format(data[data.customer_ID.isnull()].shape[0]))
    st.code('{:,} unique customer_id'.format(len(data.customer_ID.unique())))
    
    # Calculate RFM
    data_RFM = calculate_RFM(data)
    st.write(""" ## Calculate RFM for each customers """)
    st.write('Dữ liệu sau khi tính toán RFM cho',len(data_RFM),'khách hàng'
             ,(data_RFM.head(5)))
    st.write('Thông tin của dữ liệu')
    st.text(info_dataframe(data_RFM))
    
    # Build Model with Kmeans
    st.write("## Customer Segmentation")
    st.write("## Hierarchical")
    st.write("Áp dụng thuật toán Kmeans với số lượng Cluster mong muốn là 5")
    data_RFM["RFM_Cluster"] = Kmean_model(data_RFM)
    # st.write("Dataframe:",data_RFM)
    rfm_hc_agg = calculate_segment(data_RFM,'RFM_Cluster')
    st.code(model(data_RFM))
    st.write(rfm_hc_agg,'Kết quả phân cụm theo thuật toán Kmeans với số lượng nhóm là 5:')
    st.write("""Dựa trên kết quả phân cụm của thuật toán Kmeans, 
             dữ liệu được phân ra các nhóm (từ trên xuống):  
        - Nhóm 1: Các khách hàng chi tiêu nhiều và thường xuyên, với lượng chi tiêu lớn  
        - Nhóm 2: Các khách hàng chi tiêu và mức độ thường xuyên nằm ở mức khá  
        - Nhóm 3: Các khách hàng chi tiêu ít và không thường xuyên  
        - Nhóm 4: Các khách hàng chi tiếu ít và đã lâu không phát sinh giao dịch.
        - Nhóm 5: Các khách hàng rời bỏ và chỉ nhận KM không mua SP.""")
    st.image('Elbow_method.png')
    st.image('Kmeans_cluster.png')

elif choice == 'Predict new customers':
    st.subheader("Dự đoán khách hàng mới bằng KMeans")
    current_labels = ['STARS','BIG SPENDER','REGULAR','RISK', 'LOST']
    st.subheader("Select data")
    flag = False
    lines = None
    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
    if type=="Upload":
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1, header=None)
            st.dataframe(lines)
            # st.write(lines.columns)
            lines = lines[0]     
            flag = True       
    if type=="Input":        
        email = st.text_area(label="Input your content:")
        if email!="":
            lines = np.array([email])
            flag = True
    
    #if flag:
        #st.write("Content:")
        #if len(lines)>0:
            #st.code(lines)                
            #pre_new = K_model.predict(x_new)       
            #st.code("New predictions (0: Ham, 1: Spam): " + str(y_pred_new))
    

