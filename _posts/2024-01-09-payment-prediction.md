---
title: "Default payment prediction project!"
date: 2023-01-09T15:34:30-04:00
categories:
  - blog
tags:
  - notebook
description: "Abcxyz"
---

<img src="https://cdn.dribbble.com/users/957410/screenshots/3226085/dribbble-gif.gif"/>


[*Đường dẫn tới notebook*](https://github.com/lenlenlim/portfolio/blob/master/notebooks/payment_prediction.ipynb)

## **Mục lục**:
1. Overview
2. Importing and Loading Dataset
3. Data cleaning
  * Handling missing values
  * Reconstructing Dataset
4. Exploratory Data Analysis
  * Data preparation
  * Categorical features
  * Numerical variables
5. Building model
  * Data Preparation
  * Modeling without imbalance resolve
  * Modeling with resampled data
  * Hyperparameters tuning
  * Retraining and Predicting


---



### **Đặt vấn đề**
Dataset "credit_card_clients.xls" là bộ dữ liệu được nhóm tác giả lựa chọn cho dự án. Bộ dữ liệu này chứa thông tin liên quan đến thanh toán và các đặc điểm của hơn 30,000 khách hàng trong khoảng thời gian từ tháng 4 đến tháng 9 của năm 2005. Nhiệm vụ bao gồm khám phá tập dữ liệu và tạo một mô hình đơn giản để phục vụ cho mục tiêu của dự án là dự đoán khả năng thanh toán khoản vay của khách hàng trong tháng tiếp theo (tháng 10/2005). 

Để thực hiện yêu cầu của đề bài, nhóm tác giả tiến hành xử lý bộ dữ liệu "credit_card_clients.xls" theo trình tự sau đây:

1. Tìm hiểu bộ dữ liệu.
2. Importing and Loading Data
3. Data Cleaning
4. Exploratory Data Analysis
5. Build Model (nhóm xây dựng model trong trường hợp dữ liệu imbalance và cả trường hợp resample dữ liệu)


---


### **Giải thích thuật ngữ**

Thuật ngữ "default payment" được hiểu là "không có khả năng thanh toán đúng hạn" hoặc "vi phạm nghĩa vụ thanh toán". Đây là tình trạng khi một người không thực hiện thanh toán đúng hạn theo thỏa thuận hoặc hợp đồng đã ký kết, dẫn đến việc phải trả một khoản phạt hoặc các khoản lãi suất phát sinh. Nếu trường hợp vi phạm này kéo dài thì có thể dẫn đến những hậu quả nghiêm trọng như mất tín nhiệm tại các tổ chức tín dụng và tăng nguy cơ nợ xấu. Như vậy, biến `default payment next month` với 2 giá trị (1,0) được hiểu là: 
  * 1: khách hàng không có khả năng thanh toán vào tháng tiếp theo 
  * 0: khách hàng có khả năng thanh toán vào tháng tiếp theo.


---



# **1. Overview**

Bộ dữ liệu này chứa thông tin về các khoản thanh toán mặc định, các yếu tố nhân khẩu học, dữ liệu tín dụng, lịch sử thanh toán và bảng sao kê hóa đơn của các khách hàng sử dụng thẻ tín dụng ở Heaven từ tháng 4 năm 2005 đến tháng 9 năm 2005.

Có tổng cộng 25 thuộc tính:

* `ID`: ID của mỗi khách hàng

* `LIMIT_BAL`: Số tiền tín dụng được cấp bằng Đài tệ (bao gồm cá nhân và gia đình/thẻ tín dụng phụ) 

* `SEX`: Giới tính 
  - 1 = male
  - 2 = female

* `EDUCATION`: Trình độ học vấn của khách hàng
  - 1 = graduate school 
  - 2 = university
  - 3 = high school 
  - 4 = others
  - 5 = unknown
  - 6 = unknown

* `MARRIAGE`: Tình trạng hôn nhân của khách hàng 
  - 1 = married
  - 2 = single 
  - 3 = others

* `AGE`: Tuổi

* `PAY_0`: Tình trạng trả nợ tháng 9/2005 
  - -1 = trả đúng hạn
  - 1 = trả chậm một tháng
  - 2 = trả chậm hai tháng
  -  … 
  - 8 = trả chậm tám tháng
  - 9 = trả chậm từ chín tháng trở lên

* `PAY_2`: Tình trạng trả nợ tháng 8/2005 (mô tả thành phần như trên)

* `PAY_3`: Tình trạng trả nợ tháng 7/2005 (mô tả thành phần như trên)

* `PAY_4`: Tình trạng trả nợ tháng 6/2005 (mô tả thành phần như trên)

* `PAY_5`: Tình trạng trả nợ tháng 5/2005 (mô tả thành phần như trên)

* `PAY_6`: Tình trạng trả nợ tháng 4/2005 (mô tả thành phần như trên)

* `BILL_AMT1`: Tổng sao kê hóa đơn tháng 9/2005 (Đài tệ)

* `BILL_AMT2`: Tổng sao kê hóa đơn tháng 8/2005 (Đài tệ)

* `BILL_AMT3`: Tổng sao kê hóa đơn tháng 7/2005 (Đài tệ)

* `BILL_AMT4`: Tổng sao kê hóa đơn tháng 6/2005 (Đài tệ)

* `BILL_AMT5`: Tổng sao kê hóa đơn tháng 5/2005 (Đài tệ)

* `BILL_AMT6`: Tổng sao kê hóa đơn tháng 3/2005 (Đài tệ)

* `PAY_AMT1`: Số tiền đã trả trước trong tháng 9/2005 (Đài tệ)

* `PAY_AMT2`: Số tiền đã trả trước trong tháng 8/2005 (Đài tệ)

* `PAY_AMT3`: Số tiền đã trả trước trong tháng 7/2005 (Đài tệ)

* `PAY_AMT4`: Số tiền đã trả trước trong tháng 6/2005 (Đài tệ)

* `PAY_AMT5`: Số tiền đã trả trước trong tháng 5/2005 (Đài tệ)

* `PAY_AMT6`: Số tiền đã trả trước trong tháng 4/2005 (Đài tệ)

* `default.payment.next.month`: Khách hàng không có khả năng thanh toán khoản nợ vào tháng 10/2005 
  - 1 = có
  - 0 = không

# **2. Importing and Loading Dataset**


## 2.1. Import thư viện


```python
import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.under_sampling import NearMiss
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_recall_curve, precision_score,
                             average_precision_score, confusion_matrix,
                             recall_score, roc_auc_score, roc_curve, auc)
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV,
                                     ShuffleSplit, StratifiedKFold,
                                     StratifiedShuffleSplit, cross_val_predict,
                                     cross_val_score, learning_curve,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
```

## 2.2. Load dataset


```python
# Đọc tập dữ liệu
data = pd.read_excel('https://github.com/trangmx/fdc104/blob/main/datasets/credit_card/credit_card_clients.xls?raw=true', index_col = 0, header=1)

# Hiển thị 5 dòng đầu
data.head()
```





  <div id="df-fa1b866e-220d-426b-860d-32a14f3084ed">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>PAY_5</th>
      <th>PAY_6</th>
      <th>BILL_AMT1</th>
      <th>BILL_AMT2</th>
      <th>BILL_AMT3</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default payment next month</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>20000</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>24</td>
      <td>2</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>-2</td>
      <td>-2</td>
      <td>3913</td>
      <td>3102</td>
      <td>689</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>689</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>120000</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>26</td>
      <td>-1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2682</td>
      <td>1725</td>
      <td>2682</td>
      <td>3272</td>
      <td>3455</td>
      <td>3261</td>
      <td>0</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>0</td>
      <td>2000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>90000</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>29239</td>
      <td>14027</td>
      <td>13559</td>
      <td>14331</td>
      <td>14948</td>
      <td>15549</td>
      <td>1518</td>
      <td>1500</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>5000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50000</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>46990</td>
      <td>48233</td>
      <td>49291</td>
      <td>28314</td>
      <td>28959</td>
      <td>29547</td>
      <td>2000</td>
      <td>2019</td>
      <td>1200</td>
      <td>1100</td>
      <td>1069</td>
      <td>1000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>50000</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>57</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8617</td>
      <td>5670</td>
      <td>35835</td>
      <td>20940</td>
      <td>19146</td>
      <td>19131</td>
      <td>2000</td>
      <td>36681</td>
      <td>10000</td>
      <td>9000</td>
      <td>689</td>
      <td>679</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-fa1b866e-220d-426b-860d-32a14f3084ed')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-fa1b866e-220d-426b-860d-32a14f3084ed button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-fa1b866e-220d-426b-860d-32a14f3084ed');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
#Thông tin về tập dữ liệu
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 30000 entries, 1 to 30000
    Data columns (total 24 columns):
     #   Column                      Non-Null Count  Dtype
    ---  ------                      --------------  -----
     0   LIMIT_BAL                   30000 non-null  int64
     1   SEX                         30000 non-null  int64
     2   EDUCATION                   30000 non-null  int64
     3   MARRIAGE                    30000 non-null  int64
     4   AGE                         30000 non-null  int64
     5   PAY_0                       30000 non-null  int64
     6   PAY_2                       30000 non-null  int64
     7   PAY_3                       30000 non-null  int64
     8   PAY_4                       30000 non-null  int64
     9   PAY_5                       30000 non-null  int64
     10  PAY_6                       30000 non-null  int64
     11  BILL_AMT1                   30000 non-null  int64
     12  BILL_AMT2                   30000 non-null  int64
     13  BILL_AMT3                   30000 non-null  int64
     14  BILL_AMT4                   30000 non-null  int64
     15  BILL_AMT5                   30000 non-null  int64
     16  BILL_AMT6                   30000 non-null  int64
     17  PAY_AMT1                    30000 non-null  int64
     18  PAY_AMT2                    30000 non-null  int64
     19  PAY_AMT3                    30000 non-null  int64
     20  PAY_AMT4                    30000 non-null  int64
     21  PAY_AMT5                    30000 non-null  int64
     22  PAY_AMT6                    30000 non-null  int64
     23  default payment next month  30000 non-null  int64
    dtypes: int64(24)
    memory usage: 5.7 MB
    

Tập dữ liệu gồm 24 trường biến và 3000 quan sát, các dữ liệu ở dạng interger

## 2.3. Options hiển thị Pandas


```python
pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 40)
pd.set_option('display.width', 40) 
```

# **3. Data Cleaning**

## **3.1. Handling missing values**


```python
# Kiểm tra tổng giá trị NULL cho từng thuộc tính
data.isnull().sum()
```




    LIMIT_BAL                     0
    SEX                           0
    EDUCATION                     0
    MARRIAGE                      0
    AGE                           0
    PAY_0                         0
    PAY_2                         0
    PAY_3                         0
    PAY_4                         0
    PAY_5                         0
    PAY_6                         0
    BILL_AMT1                     0
    BILL_AMT2                     0
    BILL_AMT3                     0
    BILL_AMT4                     0
    BILL_AMT5                     0
    BILL_AMT6                     0
    PAY_AMT1                      0
    PAY_AMT2                      0
    PAY_AMT3                      0
    PAY_AMT4                      0
    PAY_AMT5                      0
    PAY_AMT6                      0
    default payment next month    0
    dtype: int64



Kết quả cho thấy, các trường dữ liệu trong dataset không có giá trị null.

## <b>3.2. Reconstructing Dataset</b>

### Kiểm tra dạng dữ liệu của các biến


```python
# Kiểm tra dạng dữ liệu của các biến
data.dtypes
```




    LIMIT_BAL                     int64
    SEX                           int64
    EDUCATION                     int64
    MARRIAGE                      int64
    AGE                           int64
    PAY_0                         int64
    PAY_2                         int64
    PAY_3                         int64
    PAY_4                         int64
    PAY_5                         int64
    PAY_6                         int64
    BILL_AMT1                     int64
    BILL_AMT2                     int64
    BILL_AMT3                     int64
    BILL_AMT4                     int64
    BILL_AMT5                     int64
    BILL_AMT6                     int64
    PAY_AMT1                      int64
    PAY_AMT2                      int64
    PAY_AMT3                      int64
    PAY_AMT4                      int64
    PAY_AMT5                      int64
    PAY_AMT6                      int64
    default payment next month    int64
    dtype: object



Các biến trong bộ dữ liệu đều ở dạng `int64`

### Đổi tên biến

Để thuận tiện cho việc xử lý dữ liệu ở các bước tiếp theo, nhóm tiến hành thay đổi tên của các cột "PAY_0" và "default payment next month"


```python
# Đổi tên biến "PAY_0" thành "PAY_1"
data.rename(columns = {'PAY_0':'PAY_1'}, inplace = True)
# Đổi tên biến "default payment next month" thành "default"
data = data.rename(columns={'default payment next month': 'default'})

data.head()
```





  <div id="df-66e79d13-49a4-4d6f-a776-fb73f28308db">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_1</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>PAY_5</th>
      <th>PAY_6</th>
      <th>BILL_AMT1</th>
      <th>BILL_AMT2</th>
      <th>BILL_AMT3</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>20000</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>24</td>
      <td>2</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>-2</td>
      <td>-2</td>
      <td>3913</td>
      <td>3102</td>
      <td>689</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>689</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>120000</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>26</td>
      <td>-1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2682</td>
      <td>1725</td>
      <td>2682</td>
      <td>3272</td>
      <td>3455</td>
      <td>3261</td>
      <td>0</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>0</td>
      <td>2000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>90000</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>29239</td>
      <td>14027</td>
      <td>13559</td>
      <td>14331</td>
      <td>14948</td>
      <td>15549</td>
      <td>1518</td>
      <td>1500</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>5000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50000</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>46990</td>
      <td>48233</td>
      <td>49291</td>
      <td>28314</td>
      <td>28959</td>
      <td>29547</td>
      <td>2000</td>
      <td>2019</td>
      <td>1200</td>
      <td>1100</td>
      <td>1069</td>
      <td>1000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>50000</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>57</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8617</td>
      <td>5670</td>
      <td>35835</td>
      <td>20940</td>
      <td>19146</td>
      <td>19131</td>
      <td>2000</td>
      <td>36681</td>
      <td>10000</td>
      <td>9000</td>
      <td>689</td>
      <td>679</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-66e79d13-49a4-4d6f-a776-fb73f28308db')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-66e79d13-49a4-4d6f-a776-fb73f28308db button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-66e79d13-49a4-4d6f-a776-fb73f28308db');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




### Kiểm tra giá trị các trường dữ liệu

Tại bước này, nhóm sẽ tiến hành kiểm tra miền giá trị của các trường dữ liệu để so sánh bộ dữ liệu với phần description được cung cấp sẵn.


```python
# Kiểm tra giá trị các trường dữ liệu
for col in data.columns:
  range = data[col].unique()
  range = np.sort(range)
  print(f'{col} ({len(range)} distinct values): {range} \n')
```

    LIMIT_BAL (81 distinct values): [  10000   16000   20000   30000   40000   50000   60000   70000   80000
       90000  100000  110000  120000  130000  140000  150000  160000  170000
      180000  190000  200000  210000  220000  230000  240000  250000  260000
      270000  280000  290000  300000  310000  320000  327680  330000  340000
      350000  360000  370000  380000  390000  400000  410000  420000  430000
      440000  450000  460000  470000  480000  490000  500000  510000  520000
      530000  540000  550000  560000  570000  580000  590000  600000  610000
      620000  630000  640000  650000  660000  670000  680000  690000  700000
      710000  720000  730000  740000  750000  760000  780000  800000 1000000] 
    
    SEX (2 distinct values): [1 2] 
    
    EDUCATION (7 distinct values): [0 1 2 3 4 5 6] 
    
    MARRIAGE (4 distinct values): [0 1 2 3] 
    
    AGE (56 distinct values): [21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44
     45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68
     69 70 71 72 73 74 75 79] 
    
    PAY_1 (11 distinct values): [-2 -1  0  1  2  3  4  5  6  7  8] 
    
    PAY_2 (11 distinct values): [-2 -1  0  1  2  3  4  5  6  7  8] 
    
    PAY_3 (11 distinct values): [-2 -1  0  1  2  3  4  5  6  7  8] 
    
    PAY_4 (11 distinct values): [-2 -1  0  1  2  3  4  5  6  7  8] 
    
    PAY_5 (10 distinct values): [-2 -1  0  2  3  4  5  6  7  8] 
    
    PAY_6 (10 distinct values): [-2 -1  0  2  3  4  5  6  7  8] 
    
    BILL_AMT1 (22723 distinct values): [-165580 -154973  -15308 ...  653062  746814  964511] 
    
    BILL_AMT2 (22346 distinct values): [-69777 -67526 -33350 ... 671563 743970 983931] 
    
    BILL_AMT3 (22026 distinct values): [-157264  -61506  -46127 ...  693131  855086 1664089] 
    
    BILL_AMT4 (21548 distinct values): [-170000  -81334  -65167 ...  628699  706864  891586] 
    
    BILL_AMT5 (21010 distinct values): [-81334 -61372 -53007 ... 587067 823540 927171] 
    
    BILL_AMT6 (20604 distinct values): [-339603 -209051 -150953 ...  568638  699944  961664] 
    
    PAY_AMT1 (7943 distinct values): [     0      1      2 ... 493358 505000 873552] 
    
    PAY_AMT2 (7899 distinct values): [      0       1       2 ... 1215471 1227082 1684259] 
    
    PAY_AMT3 (7518 distinct values): [     0      1      2 ... 508229 889043 896040] 
    
    PAY_AMT4 (6937 distinct values): [     0      1      2 ... 497000 528897 621000] 
    
    PAY_AMT5 (6897 distinct values): [     0      1      2 ... 388071 417990 426529] 
    
    PAY_AMT6 (6939 distinct values): [     0      1      2 ... 443001 527143 528666] 
    
    default (2 distinct values): [0 1] 
    
    

Kết quả ở trên cho ta thấy được rằng, có một sự khác nhau về giá trị các biến của các trường ```EDUCATION```, ```MARRIAGE```, ```PAY_1```, ```PAY_2```,..., ```PAY_6``` so với data description.
Cụ thể là:
- ```EDUCATION```có giá trị [0,6] thay vì [1, 6] như trong mô tả
- ```MARRIAGE```có giá trị từ [0,3] thay vì [1,3] như trong mô tả
- Các trường ```PAY_1```, ```PAY_2```,..., ```PAY_6``` có giá trị [-2,8] khác với [-1,9] như trong mô tả

Do đó, nhóm đề xuất phương án giải quyết như sau:


Đối với `EDUCATION` & `MARRIAGE` 

Đầu tiên, hãy kiểm tra phân bố giá trị của 2 thuộc tính này


```python
df = data[['EDUCATION','MARRIAGE']] 
# Đồ thị phân bổ giá trị của 2 thuốc tính EDUCATION và MARRIAGE
df.plot(kind = 'hist', 
        figsize = (14,4), 
        subplots = True, 
        ylabel = 'Count', 
        colormap = 'tab20b',
        layout=(1, len(df.columns)))

plt.show()
```


    
[png](/portfolio/assets/images/payment_prediction_files/payment_prediction_31_0.png)
    


* ```EDUCATION```: 

Vì tần suất của những giá trị không xác định (0,5,6) là rất nhỏ, vậy nên nhóm sẽ đổi những giá trị này về 4 (others). Khi đó, miền giá trị của thuộc tính này sẽ là:
```
{ 
    1:'graduate school',
    2:'university',
    3:'high school',
    4:'others'
}
```
* ```MARRIAGE```: 

Tương tự, tần suất xuất hiện của giá trị 0 rất nhỏ, do đó nhóm quyết định biến đổi các giá trị bằng 0 thành 3 (others). Khi đó, miền giá trị của thuộc tính này sẽ là:
```
{ 
    1:'married',
    2:'single',
    3:'others'
}
```


```python
# Thay các giá trị 0,5,6 ở biến EDUCATION bằng 4
data['EDUCATION'].replace([0,5,6], 4, inplace=True)
```


```python
# Thay giá trị 0 bằng 3 ở biến MARRIAGE
data['MARRIAGE'].replace(0, 3, inplace=True)
```



Đối với `PAY_1`, `PAY_2`, ... , `PAY_6`

Đầu tiên, hãy kiểm tra phân bố giá trị của những thuộc tính này


```python
# Vẽ biểu đồ phân bổ giá trị của các thuộc tính PAY_1, PAY_2,..., PAY_6
my_lst = [f"PAY_{i}" for i in np.arange(1,7)]
df_pay = data[my_lst]

df_pay.plot(kind='hist',
            subplots=True,
            figsize = (18,6),
            xlim = (-2,9),
            xticks = np.arange(-2,9),
            ylabel='Count',
            colormap='tab20b',
            layout=(2,3))
plt.show()
```


    
[png](/portfolio/assets/images/payment_prediction_files/payment_prediction_36_0.png)
    


Từ những biểu đồ phân bổ kể trên, ta có thể thấy rằng ở tất các các tháng, số lượng người thanh toán tín dụng đúng hạn chiếm phần lớn tỷ lệ khi so sánh với các giá trị còn lại. Do đó, ta có thể chuyển giá trị (-2,-1) thành 0 


```python
# Thay thế các giá trị(-2,-1) thành 0 ở các trường PAY_1, PAY_2, ..., PAY_6
for col in my_lst: 
  data[col].replace([-2,-1,0], 0, inplace=True)
```

### Đổi đơn vị dữ liệu

Các dữ liệu về tiền tệ có đơn vị là Đài tệ, để việc sử dụng notebook linh hoạt hơn, nhóm quyết định đổi sang đơn vị USD


```python
# Đổi đơn vị giá trị các trường từ BILL_AMT1 tới BILL_AMT6
for i in np.arange(data.columns.get_loc('BILL_AMT1'), data.columns.get_loc('default')):
  data.iloc[:,i] = data.iloc[:,i] * 0.032 
```


```python
# Đổi đơn vị trường LIMIT_BAL
data['LIMIT_BAL'] = data['LIMIT_BAL'] * 0.032
```

# **4. Explorary Data Analysis**



## 4.1. Data describe


```python
data.describe()
```





  <div id="df-97fc30db-6c21-4a47-baee-f5f825986431">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_1</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>PAY_5</th>
      <th>PAY_6</th>
      <th>BILL_AMT1</th>
      <th>BILL_AMT2</th>
      <th>BILL_AMT3</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.00000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5359.498325</td>
      <td>1.603733</td>
      <td>1.842267</td>
      <td>1.557267</td>
      <td>35.485500</td>
      <td>0.356767</td>
      <td>0.320033</td>
      <td>0.304067</td>
      <td>0.258767</td>
      <td>0.22150</td>
      <td>0.226567</td>
      <td>1639.146589</td>
      <td>1573.730405</td>
      <td>1504.420954</td>
      <td>1384.414367</td>
      <td>1289.964831</td>
      <td>1243.896333</td>
      <td>181.234576</td>
      <td>189.477232</td>
      <td>167.221808</td>
      <td>154.434460</td>
      <td>153.580404</td>
      <td>166.896082</td>
      <td>0.221200</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4151.925170</td>
      <td>0.489129</td>
      <td>0.744494</td>
      <td>0.521405</td>
      <td>9.217904</td>
      <td>0.760594</td>
      <td>0.801727</td>
      <td>0.790589</td>
      <td>0.761113</td>
      <td>0.71772</td>
      <td>0.715438</td>
      <td>2356.347538</td>
      <td>2277.560601</td>
      <td>2219.180398</td>
      <td>2058.651396</td>
      <td>1945.508985</td>
      <td>1905.731441</td>
      <td>530.024971</td>
      <td>737.307853</td>
      <td>563.422767</td>
      <td>501.317112</td>
      <td>488.905782</td>
      <td>568.878905</td>
      <td>0.415062</td>
    </tr>
    <tr>
      <th>min</th>
      <td>320.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>21.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>-5298.560000</td>
      <td>-2232.864000</td>
      <td>-5032.448000</td>
      <td>-5440.000000</td>
      <td>-2602.688000</td>
      <td>-10867.296000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1600.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>113.880000</td>
      <td>95.512000</td>
      <td>85.320000</td>
      <td>74.456000</td>
      <td>56.416000</td>
      <td>40.192000</td>
      <td>32.000000</td>
      <td>26.656000</td>
      <td>12.480000</td>
      <td>9.472000</td>
      <td>8.080000</td>
      <td>3.768000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4480.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>34.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>716.208000</td>
      <td>678.400000</td>
      <td>642.832000</td>
      <td>609.664000</td>
      <td>579.344000</td>
      <td>546.272000</td>
      <td>67.200000</td>
      <td>64.288000</td>
      <td>57.600000</td>
      <td>48.000000</td>
      <td>48.000000</td>
      <td>48.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7680.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>41.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>2146.912000</td>
      <td>2048.200000</td>
      <td>1925.272000</td>
      <td>1744.192000</td>
      <td>1606.096000</td>
      <td>1574.344000</td>
      <td>160.192000</td>
      <td>160.000000</td>
      <td>144.160000</td>
      <td>128.424000</td>
      <td>129.008000</td>
      <td>128.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>32000.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>79.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>8.00000</td>
      <td>8.000000</td>
      <td>30864.352000</td>
      <td>31485.792000</td>
      <td>53250.848000</td>
      <td>28530.752000</td>
      <td>29669.472000</td>
      <td>30773.248000</td>
      <td>27953.664000</td>
      <td>53896.288000</td>
      <td>28673.280000</td>
      <td>19872.000000</td>
      <td>13648.928000</td>
      <td>16917.312000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-97fc30db-6c21-4a47-baee-f5f825986431')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-97fc30db-6c21-4a47-baee-f5f825986431 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-97fc30db-6c21-4a47-baee-f5f825986431');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




Từ bảng kết quả ta quan sát thấy:
- Biến LIMIT_BAL có giá trị [320, 32000], giá trị min và max có sự chênh lệch lớn.
- Biến SEX có 2 giá trị là 1-male và 2-female trong đó phần lớn các quan sát có giá trị là 2
- Biến EDUCATION có giá trị nhỏ nhất là 1 và lớn nhất là 4, trung vị là 2 và các giá trị chủ yếu = 2 (University)
- Biến MARRIAGE có giá trị [1,3], phần lớn các quan sát có giá trị là 2 (single)
- Biến AGE có giá trị trong khoảng từ 21 đến 79, phần lớn dữ liệu có giá trị < 50, độ tuổi trung bình đạt 35,5
- Các biến PAY_i (với i = [1,6]) đều có giá trị từ 0 đến 8 và phần lớn các quan sát có giá trị bằng 0
- Các biến BILL_AMTi (i=[1,6]) có khoảng giá trị rất lớn, các quan sát có thể có giá trị âm hoặc dương
- Các biến PAY_AMTi (với i =[1,6]) đều có giá trị nhỏ nhất bằng 0, giá trị lớn nhất chênh lệch tương đối lớn so với giá trị min và đều > 13000
- Biến default có 2 giá trị là 0 và 1, dễ dàng quan sát thấy phần lớn các quan sát có giá trị = 0

## 4.2. Categorical features

#### `Marriage`


```python
# Đếm số giá trị khác biệt của biến MARRIAGE
marriage_count = data['MARRIAGE'].value_counts().reset_index().rename(columns={'index':'index','MARRIAGE':'count'})
```


```python
# đổi tên các trường 
marriage_count['index'][1]= 'Marriaged'
marriage_count['index'][0]= 'Single'
marriage_count['index'][2]= 'Others'
```


```python
# biểu đồ phân loại theo tình trạng hôn nhân 
plt.figure(figsize = (4,4))
sns.barplot(x=marriage_count['index'], y = marriage_count['count'], data=data, palette='plasma').set_title('Marriage Status Distribution of Clients')
plt.xlabel("Marriage Status", fontsize= 12)
plt.ylabel("Count of CLients", fontsize= 12)
plt.show()
```


    
[png](/portfolio/assets/images/payment_prediction_files/payment_prediction_51_0.png)
    


Từ biểu đồ, ta rút ra nhận xét các khách hàng ở tình trạng độc thân chiếm tỷ lệ lớn nhất (khoảng gần 16000 người), khoảng gần 14000 khách hàng đã kết hôn và còn lại một phần vô cùng nhỏ khách hàng có tình trạng hôn nhân là "khác".


```python
# vẽ biểu đồ thể hiện khả năng default đối với từng nhóm khách hàng theo tình trạng hôn nhân 
sns.countplot(data = data, x = 'MARRIAGE', hue="default", palette = 'rocket').set_title('Default of Marriage Status')
plt.xlabel("Marriage Status", fontsize= 12)
plt.ylabel("Count of Clients", fontsize= 12)
plt.xticks([0,1,2],['Married', 'Single', 'Others'])
plt.show()
```


    
[png](/portfolio/assets/images/payment_prediction_files/payment_prediction_53_0.png)
    


Đối với những khách hàng độc thân và đã kết hôn, số lượng khách hàng có default = 1 ở 2 trường hợp này xấp xỉ bằng nhau tuy nhiên khách hàng độc thân có khả năng trả nợ vào tháng tới cao hơn so với khách hàng đã kết hôn.

Khách hàng có default = 1 và default = 0 trong trường hợp tình trạng hôn nhân "khác" không có sự chênh lệch quá lớn về số lượng

#### `Education`


```python
# Đếm tần suất của các giá trị khác biệt của biến EDUCATION
edu_count = data['EDUCATION'].value_counts().reset_index().rename(columns={'index':'index','EDUCATION':'count'})

```


```python
# đổi tên theo thứ tự 
edu_count['index'][0] = 'University'
edu_count['index'][1] = 'Graduate School'
edu_count['index'][2] = 'High school'
edu_count['index'][3] = 'Others'
```


```python
# vẽ biểu đồ thể hiện khả năng default đối với từng trình độ học vấn khác nhau
sns.countplot(data = data, x = 'EDUCATION', hue="default", palette = 'tab10').set_title('Education Status Distribution of Clients')
plt.xlabel("Education Status", fontsize= 12)
plt.ylabel("Count of Clients", fontsize= 12)
plt.xticks([0,1,2,3],['Graduate School','University','High School','Others'])
plt.show()
```


    
[png](/portfolio/assets/images/payment_prediction_files/payment_prediction_58_0.png)
    


Hầu hết tất cả khách hàng đều có trình độ giáo dục tốt. Khách hàng có trình độ đại học chiếm tỷ lệ lớn nhất sau đó là tôt nghiệp cấp ba và đang là học sinh cấp ba, chỉ một lượng nhỏ ở nhóm khác ( có thể là ở trình độ thấp hơn 3 nhóm kể trên).


```python
# vẽ biểu đồ thể hiện khả năng default đối với từng trình độ học vấn khác nhau
sns.countplot(data = data, x = 'EDUCATION', hue="default", palette = 'rocket').set_title('Default of Education Status')
plt.xlabel("Educaiton Status", fontsize= 12)
plt.ylabel("Count of Clients", fontsize= 12)
plt.xticks([0,1,2,3],['Graduate School','University','High School','Others'])
plt.show()
```


    
[png](/portfolio/assets/images/payment_prediction_files/payment_prediction_60_0.png)
    


 Xét phân phối default theo trình độ học vấn, ta có thể thấy số lượng khách hàng default = 0 có xu hướng tăng theo trình độ học vấn (tức là trình độ học vấn càng cao thì càng nhiều khách hàng có default = 0)

#### Gender


```python
# Đếm số lần xuất hiện của các giá trị khác biệt trong biến "SEX"
sex_count = data['SEX'].value_counts().reset_index().rename(columns={'index':'index','SEX':'count'})
```


```python
# Đổi tên cột theo thứ tự
sex_count['index'][1] = 'Male'
sex_count['index'][0] = 'Female'
```


```python
# biểu đồ phân loại giới tính trong bộ dữ liệu
plt.figure(figsize = (3,4))
sns.barplot(x=sex_count['index'], y = sex_count['count'], data=data, palette='husl').set_title('Sex Distribution of Clients')
plt.xlabel("Sex", fontsize= 12)
plt.ylabel("Count of Clients", fontsize= 12)
plt.show()
```


    
[png](/portfolio/assets/images/payment_prediction_files/payment_prediction_65_0.png)
    


Nhìn vào biểu đồ ta có thể thấy lượng khách hàng là nữ gấp lượng khách hàng nam khoảng 1,5 lần


```python
# vẽ biểu đồ thể hiện khả năng default đối với từng giới tính
plt.figure(figsize = (5,5))
sns.countplot(data = data, x = 'SEX', hue="default", palette = 'rocket').set_title('Default of Sex')
plt.xlabel("Sex", fontsize= 12)
plt.ylabel("Count of Clients", fontsize= 12)
plt.xticks([0,1],['Male', 'Female'])
plt.show()
```


    
[png](/portfolio/assets/images/payment_prediction_files/payment_prediction_67_0.png)
    


So sánh tỷ lệ default = 1 và default = 0 ở 2 giới tính nam và nữ, có thể thấy nam giới có khả năng default cao hơn nữ giới.

## 4.3. Numerical variables

#### Age


```python
# vẽ biểu đồ thể hiện phân phối theo độ tuổi
plt.figure(figsize = (10,4))
sns.histplot(data=data, x = 'AGE', kde = True, bins=np.arange(20,81,2))
plt.title("Age distribution")
plt.xlabel("Age", fontsize= 12)
plt.ylabel("Count of clients", fontsize= 12)
plt.show()
```


    
[png](/portfolio/assets/images/payment_prediction_files/payment_prediction_71_0.png)
    


Quan sát biểu đồ, ta thấy khách hàng có độ tuổi tương đối trẻ, phần lớn < 50, đồ thị phân phối lệch sang bên phải.


```python
# Nhóm theo độ tuổi
data['AgeBin'] = pd.cut(data['AGE'],[20, 25, 30, 35, 40, 50, 60, 80])
print(data['AgeBin'].value_counts())
```

    (25, 30]    7142
    (40, 50]    6005
    (30, 35]    5796
    (35, 40]    4917
    (20, 25]    3871
    (50, 60]    1997
    (60, 80]     272
    Name: AgeBin, dtype: int64
    


```python
# show default theo nhóm tuổi
data['default'].groupby(data['AgeBin']).value_counts(normalize = True)
```




    AgeBin    default
    (20, 25]  0          0.733402
              1          0.266598
    (25, 30]  0          0.798516
              1          0.201484
    (30, 35]  0          0.805728
              1          0.194272
    (35, 40]  0          0.783811
              1          0.216189
    (40, 50]  0          0.767027
              1          0.232973
    (50, 60]  0          0.747621
              1          0.252379
    (60, 80]  0          0.731618
              1          0.268382
    Name: default, dtype: float64




```python
# vẽ đồ thị thể hiện default theo nhóm tuổi
plt.figure(figsize=(8,4))


data['AgeBin'] = data['AgeBin'].astype('str')
AgeBin_order = ['(20, 25]', '(25, 30]', '(30, 35]', '(35, 40]', '(40, 50]', '(50, 60]', '(60, 80]']


ax = sns.countplot(data = data, x = 'AgeBin', hue="default", palette = 'rocket', order = AgeBin_order)


plt.xlabel("Age Group", fontsize= 12)
plt.ylabel("Count of Clients", fontsize= 12)
plt.title("Default of Age Group")
plt.ylim(0,8000)


for p in ax.patches:
    ax.annotate(  str(int(p.get_height())), 
                  (p.get_x() + p.get_width() / 2., p.get_height()), 
                  ha='center', 
                  va='center', 
                  xytext=(0, 10), 
                  textcoords='offset points')


plt.show()

```


    
[png](/portfolio/assets/images/payment_prediction_files/payment_prediction_75_0.png)
    


Từ đồ thị, ta rút ra nhận xét như sau:
- Nhìn chung default = 0 có xu hướng giảm theo độ tuổi. Số lượng khách hàng có default = 0  cao nhất ở dộ tuổi 25-30 và thấp nhất ở độ tuổi 60-80
- Số lượng khách hàng có default = 1 ở các nhóm tuổi có giá trị khá tương đồng
- Số lượng khách hàng default cao nhất thuộc độ tuổi 25-30 và thấp nhất ở độ tuổi 60-80. Tuy nhiên nếu so sánh về tỷ lệ default =1 và default =0 ở các nhóm tuổi thi khả năng default thấp nhất là ở độ tuổi từ 25-30, trong khi khả năng cao nhất xảy ra ở độ tuổi 60 trở lên

#### Bill amount


```python
# Lấy ra tên các cột từ cột BILL_AMT1 tới default)
money_lst = [data.columns[i] 
             for i in np.arange(
                 data.columns.get_loc('BILL_AMT1'), 
                 data.columns.get_loc('default'
                 ))
             ]
print(money_lst)
print(len(money_lst))
```

    ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    12
    


```python
def kde_plot(columns, title):
  # Set the color palette for the curves
  colors = ['blue', 'green', 'red', 'purple', 'orange', 'gray']

  # Create a figure and axes
  fig, ax = plt.subplots(figsize = (12,6))

  # Plot the KDE curves
  for i, column in enumerate(columns):
      sns.kdeplot(data[column], color=colors[i], ax=ax, label=column, bw_method = 0.6)

  # Set the legend
  ax.legend()

  # Set the title and labels
  ax.set_title(title)
  ax.set_xlabel('Values')
  ax.set_ylabel('Density')

  # Show the plot
  plt.show()
```


```python
kde_plot(['BILL_AMT1','BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'], 'KDE Curves for Bill Amounts')
```


    
[png](/portfolio/assets/images/payment_prediction_files/payment_prediction_80_0.png)
    


Các biến BILL_AMTi (với i=[1,6]) có đồ thị lệch phảỉ.

#### Previous payment


```python
kde_plot(['PAY_1','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'], 'KDE Curves for previous payment status')
```


    
[png](/portfolio/assets/images/payment_prediction_files/payment_prediction_83_0.png)
    


Từ đồ thị biểu diễn phân phối của các biến PAY_i (với i= [1,6]), đồ thị phân phối lệch sang bên phải và phần lớn các quan sát có giá trị bằng 0.

#### Amount given credit ( LIMIT_BAL)


```python
# vẽ biểu đồ thể hiện phân phối theo độ tuổi
plt.figure(figsize = (10,4))
sns.histplot(data=data, x = 'LIMIT_BAL', kde = True)
plt.title("Customers' limit balance distribution ")
plt.xlabel("Balance", fontsize= 12)
plt.ylabel("Count of Clients", fontsize= 12)
plt.show()
```


    
[png](/portfolio/assets/images/payment_prediction_files/payment_prediction_86_0.png)
    


Đồ thị biểu diễn LIMIT_BAL lệch phải, các giá trị tập trung chủ yếu trong khoảng từ 0 tới < 10000

#### Target column


```python
target_count = data['default'].value_counts().reset_index().rename(columns={'index':'index','default':'count'})
fig = go.Figure(go.Bar(
    x = target_count['index'],y = target_count['count'],text=target_count['count'],marker={'color': target_count['count']}
    ,textposition = "outside"))
fig.update_layout(title_text='Biểu đồ phân phối giá trị default',xaxis_title="Default",yaxis_title="Số lượng", height = 600, width = 500)
fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-2.18.2.min.js"></script>                <div id="c9a9788c-ee43-4277-9837-4114a0be24f2" class="plotly-graph-div" style="height:600px; width:500px;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("c9a9788c-ee43-4277-9837-4114a0be24f2")) {                    Plotly.newPlot(                        "c9a9788c-ee43-4277-9837-4114a0be24f2",                        [{"marker":{"color":[23364,6636]},"text":[23364.0,6636.0],"textposition":"outside","x":[0,1],"y":[23364,6636],"type":"bar"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"title":{"text":"Bi\u1ec3u \u0111\u1ed3 ph\u00e2n ph\u1ed1i gi\u00e1 tr\u1ecb default"},"xaxis":{"title":{"text":"Default"}},"yaxis":{"title":{"text":"S\u1ed1 l\u01b0\u1ee3ng"}},"height":600,"width":500},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('c9a9788c-ee43-4277-9837-4114a0be24f2');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

                        })                };                            </script>        </div>
</body>
</html>


Từ bộ dữ liệu có 30.000 credit cards của người dùng,có thể thấy nhãn 0 chiếm đa số, trong khi đó chỉ có 6.636 nhãn 1 cho thấy khả năng default chiếm tỷ lệ khoảng 22,1%. Phân phối 2 giá trị 0 và 1 của biến default có sự chênh lệch tương đối lớn.

# **5. Building model**

## **5.1 Data Preparation**

### Drop những cột không cần thiết


```python
# Những cột cần drop
columns_to_drop = ['AgeBin']
```


```python
data.drop(columns = columns_to_drop, axis = 1, inplace = True)
```


```python
# Chia dữ liệu thành tập biến và tập nhãn
X = data.drop(['default'], axis=1)
y = data['default']
```


```python
# Train Test split:
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4, random_state = 42)
```

### Scale dữ liệu

Ta sẽ tiến hành Robust Scale để giảm thiểu ảnh hưởng của outlier 


```python
#Các biến cần scale
columns_to_scale = ['LIMIT_BAL', 'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
                    'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']

categorical_columns = ['SEX','EDUCATION','MARRIAGE','AGE'] + [f'PAY_{i}' for i in np.arange(1,7)]

#Tạo các dataframe với các cột là categorical columns
X_train_cat = X_train[categorical_columns]
X_test_cat = X_test[categorical_columns]
```


```python
#Khởi tạo Scaler
scaler = RobustScaler()
#Áp dụng Robust Scale
X_train_scaled = scaler.fit_transform(X_train[columns_to_scale])
X_test_scaled = scaler.transform(X_test[columns_to_scale])
```


```python
#Chuyển X_train và X_test về định dạng DataFrame sau khi scale:
X_train_scaled = pd.DataFrame(X_train_scaled, columns=columns_to_scale)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=columns_to_scale)

#Kiểm tra 
X_test_scaled
```





  <div id="df-231011b9-30ff-4ed3-a7b5-ce51338b6cb0">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LIMIT_BAL</th>
      <th>BILL_AMT1</th>
      <th>BILL_AMT2</th>
      <th>BILL_AMT3</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.578947</td>
      <td>-0.212975</td>
      <td>-0.183055</td>
      <td>-0.149532</td>
      <td>-0.122138</td>
      <td>-0.091725</td>
      <td>-0.048447</td>
      <td>-0.148680</td>
      <td>-0.005732</td>
      <td>-0.071284</td>
      <td>0.000000</td>
      <td>-0.003157</td>
      <td>0.129870</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.052632</td>
      <td>1.787799</td>
      <td>1.713946</td>
      <td>1.682624</td>
      <td>1.561367</td>
      <td>1.217539</td>
      <td>1.246435</td>
      <td>0.590018</td>
      <td>0.539390</td>
      <td>0.323393</td>
      <td>0.305724</td>
      <td>0.304434</td>
      <td>0.303636</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.368421</td>
      <td>0.745510</td>
      <td>0.785525</td>
      <td>0.843203</td>
      <td>0.959322</td>
      <td>1.061511</td>
      <td>1.100181</td>
      <td>0.081638</td>
      <td>0.265488</td>
      <td>0.285137</td>
      <td>0.250017</td>
      <td>0.259966</td>
      <td>0.273766</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.052632</td>
      <td>-0.028125</td>
      <td>-0.037090</td>
      <td>-0.069501</td>
      <td>-0.040460</td>
      <td>-0.142439</td>
      <td>-0.211956</td>
      <td>-0.121467</td>
      <td>-0.052561</td>
      <td>1.238921</td>
      <td>-0.392617</td>
      <td>1.446915</td>
      <td>0.755325</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.473684</td>
      <td>1.122690</td>
      <td>0.433578</td>
      <td>0.387025</td>
      <td>0.010158</td>
      <td>0.042811</td>
      <td>0.047182</td>
      <td>-0.024986</td>
      <td>-0.127683</td>
      <td>-0.190091</td>
      <td>0.079963</td>
      <td>-0.397842</td>
      <td>-0.129870</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11995</th>
      <td>-0.473684</td>
      <td>0.388468</td>
      <td>0.446576</td>
      <td>0.486527</td>
      <td>0.568299</td>
      <td>0.619275</td>
      <td>0.670296</td>
      <td>0.174655</td>
      <td>0.240122</td>
      <td>0.240466</td>
      <td>0.133271</td>
      <td>0.128404</td>
      <td>0.129870</td>
    </tr>
    <tr>
      <th>11996</th>
      <td>1.894737</td>
      <td>-0.341028</td>
      <td>-0.330234</td>
      <td>-0.325124</td>
      <td>-0.324875</td>
      <td>-0.297036</td>
      <td>-0.253082</td>
      <td>-0.247387</td>
      <td>-0.225244</td>
      <td>0.071997</td>
      <td>0.266542</td>
      <td>0.128404</td>
      <td>0.129870</td>
    </tr>
    <tr>
      <th>11997</th>
      <td>-0.210526</td>
      <td>1.058538</td>
      <td>1.154162</td>
      <td>1.264291</td>
      <td>1.420276</td>
      <td>0.726327</td>
      <td>0.781915</td>
      <td>0.519265</td>
      <td>0.457683</td>
      <td>0.488060</td>
      <td>0.133271</td>
      <td>0.128404</td>
      <td>0.129870</td>
    </tr>
    <tr>
      <th>11998</th>
      <td>0.894737</td>
      <td>-0.274467</td>
      <td>-0.327264</td>
      <td>-0.324009</td>
      <td>-0.355556</td>
      <td>-0.397278</td>
      <td>-0.398984</td>
      <td>-0.204342</td>
      <td>-0.110122</td>
      <td>-0.370678</td>
      <td>-0.141267</td>
      <td>-0.397842</td>
      <td>1.320779</td>
    </tr>
    <tr>
      <th>11999</th>
      <td>-0.473684</td>
      <td>0.358755</td>
      <td>0.281738</td>
      <td>0.343271</td>
      <td>0.155505</td>
      <td>0.193808</td>
      <td>0.237569</td>
      <td>-0.099202</td>
      <td>0.032317</td>
      <td>-0.142569</td>
      <td>-0.130606</td>
      <td>-0.029470</td>
      <td>-0.137662</td>
    </tr>
  </tbody>
</table>
<p>12000 rows × 13 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-231011b9-30ff-4ed3-a7b5-ce51338b6cb0')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-231011b9-30ff-4ed3-a7b5-ce51338b6cb0 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-231011b9-30ff-4ed3-a7b5-ce51338b6cb0');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




Sau khi đã scale xong, ta sẽ tiến hành gộp lại để tập Train và tập Test đầy đủ các cột như ban đầu


```python
X_train = pd.concat([X_train_scaled,X_train_cat.reset_index(drop=True)], axis = 1)
X_test = pd.concat([X_test_scaled,X_test_cat.reset_index(drop=True)], axis = 1)
```

Ta cũng cần gộp lại thành 1 `data_train` gồm biến và nhãn để drop outliers


```python
# Gộp lại thành tập dữ liệu train để xử lí outlier
data_train = pd.concat([X_train, y_train.to_frame().reset_index(drop=True)], axis = 1)
```


```python
data_train
```





  <div id="df-09a49b0a-2d9e-40b7-8c02-627ccea0adac">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LIMIT_BAL</th>
      <th>BILL_AMT1</th>
      <th>BILL_AMT2</th>
      <th>BILL_AMT3</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_1</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>PAY_5</th>
      <th>PAY_6</th>
      <th>default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.315789</td>
      <td>-0.344032</td>
      <td>-0.333779</td>
      <td>-0.342905</td>
      <td>-0.350865</td>
      <td>-0.372207</td>
      <td>-0.316254</td>
      <td>-0.302554</td>
      <td>-0.374512</td>
      <td>-0.311750</td>
      <td>-0.399813</td>
      <td>0.106039</td>
      <td>-0.363636</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>23</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.105263</td>
      <td>0.465622</td>
      <td>0.302433</td>
      <td>-0.004166</td>
      <td>-0.186924</td>
      <td>-0.082748</td>
      <td>-0.275771</td>
      <td>-0.130126</td>
      <td>-0.054512</td>
      <td>-0.071284</td>
      <td>3.331779</td>
      <td>0.654651</td>
      <td>0.350649</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>38</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.736842</td>
      <td>-0.322314</td>
      <td>-0.303615</td>
      <td>-0.318709</td>
      <td>-0.324610</td>
      <td>-0.333845</td>
      <td>-0.317188</td>
      <td>0.152143</td>
      <td>-0.035976</td>
      <td>0.018059</td>
      <td>0.100220</td>
      <td>0.094198</td>
      <td>2.277143</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.578947</td>
      <td>-0.134335</td>
      <td>-0.095877</td>
      <td>-0.070042</td>
      <td>-0.065145</td>
      <td>-0.046900</td>
      <td>-0.008441</td>
      <td>-0.024986</td>
      <td>-0.176463</td>
      <td>-0.427706</td>
      <td>-0.249484</td>
      <td>-0.108407</td>
      <td>-0.129870</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>26</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.631579</td>
      <td>0.371899</td>
      <td>0.183334</td>
      <td>0.188562</td>
      <td>0.197062</td>
      <td>0.241516</td>
      <td>0.061077</td>
      <td>-0.124188</td>
      <td>-0.173780</td>
      <td>0.047523</td>
      <td>-0.133271</td>
      <td>0.128404</td>
      <td>-0.221558</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>17995</th>
      <td>-0.473684</td>
      <td>0.469393</td>
      <td>0.531473</td>
      <td>0.620231</td>
      <td>0.698363</td>
      <td>0.225688</td>
      <td>-0.229066</td>
      <td>-0.024986</td>
      <td>0.238171</td>
      <td>-0.049661</td>
      <td>-0.380622</td>
      <td>-0.099461</td>
      <td>18.680779</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>32</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17996</th>
      <td>0.315789</td>
      <td>2.106914</td>
      <td>2.385820</td>
      <td>2.583894</td>
      <td>2.745500</td>
      <td>3.084312</td>
      <td>3.212941</td>
      <td>2.819964</td>
      <td>0.969878</td>
      <td>-0.427706</td>
      <td>1.599254</td>
      <td>1.180897</td>
      <td>0.649351</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>37</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17997</th>
      <td>-0.473684</td>
      <td>-0.351667</td>
      <td>-0.348189</td>
      <td>-0.351412</td>
      <td>-0.360096</td>
      <td>-0.372207</td>
      <td>-0.355970</td>
      <td>-0.519760</td>
      <td>-0.493537</td>
      <td>-0.427706</td>
      <td>-0.399813</td>
      <td>-0.397842</td>
      <td>-0.389610</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17998</th>
      <td>-0.368421</td>
      <td>0.805234</td>
      <td>0.808633</td>
      <td>0.540270</td>
      <td>0.304257</td>
      <td>0.213317</td>
      <td>0.225541</td>
      <td>0.222401</td>
      <td>-0.005732</td>
      <td>0.641559</td>
      <td>-0.079963</td>
      <td>-0.397842</td>
      <td>-0.077922</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17999</th>
      <td>0.105263</td>
      <td>-0.351980</td>
      <td>-0.348517</td>
      <td>-0.287959</td>
      <td>-0.304579</td>
      <td>-0.339427</td>
      <td>-0.062944</td>
      <td>-0.519760</td>
      <td>0.399146</td>
      <td>0.317215</td>
      <td>0.039981</td>
      <td>3.338508</td>
      <td>0.000000</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>18000 rows × 24 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-09a49b0a-2d9e-40b7-8c02-627ccea0adac')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-09a49b0a-2d9e-40b7-8c02-627ccea0adac button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-09a49b0a-2d9e-40b7-8c02-627ccea0adac');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




### Xử lí outlier

Trước tiên, hãy xem xét tình trạng của các biến continuous:

Với biến `LIMIT_BAL`


```python
data[["LIMIT_BAL"]].boxplot(figsize=(3,4), fontsize=12)
plt.show()
```


    
[png](/portfolio/assets/images/payment_prediction_files/payment_prediction_110_0.png)
    


Với các biến `BILL_AMT` từ 1 đến 6: 


```python
columns_bill = [f'BILL_AMT{i}' for i in np.arange(1, 7)]

def box_plot_columns(columns, data):
  # Create a figure and axes for the box plots
  fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

  # Iterate over the columns and plot the box plots
  for i, column in enumerate(columns):
      ax = axes[i // 3, i % 3]  # Select the current axis
      ax.boxplot(data[column])  # Plot the box plot
      ax.set_title(column)  # Set the title for the subplot

  # Adjust spacing between subplots
  plt.tight_layout()

  # Show the plot
  plt.show()

box_plot_columns(columns_bill,data)
```


    
[png](/portfolio/assets/images/payment_prediction_files/payment_prediction_112_0.png)
    


Với các biến `PAY_AMT` từ 1 đến 6: 


```python
columns_pay = [f'PAY_AMT{i}' for i in np.arange(1, 7)]
box_plot_columns(columns_pay,data)
```


    
[png](/portfolio/assets/images/payment_prediction_files/payment_prediction_114_0.png)
    


* Nhìn qua các đồ thị, ta thấy được hầu như ở tất cả các biến giá trị đều tồn tại outlier, và ta cần phải xử lí chúng để mô hình dự đoán không bị ảnh hưởng. Ta sẽ tiến hành kiểm tra số lượng outlier và drop hết những giá trị này

* Có một điều nữa ta cần phải lưu ý, đó là ta chỉ drop outlier trên bộ dữ liệu của tập TRAIN


```python
def remove_outliers(data: pd.DataFrame, feat: str):
    feat_fraud = data[feat]
    q25, q75 = np.percentile(feat_fraud, 25), np.percentile(feat_fraud, 75)
    iqr = q75 - q25
    
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off

    outliers = [x for x in feat_fraud if x < lower or x > upper]
    print(f'Số điểm bất thường cho thuộc tính {feat} trong tập data_train: {len(outliers)}')

    return data.drop(data[(data[feat] > upper) | (data[feat] < lower)].index)
```


```python
columns_to_drop_outliers = ['LIMIT_BAL'] + [f'BILL_AMT{i}' for i in np.arange(1,7)] + [f'PAY_AMT{i}' for i in np.arange(1, 7)]

for col in columns_to_drop_outliers:
  data_train = remove_outliers(data_train, col)
```

    Số điểm bất thường cho thuộc tính LIMIT_BAL trong tập data_train: 98
    Số điểm bất thường cho thuộc tính BILL_AMT1 trong tập data_train: 1402
    Số điểm bất thường cho thuộc tính BILL_AMT2 trong tập data_train: 832
    Số điểm bất thường cho thuộc tính BILL_AMT3 trong tập data_train: 436
    Số điểm bất thường cho thuộc tính BILL_AMT4 trong tập data_train: 643
    Số điểm bất thường cho thuộc tính BILL_AMT5 trong tập data_train: 540
    Số điểm bất thường cho thuộc tính BILL_AMT6 trong tập data_train: 285
    Số điểm bất thường cho thuộc tính PAY_AMT1 trong tập data_train: 1403
    Số điểm bất thường cho thuộc tính PAY_AMT2 trong tập data_train: 951
    Số điểm bất thường cho thuộc tính PAY_AMT3 trong tập data_train: 852
    Số điểm bất thường cho thuộc tính PAY_AMT4 trong tập data_train: 582
    Số điểm bất thường cho thuộc tính PAY_AMT5 trong tập data_train: 657
    Số điểm bất thường cho thuộc tính PAY_AMT6 trong tập data_train: 571
    

Kiểm tra lại về outliers


```python
columns_pay = [f'PAY_AMT{i}' for i in np.arange(1, 7)]
box_plot_columns(columns_pay,data_train)
```


    
[png](/portfolio/assets/images/payment_prediction_files/payment_prediction_119_0.png)
    



```python
columns_pay = [f'BILL_AMT{i}' for i in np.arange(1, 7)]
box_plot_columns(columns_pay,data_train)
```


    
[png](/portfolio/assets/images/payment_prediction_files/payment_prediction_120_0.png)
    


Nhìn chung, hầu hết outlier đã được xóa bỏ khỏi `data_train`, còn lại một số ít không đáng kể sẽ là những điểm quan trọng cho quá trình dự đoán

## 5.2 Modeling without imbalance resolve


```python
# Tiến hành chia lại tập biến và tập nhãn:
X_train = data_train.drop('default', axis = 1)
y_train = data_train['default']
```

Ta sẽ train một số model với tham số mặc định


```python
def predictProcess(model_type, X_train, X_test, y_train, y_test, **kwargs):
    # Check if model_type is valid
    if model_type not in ('KNN', 'SVC','RandomForest','XGB','Logistic'):
        raise ValueError('Choose one of these models: KNN, SVC, RandomForest, XGB, Logistic')
    
    # Getting the model
    if model_type == 'KNN':
        model = KNeighborsClassifier()
    elif model_type == 'RandomForest':
        model = RandomForestClassifier()
    elif model_type == 'SVC':
        model = SVC()
    elif model_type == 'XGB':
        model = XGBClassifier()
    elif model_type == 'Logistic':
        model = LogisticRegression()

    # Fit the model
    model.fit(X_train, y_train)

    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    return y_pred_train, y_pred_test
  
def confusion_heat(y_true, y_pred, ax, title):
    cm = confusion_matrix(y_true,y_pred)
    ax.set_title(title)
    plt.figure(figsize = (4,4))
    sns.heatmap(cm, cmap = 'Greens', linecolor = 'black', linewidth = 1, 
                annot = True, fmt = '', ax = ax)
```


```python
classifiers = ['KNN','SVC','RandomForest','XGB','Logistic']
metrics_result = {}

for classifier in classifiers: 
  time_start = time.time()
  print(f'Prediting by classifier: {classifier} ....')
  y_pred_train, y_pred_test = predictProcess(classifier, X_train, X_test, y_train, y_test)
  print(f'Time consumed: {time.time() - time_start}')

  # Metrics
  acc_train = accuracy_score(y_train, y_pred_train) 
  acc_test = accuracy_score(y_test, y_pred_test)  
  pre_train = precision_score(y_train, y_pred_train)
  pre_test = precision_score(y_test, y_pred_test)
  re_train = recall_score(y_train, y_pred_train)
  re_test = recall_score(y_test, y_pred_test)
  f1_train = f1_score(y_train, y_pred_train)
  f1_test = f1_score(y_test, y_pred_test)

  # Confusion matrix
  fig, axs = plt.subplots(1,2, figsize = (10,4))
  confusion_heat(y_train, y_pred_train, axs[0], f'{classifier}: Train_data')
  confusion_heat(y_test, y_pred_test, axs[1], f'{classifier}: Test_data')
  plt.show()

  # Store metrics
  if classifier not in metrics_result:
    metrics_result[classifier] = [acc_train, acc_test, pre_train, pre_test, re_train, re_test, f1_train, f1_test]
  
  print('-' * 20)
```

    Prediting by classifier: KNN ....
    Time consumed: 2.427715301513672
    


    
[png](/portfolio/assets/images/payment_prediction_files/payment_prediction_126_1.png)
    



    <Figure size 400x400 with 0 Axes>



    <Figure size 400x400 with 0 Axes>


    --------------------
    Prediting by classifier: SVC ....
    Time consumed: 14.027998924255371
    


    
[png](/portfolio/assets/images/payment_prediction_files/payment_prediction_126_5.png)
    



    <Figure size 400x400 with 0 Axes>



    <Figure size 400x400 with 0 Axes>


    --------------------
    Prediting by classifier: RandomForest ....
    Time consumed: 2.1818292140960693
    


    
[png](/portfolio/assets/images/payment_prediction_files/payment_prediction_126_9.png)
    



    <Figure size 400x400 with 0 Axes>



    <Figure size 400x400 with 0 Axes>


    --------------------
    Prediting by classifier: XGB ....
    Time consumed: 1.3470745086669922
    


    
[png](/portfolio/assets/images/payment_prediction_files/payment_prediction_126_13.png)
    



    <Figure size 400x400 with 0 Axes>



    <Figure size 400x400 with 0 Axes>


    --------------------
    Prediting by classifier: Logistic ....
    Time consumed: 0.09206700325012207
    

    /usr/local/lib/python3.10/dist-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    


    
[png](/portfolio/assets/images/payment_prediction_files/payment_prediction_126_18.png)
    



    <Figure size 400x400 with 0 Axes>



    <Figure size 400x400 with 0 Axes>


    --------------------
    

Tổng hợp kết quả chạy được ta thu được bảng sau


```python
metric_df = pd.DataFrame(metrics_result,  index = ['acc_train', 'acc_test', 'precision_train', 'precision_test', 'recall_train', 'recall_test', 'f1_train', 'f1_test'])
metric_df
```





  <div id="df-1c4b7abe-6c88-4d05-89fe-45b41bf80b57">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>KNN</th>
      <th>SVC</th>
      <th>RandomForest</th>
      <th>XGB</th>
      <th>Logistic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>acc_train</th>
      <td>0.810242</td>
      <td>0.763146</td>
      <td>0.999086</td>
      <td>0.920668</td>
      <td>0.774005</td>
    </tr>
    <tr>
      <th>acc_test</th>
      <td>0.786750</td>
      <td>0.806417</td>
      <td>0.794083</td>
      <td>0.798833</td>
      <td>0.807750</td>
    </tr>
    <tr>
      <th>precision_train</th>
      <td>0.723885</td>
      <td>0.633124</td>
      <td>0.998776</td>
      <td>0.966578</td>
      <td>0.690096</td>
    </tr>
    <tr>
      <th>precision_test</th>
      <td>0.525362</td>
      <td>0.610650</td>
      <td>0.553284</td>
      <td>0.565842</td>
      <td>0.665354</td>
    </tr>
    <tr>
      <th>recall_train</th>
      <td>0.522625</td>
      <td>0.369344</td>
      <td>0.997962</td>
      <td>0.742764</td>
      <td>0.352222</td>
    </tr>
    <tr>
      <th>recall_test</th>
      <td>0.329171</td>
      <td>0.334090</td>
      <td>0.337874</td>
      <td>0.372304</td>
      <td>0.255770</td>
    </tr>
    <tr>
      <th>f1_train</th>
      <td>0.607008</td>
      <td>0.466529</td>
      <td>0.998369</td>
      <td>0.840018</td>
      <td>0.466397</td>
    </tr>
    <tr>
      <th>f1_test</th>
      <td>0.404745</td>
      <td>0.431890</td>
      <td>0.419544</td>
      <td>0.449110</td>
      <td>0.369500</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1c4b7abe-6c88-4d05-89fe-45b41bf80b57')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-1c4b7abe-6c88-4d05-89fe-45b41bf80b57 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1c4b7abe-6c88-4d05-89fe-45b41bf80b57');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




Có thể thấy rằng, với một số mô hình vừa thử nghiệm trên tập data bị imbalance, các mô hình chưa hoạt động chưa ổn định


## 5.3 Modeling with resampled dataset

---



- Hàm dưới đây sẽ tiến hành chia tập train thành các folds, sau đó tiến hành upsampling cho tập train và đánh giá metric `precision` cho mỗi iteration

- Trong mỗi iteration, ta cũng tiến hành luôn đồng thời robust scale rồi min max scale (`fit_transform` trên tập train, `transform` trên tập evaluate)

- Kết quả trả về là kết quả: accuracy, precision, recall và f1_score tương ứng với từng fold


```python
def score_model(model, X_train, y_train, cv=None):
    """
    Creates folds manually, and upsamples within each fold.
    Returns an array of validation (precision) scores
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    smoter = SMOTE(random_state=42)
    
    scores = []

    X_train.reset_index(drop=True, inplace=True)  # Reset indices of X_train
    y_train.reset_index(drop=True, inplace=True)  # Reset indices of y_train

    num_fold = 0

    for train_fold_index, val_fold_index in cv.split(X_train, y_train):
        num_fold += 1
        # Get the training data
        X_train_fold, y_train_fold = X_train.iloc[train_fold_index], y_train[train_fold_index]

        # Get the validation data
        X_val_fold, y_val_fold = X_train.iloc[val_fold_index], y_train[val_fold_index]

        # Robust scale
        X_train_fold_scaled = scaler.fit_transform(X_train_fold[columns_to_scale])
        X_val_fold_scaled = scaler.transform(X_val_fold[columns_to_scale])

        # Upsample only the data in the training section
        X_train_fold_scaled_upsample, y_train_fold_upsample = smoter.fit_resample(X_train_fold_scaled,
                                                                                   y_train_fold)
        # Fit the model on the upsampled training data
        model_obj = model().fit(X_train_fold_scaled_upsample, y_train_fold_upsample)
        
        # Prediction
        y_pred_fold = model_obj.predict(X_val_fold_scaled)

        # Score the mean of metrics on the (non-upsampled) validation data
        acc = accuracy_score(y_val_fold, y_pred_fold)
        pre = precision_score(y_val_fold, y_pred_fold)
        recall = recall_score(y_val_fold, y_pred_fold)
        f1 = f1_score(y_val_fold, y_pred_fold)
        scores.append({f'fold_{num_fold}':[acc,pre,recall,f1]})

    return scores

```


```python
# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```


```python
score_model(LogisticRegression, X_train, y_train, cv=None)
```




    [{'fold_1': [0.5531111111111111,
       0.29653505237711525,
       0.7352647352647352,
       0.422624174562159]},
     {'fold_2': [0.5297777777777778,
       0.2754032258064516,
       0.6816367265469062,
       0.392303273980471]},
     {'fold_3': [0.5468888888888889,
       0.2879345603271984,
       0.7025948103792415,
       0.4084711343196983]},
     {'fold_4': [0.5508888888888889,
       0.2929703372612759,
       0.719560878243513,
       0.4164019636153624]},
     {'fold_5': [0.5402222222222223,
       0.28736548425667596,
       0.719560878243513,
       0.41070919965821706]}]




```python
score_model(RandomForestClassifier, X_train, y_train, cv=None)
```




    [{'fold_1': [0.7555555555555555,
       0.44761904761904764,
       0.4225774225774226,
       0.4347379239465571]},
     {'fold_2': [0.7506666666666667,
       0.43788819875776397,
       0.4221556886227545,
       0.4298780487804878]},
     {'fold_3': [0.7528888888888889,
       0.43861607142857145,
       0.39221556886227543,
       0.41412012644889357]},
     {'fold_4': [0.7555555555555555,
       0.44638949671772427,
       0.40718562874251496,
       0.42588726513569936]},
     {'fold_5': [0.7504444444444445,
       0.4379487179487179,
       0.42614770459081835,
       0.4319676277187658]}]




```python
score_model(XGBClassifier, X_train, y_train, cv=None)
```




    [{'fold_1': [0.7511111111111111,
       0.43539630836047777,
       0.4005994005994006,
       0.4172736732570239]},
     {'fold_2': [0.7446666666666667,
       0.42729970326409494,
       0.4311377245508982,
       0.42921013412816694]},
     {'fold_3': [0.7431111111111111,
       0.41630434782608694,
       0.38223552894211577,
       0.39854318418314255]},
     {'fold_4': [0.7466666666666667,
       0.4228187919463087,
       0.3772455089820359,
       0.39873417721518994]},
     {'fold_5': [0.7464444444444445,
       0.4287179487179487,
       0.4171656686626746,
       0.42286292362164896]}]




```python
score_model(SVC, X_train, y_train, cv=None)
```




    [{'fold_1': [0.558,
       0.30612244897959184,
       0.7792207792207793,
       0.4395604395604396]},
     {'fold_2': [0.5488888888888889,
       0.2968379446640316,
       0.7495009980039921,
       0.4252548131370329]},
     {'fold_3': [0.5706666666666667,
       0.3072139303482587,
       0.7395209580838323,
       0.4340949033391916]},
     {'fold_4': [0.5526666666666666,
       0.30043426766679826,
       0.7594810379241517,
       0.4305516265912305]},
     {'fold_5': [0.564, 0.309674861221253, 0.779441117764471, 0.4432463110102156]}]



Với các mô đồ đã thử nghiệm sau khi đã oversampling bằng `SMOTE`, ta thấy được Random Forest Classifier hoạt động ổn nhất trong các classifer được thử với tham số mặc định

## 5.4 Hyperparameters tunning

Vì giới hạn về điều kiện nên ta sẽ dùng random search để tìm kiếm siêu tham số tốt nhất của Random Forest


```python
params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt'],
    'criterion': ['gini', 'entropy'],
    'class_weight': ['balanced'],
    'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
    'max_leaf_nodes': [None, 5, 10],
    'min_impurity_decrease': [0.0, 0.1, 0.2],
    'oob_score': [False, True]
}
```


```python
grid_no_up = RandomizedSearchCV(RandomForestClassifier(), param_distributions=params, cv=5, 
                          scoring='precision').fit(X_train, y_train)

grid_no_up.best_score_
```

    /usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    




    0.5081221929936274




```python
grid_no_up.best_params_
```




    {'oob_score': False,
     'n_estimators': 200,
     'min_weight_fraction_leaf': 0.0,
     'min_samples_split': 10,
     'min_samples_leaf': 2,
     'min_impurity_decrease': 0.1,
     'max_leaf_nodes': 5,
     'max_features': 'sqrt',
     'max_depth': 5,
     'criterion': 'entropy',
     'class_weight': 'balanced'}



## 5.5 Retraining and Predicting


```python
final_model = RandomForestClassifier(**grid_no_up.best_params_)

final_model.fit(X_train, y_train)

y_train_pred = final_model.predict(X_train)

y_test_pred = final_model.predict(X_test)

acc_train = accuracy_score(y_train, y_train_pred)
pre_train = precision_score(y_train, y_train_pred)
recall_train = recall_score(y_train, y_train_pred)
f1_train = f1_score(y_train, y_train_pred)

acc_test = accuracy_score(y_test, y_test_pred)
pre_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)
```


```python
print(acc_train)
print(acc_test)
```

    0.7808444444444445
    0.778
    


```python
print(pre_train)
print(pre_test)
```

    0.5076500588466065
    0.488953488372093
    


```python
print(recall_train)
print(recall_test)
```

    0.5166699940107806
    0.5169022741241549
    


```python
print(f1_train)
print(f1_test)
```

    0.5121203126545958
    0.5025395876904691
    

Hãy xem xét phân phối của các giá trị được dự đoán trên dữ liệu huấn luyện/kiểm tra.


```python
def distribution_plot(ax, x_value, y_value, x_label, y_label, title):
    """
    Params:
     :x_value(array): 
     :y_value(array):
     :x_label(str): 
     :y_label(str):
     :title(str):
    
    Returns
    """
    sns.kdeplot(x_value, color="r", label=x_label, ax=ax)
    sns.kdeplot(y_value, color="b", label=y_label, ax=ax)

    ax.set_title(title)
    ax.legend()
```


```python
fig, axs = plt.subplots(1, 2, figsize=(20, 6))
title1 = 'Distribution Plot of Predicted Value Using Training Data vs Training Data'
distribution_plot(axs[0], y_train, y_train_pred, "Actual Values (Train)", "Predicted Values (Train)", title1)

title2 = 'Distribution Plot of Predicted Value Using Test Data vs Data Distribution of Test Data'
distribution_plot(axs[1], y_test, y_test_pred, "Actual Values (Test)","Predicted Values (Test)", title2)
plt.show()
```


    
[png](/portfolio/assets/images//assets/payment_prediction_files/payment_prediction_151_0.png)
    

