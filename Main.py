import streamlit as st
import pandas as pd
import numpy as np

# 파일 자동 로딩 (깃허브 또는 로컬)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/{YOUR_GITHUB_USERNAME}/{YOUR_REPO_NAME}/main/외국인%20야간선물%20전처리%20파일.csv"
    data = pd.read_csv(url, encoding='cp949')
    data['일자'] = pd.to_datetime(data['일자'])
    return data

data = load_data()

st.title('외국인 야간선물 데이터 분석')

# 콘텐츠 1: 야간선물 vs 다음날 정규장 외국인 선물
st.header("1. 외국인 야간선물 vs 다음날 정규장 외국인 선물")
st.dataframe(data[['일자', '야간_외국인_선물', '다음날_정규장_외국인_선물']].set_index('일자'))

corr_fut = np.corrcoef(data['야간_외국인_선물'], data['다음날_정규장_외국인_선물'])[0, 1]
corr_type_fut = "양의" if corr_fut > 0 else "음의"
st.write(f"두 데이터 간 관계는 {corr_type_fut} 상관관계이고, 확률은 {abs(corr_fut)*100:.2f}%이다.")

# 콘텐츠 2: 야간선물 vs 다음날 정규장 외국인 현물
st.header("2. 외국인 야간선물 vs 다음날 정규장 외국인 현물")
st.dataframe(data[['일자', '야간_외국인_선물', '다음날_정규장_외국인_현물']].set_index('일자'))

corr_cash = np.corrcoef(data['야간_외국인_선물'], data['다음날_정규장_외국인_현물'])[0, 1]
corr_type_cash = "양의" if corr_cash > 0 else "음의"
st.write(f"두 데이터 간 관계는 {corr_type_cash} 상관관계이고, 확률은 {abs(corr_cash)*100:.2f}%이다.")

# 콘텐츠 3: 야간선물 vs 다음날 K200 지수 상승률
st.header("3. 외국인 야간선물 vs 다음날 K200지수 상승률")
data['K200지수_상승률'] = data['K200지수_상승률'].round(2)
st.dataframe(data[['일자', '야간_외국인_선물', '다음날_K200지수', 'K200지수_상승률']].set_index('일자'))

corr_k200 = np.corrcoef(data['야간_외국인_선물'], data['K200지수_상승률'])[0, 1]
corr_type_k200 = "양의" if corr_k200 > 0 else "음의"
st.write(f"두 데이터 간 관계는 {corr_type_k200} 상관관계이고, 확률은 {abs(corr_k200)*100:.2f}%이다.")
