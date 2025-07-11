import streamlit as st
import pandas as pd
import numpy as np

@st.cache_data
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file, encoding='cp949')
    data['일자'] = pd.to_datetime(data['일자'])
    return data

st.title('외국인 야간선물 데이터 분석')

uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type="csv")

if uploaded_file:
    data = load_data(uploaded_file)

    # 콘텐츠 1
    st.header("1. 외국인 야간선물 vs 다음날 정규장 외국인 선물")
    st.dataframe(data[['일자', '야간_외국인_선물', '다음날_정규장_외국인_선물']].set_index('일자'))

    corr_fut = np.corrcoef(data['야간_외국인_선물'], data['다음날_정규장_외국인_선물'])[0, 1]
    corr_type_fut = "양의" if corr_fut > 0 else "음의"
    st.write(f"두 데이터 간 관계는 {corr_type_fut} 상관관계이고, 확률은 {abs(corr_fut)*100:.2f}%이다.")

    # 콘텐츠 2
    st.header("2. 외국인 야간선물 vs 다음날 정규장 외국인 현물")
    st.dataframe(data[['일자', '야간_외국인_선물', '다음날_정규장_외국인_현물']].set_index('일자'))

    corr_cash = np.corrcoef(data['야간_외국인_선물'], data['다음날_정규장_외국인_현물'])[0, 1]
    corr_type_cash = "양의" if corr_cash > 0 else "음의"
    st.write(f"두 데이터 간 관계는 {corr_type_cash} 상관관계이고, 확률은 {abs(corr_cash)*100:.2f}%이다.")

    # 콘텐츠 3
    st.header("3. 외국인 야간선물 vs 다음날 K200지수 상승률")
    data['K200지수_상승률'] = data['K200지수_상승률'].round(2)
    st.dataframe(data[['일자', '야간_외국인_선물', '다음날_K200지수', 'K200지수_상승률']].set_index('일자'))

    corr_k200 = np.corrcoef(data['야간_외국인_선물'], data['K200지수_상승률'])[0, 1]
    corr_type_k200 = "양의" if corr_k200 > 0 else "음의"
    st.write(f"두 데이터 간 관계는 {corr_type_k200} 상관관계이고, 확률은 {abs(corr_k200)*100:.2f}%이다.")
