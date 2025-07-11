import streamlit as st
import pandas as pd
import numpy as np
import io

@st.cache_data
def load_data(file):
    encodings = ['utf-8', 'cp949', 'euc-kr']
    for enc in encodings:
        try:
            file.seek(0)  # 파일의 처음으로 돌아가 다시 읽기 시도
            data = pd.read_csv(file, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("지원하지 않는 인코딩입니다.")

    data.columns = ['일자', 'K200지수', '야간_외국인_선물', '정규장_외국인_현물', '정규장_외국인_선물']
    data = data.drop(index=0)
    data['일자'] = pd.to_datetime(data['일자'])
    for col in ['K200지수', '야간_외국인_선물', '정규장_외국인_현물', '정규장_외국인_선물']:
        data[col] = data[col].astype(str).str.replace(',', '').astype(float)
    data = data.sort_values(by='일자').reset_index(drop=True)
    data['다음날_정규장_외국인_현물'] = data['정규장_외국인_현물'].shift(-1)
    data['다음날_정규장_외국인_선물'] = data['정규장_외국인_선물'].shift(-1)
    data['다음날_K200지수'] = data['K200지수'].shift(-1)
    data['K200지수_상승률'] = (data['다음날_K200지수'] - data['K200지수']) / data['K200지수'] * 100
    return data.dropna()

st.title('외국인 야간선물과 다음날 시장의 상관관계 분석')

uploaded_file = st.file_uploader("CSV 파일 업로드", type="csv")

if uploaded_file is not None:
    try:
        data = load_data(uploaded_file)

        correlation_matrix = pd.DataFrame({
            '구분': ['상관계수'],
            '야간선물 vs 다음날 정규장 외국인 선물': [np.corrcoef(data['야간_외국인_선물'], data['다음날_정규장_외국인_선물'])[0,1]],
            '야간선물 vs 다음날 정규장 외국인 현물': [np.corrcoef(data['야간_외국인_선물'], data['다음날_정규장_외국인_현물'])[0,1]],
            '야간선물 vs 다음날 K200 지수 상승률': [np.corrcoef(data['야간_외국인_선물'], data['K200지수_상승률'])[0,1]]
        })

        st.subheader("분석 결과")
        st.table(correlation_matrix)

        st.subheader("데이터 미리보기")
        st.dataframe(data[['일자', '야간_외국인_선물', '다음날_정규장_외국인_현물', '다음날_정규장_외국인_선물', 'K200지수_상승률']])
        
    except Exception as e:
        st.error(f"에러 발생: {e}")
