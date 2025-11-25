import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier


# -----------------------------
# 페이지 기본 설정
# -----------------------------
st.set_page_config(page_title="Hybrid Loan Default Model", layout="wide")

st.title("대출 부도 예측 Hybrid 모델 (Logistic + XGBoost)")
st.write("CSV 파일을 업로드하고, Target 변수를 선택한 후 모델을 학습합니다.")


# -----------------------------
# 1. 파일 업로드
# -----------------------------
uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("파일 업로드 완료!")
    st.write("데이터 미리보기:")
    st.dataframe(df.head())

    # -----------------------------
    # 2. Target 변수 선택
    # -----------------------------
    st.subheader("Target 변수 선택")

    default_target = "target" if "target" in df.columns else None
    target_col = st.selectbox(
        "Target 변수를 선택하세요",
        options=df.columns,
        index=list(df.columns).index(default_target) if default_target else 0
    )

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # -----------------------------
    # 3. 수치형 / 범주형 변수 구분
    # -----------------------------
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    st.write(f"수치형 변수 개수: {len(numeric_features)}")
    st.write(f"범주형 변수 개수: {len(categorical_features)}")

    # --------------------
