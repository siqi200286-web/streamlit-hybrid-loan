
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
# 0. 페이지 기본 설정
# -----------------------------
st.set_page_config(page_title="Hybrid Loan Default Model", layout="wide")

st.title("대출 부도 예측 Hybrid 모델 (Logistic + XGBoost)")
st.write("CSV 파일을 업로드하고, Target 변수를 선택한 후 하이브리드 모델을 학습합니다.")


# -----------------------------
# 1. 파일 업로드
# -----------------------------
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file is not None:
    # 데이터 읽기
    df = pd.read_csv(uploaded_file)

    st.success("파일 업로드 완료")
    st.write("데이터 미리보기:")
    st.dataframe(df.head())

    # -----------------------------
    # 2. Target 변수 선택
    # -----------------------------
    st.subheader("Target 변수 선택")

    # 기본값: 'target' 이라는 컬럼이 있으면 자동 선택
    default_target_col = "target" if "target" in df.columns else None
    target_col = st.selectbox("Target 변수를 선택하세요", df.columns, index=(
        list(df.columns).index(default_target_col) if default_target_col in df.columns else 0
    ))

    # y, X 분리
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # -----------------------------
    # 3. 수치형 / 범주형 변수 구분
    # -----------------------------
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    st.write(f"수치형 변수 개수: {len(numeric_features)}")
    st.write(f"범주형 변수 개수: {len(categorical_features)}")

    # -----------------------------
    # 4. 전처리 파이프라인 정의
    # -----------------------------
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # sparse=False 로 dense matrix 반환 (Logistic, XGBoost 모두 사용 가능)
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # -----------------------------
    # 5. Train / Test 분할
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(y.unique()) <= 10 else None  # 이진/다중 분류일 때만 stratify
    )

    # -----------------------------
    # 6. 모델 정의
    # -----------------------------
    # Logistic Regression 파이프라인
    logi_clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    # XGBoost 파이프라인
    xgb_clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42
        ))
    ])

    # -----------------------------
    # 7. 하이퍼파라미터: 가중치 슬라이더
    # -----------------------------
    st.subheader("Hybrid 가중치 설정 (Logistic vs XGBoost)")
    w_log = st.slider("Logistic Regression 비중", 0.0, 1.0, 0.5, 0.1)
    w_xgb = 1.0 - w_log
    st.write(f"▶ Logistic 비중: {w_log:.2f}, XGBoost 비중: {w_xgb:.2f}")

    # -----------------------------
    # 8. 모델 학습 & 평가 버튼
    # -----------------------------
    if st.button("모델 학습 및 평가 실행"):
        with st.spinner("모델을 학습 중입니다..."):

            # 8-1. Logistic Regression
            logi_clf.fit(X_train, y_train)
            proba_log = logi_clf.predict_proba(X_test)[:, 1]
            y_pred_log = (proba_log >= 0.5).astype(int)
