import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier


st.title("Hybrid Model: Logistic Regression + XGBoost")

# CSV 업로드
uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file is not None:
    st.success("파일 업로드 완료")
    df = pd.read_csv(uploaded_file, low_memory=False)
    st.write("데이터 미리보기:")
    st.dataframe(df.head())

    # Target 변수 선택
    st.subheader("Target 변수 선택")
    target_col = st.selectbox("Target 변수를 선택하세요", df.columns)

    if target_col:
        # y, X 분리
        y = df[target_col]
        X = df.drop(columns=[target_col])

        # y가 문자열(object)이면 LabelEncoder로 0/1 등 숫자로 변환
        if y.dtype == "object":
            le = LabelEncoder()
            y = le.fit_transform(y)

        # bool 타입을 int로 변환
        bool_cols = X.select_dtypes(include=["bool"]).columns
        X[bool_cols] = X[bool_cols].astype(int)

        # 숫자형 컬럼만 사용
        X = X.select_dtypes(include=["number"]).copy()

        # 무한대 값 처리
        X = X.replace([np.inf, -np.inf], np.nan)

        # NaN 결측치 처리: 각 컬럼의 중앙값으로 대체
        X = X.fillna(X.median())

        # Train / Test 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # -----------------------------
        # Logistic Regression
        # -----------------------------
        st.subheader("Logistic Regression 결과")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        logi = LogisticRegression(max_iter=500)
        logi.fit(X_train_scaled, y_train)
        logi_proba = logi.predict_proba(X_test_scaled)[:, 1]

        auc_logi = roc_auc_score(y_test, logi_proba)
        st.write("Logistic Regression ROC AUC:", round(auc_logi, 4))

        # -----------------------------
        # XGBoost
        # -----------------------------
        st.subheader("XGBoost 결과")

        xgb = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
        )
        xgb.fit(X_train, y_train)
        xgb_proba = xgb.predict_proba(X_test)[:, 1]

        auc_xgb = roc_auc_score(y_test, xgb_proba)
        st.write("XGBoost ROC AUC:", round(auc_xgb, 4))

        # -----------------------------
        # Hybrid Model
        # -----------------------------
        st.subheader("Hybrid Model 결과")

        weight = st.slider(
            "Hybrid에서 Logistic 가중치 (나머지는 XGBoost)",
            0.0,
            1.0,
            0.5,
            0.05,
        )

        hybrid_proba = weight * logi_proba + (1 - weight) * xgb_proba
        auc_hybrid = roc_auc_score(y_test, hybrid_proba)
        st.write("Hybrid ROC AUC:", round(auc_hybrid, 4))

        # 하이브리드 분류 레이블
        hybrid_label = (hybrid_proba >= 0.5).astype(int)

        st.subheader("Hybrid Classification Report")
        st.text(classification_report(y_test, hybrid_label))
