
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

st.title("Hybrid Model: Logistic Regression + XGBoost")

uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file:
    st.success("파일 업로드 완료")
    df = pd.read_csv(uploaded_file)
    st.write("데이터 미리보기:")
    st.dataframe(df.head())

    # Target 변수 선택
    st.subheader("Target 변수 선택")
    target_col = st.selectbox("Target 변수를 선택하세요", df.columns)

    if target_col:
        # Feature / Target 분리
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # bool 타입을 int로 변환
        for col in X.select_dtypes(include=["bool"]).columns:
            X[col] = X[col].astype(int)

        # 숫자형 데이터만 사용 (object 제거)
        X = X.select_dtypes(include=["number"]).copy()

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Logistic Regression
        st.subheader("Logistic Regression 결과")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        logi = LogisticRegression(max_iter=500)
        logi.fit(X_train_scaled, y_train)
        logi_pred = logi.predict_proba(X_test_scaled)[:, 1]

        st.write("Logistic Regression ROC AUC:", roc_auc_score(y_test, logi_pred))

        # XGBoost
        st.subheader("XGBoost 결과")
        xgb = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss"
        )
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict_proba(X_test)[:, 1]

        st.write("XGBoost ROC AUC:", roc_auc_score(y_test, xgb_pred))

        # Hybrid Model
        st.subheader("Hybrid Model")
        weight = st.slider("Logistic 가중치 (나머지는 XGBoost로 계산)", 0.0, 1.0, 0.5)

        hybrid_pred = weight * logi_pred + (1 - weight) * xgb_pred

        st.write("Hybrid ROC AUC:", roc_auc_score(y_test, hybrid_pred))

        # Classification Report 출력
        hybrid_label = (hybrid_pred > 0.5).astype(int)
        st.subheader("Classification Report")
        st.text(classification_report(y_test, hybrid_label))
