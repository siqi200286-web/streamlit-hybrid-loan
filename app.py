import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier


st.title("Hybrid Model: Logistic Regression + XGBoost")

uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file is not None:
    st.success("파일 업로드 완료")
    df = pd.read_csv(uploaded_file, low_memory=False)
    st.write("데이터 미리보기:")
    st.dataframe(df.head())

    st.subheader("Target 변수 선택")
    target_col = st.selectbox("Target 변수를 선택하세요", df.columns)

    if target_col:
        y = df[target_col]
        X = df.drop(columns=[target_col])

        # y가 문자인 경우 숫자로 변환
        if y.dtype == "object":
            le = LabelEncoder()
            y = le.fit_transform(y)

        # bool → int
        bool_cols = X.select_dtypes(include=["bool"]).columns
        X[bool_cols] = X[bool_cols].astype(int)

        # object → 숫자로 변환 시도
        obj_cols = X.select_dtypes(include=["object"]).columns
        for col in obj_cols:
            X[col] = pd.to_numeric(
                X[col].astype(str)
                .str.replace("%", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip(),
                errors="coerce"
            )

        # 숫자형만 사용
        X = X.select_dtypes(include=["number"]).copy()

        # inf 제거
        X = X.replace([np.inf, -np.inf], np.nan)

        # NaN → 중앙값
        X = X.fillna(X.median())

        # 컬럼이 하나도 없으면 오류
        if X.shape[1] == 0:
            st.error("유효한 숫자형 변수가 없습니다. CSV 파일을 다시 확인하세요.")
        else:
            # Train/Test split
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
            logi_proba = logi.predict_proba(X_test_scaled)[:, 1]

            auc_logi = roc_auc_score(y_test, logi_proba)
            st.write("Logistic Regression ROC AUC:", round(auc_logi, 4))

            # XGBoost
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

            # Hybrid
            st.subheader("Hybrid Model 결과")

            weight = st.slider("Logistic 가중치 (XGBoost는 1-weight)", 0.0, 1.0, 0.5)
            hybrid_proba = weight * logi_proba + (1 - weight) * xgb_proba

            auc_hybrid = roc_auc_score(y_test, hybrid_proba)
            st.write("Hybrid ROC AUC:", round(auc_hybrid, 4))

            st.subheader("Classification Report")
            st.text(classification_report(y_test, (hybrid_proba >= 0.5).astype(int)))
