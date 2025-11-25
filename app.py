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

    # -----------------------------
    # 4. 전처리 파이프라인
    # -----------------------------
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # ⚠️ Streamlit Cloud sklearn 버전이 낮아서 sparse=False 사용 불가!
    # → 최신/구버전 모두 지원되는 sparse_output=False 사용
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # -----------------------------
    # 5. Train / Test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # -----------------------------
    # 6. 모델 정의
    # -----------------------------
    logi_clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    xgb_clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ))
    ])

    # -----------------------------
    # 7. Hybrid 가중치 설정
    # -----------------------------
    st.subheader("Hybrid 가중치 설정")
    w_log = st.slider("Logistic Regression 비중", 0.0, 1.0, 0.5, 0.1)
    w_xgb = 1 - w_log
    st.write(f"Logistic 비중: {w_log:.2f}, XGBoost 비중: {w_xgb:.2f}")

    # -----------------------------
    # 8. 모델 학습
    # -----------------------------
    if st.button("모델 학습 및 평가 실행"):
        with st.spinner("모델 학습 중..."):

            # Logistic
            logi_clf.fit(X_train, y_train)
            proba_log = logi_clf.predict_proba(X_test)[:, 1]
            y_pred_log = (proba_log >= 0.5).astype(int)

            acc_log = accuracy_score(y_test, y_pred_log)
            auc_log = roc_auc_score(y_test, proba_log)

            # XGB
            xgb_clf.fit(X_train, y_train)
            proba_xgb = xgb_clf.predict_proba(X_test)[:, 1]
            y_pred_xgb = (proba_xgb >= 0.5).astype(int)

            acc_xgb = accuracy_score(y_test, y_pred_xgb)
            auc_xgb = roc_auc_score(y_test, proba_xgb)

            # Hybrid
            proba_hybrid = w_log * proba_log + w_xgb * proba_xgb
            y_pred_hybrid = (proba_hybrid >= 0.5).astype(int)

            acc_hybrid = accuracy_score(y_test, y_pred_hybrid)
            auc_hybrid = roc_auc_score(y_test, proba_hybrid)

        # -----------------------------
        # 9. 결과 출력
        # -----------------------------
        st.subheader("📊 모델 성능 비교")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Logistic Regression")
            st.write(f"Accuracy: **{acc_log:.4f}**")
            st.write(f"ROC AUC: **{auc_log:.4f}**")

        with col2:
            st.markdown("### XGBoost")
            st.write(f"Accuracy: **{acc_xgb:.4f}**")
            st.write(f"ROC AUC: **{auc_xgb:.4f}**")

        with col3:
            st.markdown("### Hybrid 모델")
            st.write(f"Accuracy: **{acc_hybrid:.4f}**")
            st.write(f"ROC AUC: **{auc_hybrid:.4f}**")

        st.markdown("---")
        st.subheader("Hybrid 모델 분류 리포트")
        st.text(classification_report(y_test, y_pred_hybrid))

else:
    st.info("CSV 파일을 먼저 업로드하세요.")
