import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.title("Detector Burnout Empleados")
st.markdown("Pipeline ROC 1.000 - Prediccion individual y analisis empresa")

#cargar pipeline
@st.cache_data
def cargar_modelo():
    with st.spinner("Cargando pipeline.."):
        return pickle.load(open('burnout_pipeline.pkl', 'rb'))

pipeline = cargar_modelo()

#Sidebar: Input empleado individual
st.sidebar.header("Nuevo Empleado")
age = st.sidebar.slider("Edad", 18, 65, 32)
horas = st.sidebar.slider("Horas/semana", 30, 80, 60)
experience = st.sidebar.slider("Anos experiencia", 0, 30, 5)
remote = st.sidebar.slider("Remote %", 0.0, 1.0, 0.5, 0.1)
satisfaction = st.sidebar.slider("Satisfaccion (1-10)", 1.0, 10.0, 2.5)
jobrole = st.sidebar.selectbox("Rol", ['Engineer', 'Manager', 'Sales', 'HR', 'Data Scientist'])
gender = st.sidebar.selectbox("Genero", ['Male', 'Female'])

#prediccion individual
if st.sidebar.button("PREDECIR RIESGO"):
    nuevo_empleado = pd.DataFrame({
        'Age': [age], 'WorkHoursPerWeek': [horas], 'Experience': [experience], 'RemoteRatio': [remote], 'SatisfactionLevel': [satisfaction], 'StressLevel': [stress], 'JobRole': [jobrole], 'Gender': [gender]
    })
    
    riesgo = pipeline.predict_proba(nuevo_empleado)[:,1][0]

    st.markdown("---")
    st.subheader("Resultado INDIVIDUAL")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("riesgo burnout", f"{riesgo:.1%}")
    with col2:
        color = "🔴" if riesgo > 0.8 else "🟡" if riesgo > 0.5 else "🟢"
        st.markdown(f"**{color} {riesgo:.0%}**")
    with col3:
        accion = "URGENTE" if riesgo > 0.8 else "MONITOREAR" if riesgo > 0.5 else "OK"
        st.markdown(f"**{accion}**")

#analisis empresa
st.markdown("---")
st.subheader("Analisis Empresa Completa (2k empleados)")

#cargar mis predicciones
@st.cache_data
def cargar_riesgos():
    df = pd.read_csv('synthetic_employee_burnout.csv')
    X_empresa = df.drop('Burnout', axis=1)
    riesgos = pipeline.predict_proba(X_empresa)[:,1]
    return pd.DataFrame({
        'Empleado_ID': range(1, len(riesgos)+1),
        'Riesgo_Burnout': riesgos,
        'Age': X_empresa['Age'],
        'WorkHoursPerWeek': X_empresa['WorkHoursPerWeek'],
        'StressLevel': X_empresa['StressLevel']
    })

analisis = cargar_riesgos()

st.metric("Riesgo Promedio", f"{analisis['Riesgo_Burnout'].mean():.1%}")
st.metric("Empleados Criticos (>80%)", f"{(analisis['Riesgo_Burnout'] > 0.8).sum()}")

#graficos
fig_hist = px.histogram(analisis, x='Riesgo_Burnout', nbins=50, title="Distribucion Riesgos")
st.plotly_chart(fig_hist, use_container_width=True)

fig_top = px.histogram(analisis.nlargest(10, 'Riesgo_Burnout'), title="TOP 10 mas criticos")
st.plotly_chart(fig_top, use_container_width=True)
