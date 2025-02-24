FROM tensorflow/tensorflow:latest-gpu-jupyter
# Crear y establecer el directorio de trabajo
WORKDIR /Proyectos/generacion_cartoon

ADD . /Proyectos/generacion_cartoon

# Instalar paquetes adicionales
RUN pip install --no-cache-dir -r requirements.txt


# Exponer el puerto de Jupyter
EXPOSE 8888

# Comando para ejecutar Jupyter automáticamente
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
