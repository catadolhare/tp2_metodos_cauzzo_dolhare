import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Funciones generadas automáticamente
def fAndG_dw(b, d, i, w):
    if isinstance(b, np.ndarray):
        dim = b.shape
        assert dim == (1, )
    if isinstance(d, np.ndarray):
        dim = d.shape
        assert dim == (1, )
    assert isinstance(i, np.ndarray)
    dim = i.shape
    assert len(dim) == 1
    i_rows = dim[0]
    assert isinstance(w, np.ndarray)
    dim = w.shape
    assert len(dim) == 1
    w_rows = dim[0]
    assert i_rows == w_rows

    t_0 = np.tanh((b + (w).dot(i)))
    t_1 = (((1 + t_0) / 2) - d)
    functionValue = (t_1 ** 2)
    gradient = (((1 - (t_0 ** 2)) * t_1) * i)

    return functionValue, gradient

def fAndG_db(b, d, i, w):
    if isinstance(b, np.ndarray):
        dim = b.shape
        assert dim == (1, )
    if isinstance(d, np.ndarray):
        dim = d.shape
        assert dim == (1, )
    assert isinstance(i, np.ndarray)
    dim = i.shape
    assert len(dim) == 1
    i_rows = dim[0]
    assert isinstance(w, np.ndarray)
    dim = w.shape
    assert len(dim) == 1
    w_rows = dim[0]
    assert i_rows == w_rows

    t_0 = np.tanh((b + (w).dot(i)))
    t_1 = (((1 + t_0) / 2) - d)
    functionValue = (t_1 ** 2)
    gradient = ((1 - (t_0 ** 2)) * t_1)

    return functionValue, gradient

def abrirImagenesEscaladas(carpeta, escala):
    imagenes = []
    for dirpath, dirnames, filenames in os.walk(carpeta):
        for file in filenames:
            file_path = os.path.join(dirpath, file)
            if os.path.isfile(file_path):
                img = Image.open(file_path)
                img = img.resize((escala, escala))
                img = img.convert('L')
                img = np.asarray(img)
                img = img.reshape((escala**2)) / 255.0
                imagenes.append(img)
    return imagenes

def dL_dw(w, b, imagenes, diagnosticos):
    sumatoria = np.zeros_like(w)
    for imagen in range(len(imagenes)):
        _, gradient = fAndG_dw(b, diagnosticos[imagen], imagenes[imagen], w)
        sumatoria += gradient
    sumatoria /= len(imagenes)
    return sumatoria

def dL_db(w, b, imagenes, diagnosticos):
    sumatoria = 0
    for imagen in range(len(imagenes)):
        _, gradient = fAndG_db(b, diagnosticos[imagen], imagenes[imagen], w)
        sumatoria += gradient
    sumatoria /= len(imagenes)
    return sumatoria

def func_L(w, b, imagenes, diagnosticos):
    sumatoria = 0
    for imagen in range(len(imagenes)):
        t_0 = np.tanh((w).dot(imagenes[imagen]) + b)
        t_1 = (((1 + t_0) / 2) - diagnosticos[imagen])
        t_2 = t_1 * t_1
        sumatoria += t_2
    return sumatoria

def f(i, w, b):
    tan = np.tanh(np.dot(w, i) + b)
    tan_mas_uno = tan + 1
    return tan_mas_uno/2

def error_cuadratico_medio(imagenes, diagnosticos, w, b):
    error_total = 0
    for i in imagenes:
        error = (f(i, w, b) - diagnosticos[i])**2
        error_total += error

    return error_total/len(imagenes)

def gradiente_descendiente():
    carpeta_normal = "chest_xray/train/NORMAL"
    imagenes_normal = abrirImagenesEscaladas(carpeta_normal, 64)
    print(len(imagenes_normal))
    diagnosticos_normal = [0] * len(imagenes_normal)

    carpeta_neumonia = "chest_xray/train/NEUMONIA"
    imagenes_neumonia = abrirImagenesEscaladas(carpeta_neumonia, 64)
    print(len(imagenes_neumonia))
    diagnosticos_neumonia = [1] * len(imagenes_neumonia)

    imagenes = imagenes_normal + imagenes_neumonia
    diagnosticos = diagnosticos_normal + diagnosticos_neumonia
    print(diagnosticos)

    w0 = np.zeros(4096)
    b0 = 0
    alpha = 0.001

    MAX_ITER = 1000
    TOLERANCIA = 1e-8
    iter = 0

    w = w0
    b = b0

    errores = []
    
    while iter <= MAX_ITER:
        print("Iteración: ", iter, "- Mínimo alcanzado hasta el momento: ", func_L(w, b, imagenes, diagnosticos))

        gradient_dw = dL_dw(w, b, imagenes, diagnosticos)
        gradient_db = dL_db(w, b, imagenes, diagnosticos)

        prox_w = w - alpha * gradient_dw
        prox_b = b - alpha * gradient_db

        error = error_cuadratico_medio(imagenes, diagnosticos, prox_w, prox_b)
        errores.append(error)

        diferencia = func_L(prox_w, prox_b, imagenes, diagnosticos) - func_L(w, b, imagenes, diagnosticos)
        
        if abs(diferencia) < TOLERANCIA:
            break
    
        w = prox_w
        b = prox_b

        iter += 1
    
    return w, b, errores

def main():
    w, b, errores = gradiente_descendiente()
    print("El vector w que minimiza es: ", w)
    print("El escalar b que minimiza es: ", b)
    print(errores)

if __name__ == "__main__":
    main()
