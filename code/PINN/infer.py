from tensorflow.keras.models import load_model
from PINN import TunalbeXavierNormal, SquareLoss, Adam, LBFGS, create_mlp
from equations import Inverse_1stOrder_Equations, Data_Equations
import sys
import time
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
from numpy import cos, sin

if __name__ == '__main__':
    model_path = './my_PINN_model.h5'
    model = load_model(model_path, custom_objects={
        'TunalbeXavierNormal': TunalbeXavierNormal,
        'SquareLoss': SquareLoss,
        'Adam': Adam,
        'LBFGS': LBFGS,
        'Inverse_1stOrder_Equations': Inverse_1stOrder_Equations,
        'Data_Equations': Data_Equations
    })


    TIME_E = [-3072, 384, 3072]
    LEVEL_E = 10
    LAT_E = np.arange(-2500000, 2502000, 25000)
    LON_E = np.arange(-2500000, 2502000, 25000)
    TIME, LEV, LAT, LONG = np.meshgrid(TIME_E, LEVEL_E, LAT_E, LON_E, indexing='ij')
    X_data = np.stack([TIME.ravel(), LEV.ravel(), LAT.ravel(), LONG.ravel()], axis=1)

    omega = 7.2921e-5 # earth angular velocity
    R_e = 6378100 # Earth's radius in meters
    lat = 21 # latitude of storm center
    f = 2 * omega * sin(np.radians(lat)) # coriolis parameter
    beta = 2 * omega * cos(np.radians(lat)) / R_e # beta value
    u_0 = 17 # m/s, maximum value for scaling velocity
    x_0 = 2500000 # m domain width to scale x and y to -1 to 1

    p_max = 1024*100 # max geopotential
    p_min = 1007*100 # min geopotential
    p_0 = p_max - p_min # scaling parameter for gepotential #m^2 / s^2
    t_0 = 3072 # time scale by 6 hours to get time to -1 to 1

    z_min = 0# min pressure of system
    z_max = 20 # max pressure of system
    z_0 = 10 # pressure scaling parameter units Pa
    # # set the bounds of the domain (X)

    # lx = np.array([-x_0, -x_0, -t_0, p_min*100])
    # ux = np.array([x_0, x_0, t_0, p_max*100])

    lx = np.array([-t_0, z_min, -x_0, -x_0])
    ux = np.array([t_0,z_max, x_0, x_0])

    # set the bounds for the target fields (Y)
    ly = np.array([-u_0, -u_0, p_min, -10])
    uy = np.array([u_0, u_0, p_max,10])

    X_data2 = (X_data - lx) / (ux - lx) * 2 - 1

    predi_normalize = model.predict(X_data2)
    print(np.max(predi_normalize[:, 0]))
    print(np.min(predi_normalize[:, 0]))
    print(np.max(predi_normalize[:, 1]))
    print(np.min(predi_normalize[:, 1]))
    # 设限
    predi_normalize[predi_normalize > 1]=1
    predi_normalize[predi_normalize < -1] = -1

    # 数据去归一化
    predi = ((predi_normalize + 1) / 2) * (uy - ly) + ly

    predi_flat = predi.reshape(-1, predi.shape[-1])
    # 将预测结果数组转换为 DataFrame
    predi_df = pd.DataFrame(predi_flat, columns=['u', 'v', 'p', 'w_nouse'])

    # 保存 DataFrame 到 CSV 文件
    predi_df.to_csv('./predictions.csv', index=False)
