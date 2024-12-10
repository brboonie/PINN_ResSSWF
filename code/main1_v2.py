import sys
import time
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
from numpy import cos, sin

from PINN import create_mlp, SquareLoss, Adam, LBFGS
from equations import Inverse_1stOrder_Equations, Data_Equations_grad

# fix seed to remove randomness across multiple runs
np.random.seed(234)
tf.random.set_seed(234)

# Data type for calculation
_data_type = tf.float64

def to_tensor(var):
    return tf.constant(var, dtype=_data_type)

# 定义归一化函数
def normalize(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val) * 2 - 1

# 输入数据部分
def load_and_preprocess_data(filepath3, filepath4, filepath5, filet2m,filemsl,filebuoy, num_collocation_points_nodata, num_collocation_points_data):
    """
    从NetCDF文件加载并预处理数据

    Parameters:
    - filepath: str, NetCDF文件的路径
    - num_data_points: int, 从数据集中随机选择的数据点数量
    - num_collocation_points: int, 生成的配点数量

    Returns:
    - X_data: numpy.ndarray, 数据点的输入特征
    - Y_data: numpy.ndarray, 数据点的目标输出
    - collocation_pts: numpy.ndarray, 配点的输入特征
    """   
    # Processing data and create a mesh
    "*** YOUR CODE HERE ***"

    return X_data, Y_data, collocation_pts, weight_combined_buoy, combined_grad_extended, grad_weights_extended

def train():
    """
    Define Constants and Normalization Factors
    Set a whole bunch of parameters
    """

    omega = 7.2921e-5 # earth angular velocity
    R_e = 6378100 # Earth's radius in meters
    lat = 2 # latitude of storm center
    f = 2 * omega * sin(np.radians(lat)) # coriolis parameter
    beta = 2 * omega * cos(np.radians(lat)) / R_e # beta value
    rho_dao = 0.84    # 1/rho均值
    u_0 = 17 # m/s, maximum value for scaling velocity
    x_0 = 2500000 # m domain width to scale x and y to -1 to 1

    p_max = 1024*100 # max geopotential
    p_min = 1007*100 # min geopotential
    p_0 = 102400 - 100700 # scaling parameter for gepotential #m^2 / s^2
    t_0 = 3124 # time scale by 6 hours to get time to -1 to 1

    z_min = 0 # min pressure of system
    z_max = 20 # max pressure of system
    z_0 = 20 # pressure scaling parameter units Pa
    # # set the bounds of the domain (X)

    # lx = np.array([-x_0, -x_0, -t_0, p_min*100])
    # ux = np.array([x_0, x_0, t_0, p_max*100])

    lx = np.array([-t_0, z_min, -x_0, -x_0])
    ux = np.array([t_0,z_max, x_0, x_0])

    # set the bounds for the target fields (Y)
    ly = np.array([-u_0, -u_0, p_min])
    uy = np.array([u_0, u_0, p_max])

    # 数据文件路径
    # filepath1 = './data1/cmems_obs-wind_glo_phy_nrt_l3-hy2c-hscat-asc-0.25deg_P1D-i_1719407808395.nc'
    # filepath2 = './data1/cmems_obs-wind_glo_phy_nrt_l3-hy2b-hscat-asc-0.25deg_P1D-i_1719407690443.nc'
    filepath3 = './data/cmems_obs-wind_glo_phy_nrt_l3-metopb-ascat-asc-0.25deg_P1D-i_1720081364774.nc'
    filepath4 = './data/cmems_obs-wind_glo_phy_nrt_l3-hy2d-hscat-asc-0.25deg_P1D-i_1720081131356.nc'
    filepath5 = './data/cmems_obs-wind_glo_phy_nrt_l3-metopc-ascat-asc-0.25deg_P1D-i_1720080937787.nc'
    filet2m = './data/era5_t2m_150_to_200.nc'
    filemsl = './data/era5_msl_150_to_200.nc'
    filebuoy = './data/BUOY_choose.csv'
    # filepath6 = './data1/cmems_obs-wind_glo_phy_nrt_l3-hy2c-hscat-des-0.25deg_P1D-i_1719407865000.nc'
    # filepathgeo = './data/era5_combined_150_to_200.nc'

    # 各个图配点的数量
    num_collocation_points_nodata = 3000
    num_collocation_points_data = 500
    # Paramters
    num_iterations_adam = 20000
    num_iterations_lbfgs = 180000
    gamma = 0.8
    num_hlayers = 8
    layer_size = 100
    input_size = 4

    layers = [layer_size for i in range(num_hlayers)]
    layers.append(4) # output layer
    lyscl = [1 for i in range(len(layers))]

    X_data, Y_data, collocation_pts, weights, combined_grad_pre, grad_weights = load_and_preprocess_data(filepath3, filepath4, filepath5, filet2m, filemsl, filebuoy,num_collocation_points_nodata, num_collocation_points_data)


    collocation_pts = (collocation_pts - lx)/(ux - lx) * 2 - 1
    grad_weights = grad_weights / np.nanmean(grad_weights)

    #  the relative weights of each data point in PINN training
    weights = weights/np.mean(weights)

    # Arguments that will be used by the PINN equations
    args = x_0,t_0,z_0,u_0,p_0,f,beta,rho_dao

    # Create the MLP, the function for the equations, and the loss function
    model = create_mlp(layers, lyscl, dtype=_data_type, input_size=input_size)
    equations = Inverse_1stOrder_Equations()
    # equations = Inverse_1stOrder_Equations_terms()
    loss = SquareLoss(equations=equations, equations_data=Data_Equations_grad, args=args, gamma=gamma, combined_grad_pre = combined_grad_pre)

   

    # train the model for Adam optimizer first, then LBFGS
    start_time = time.time()
    adam = Adam(
        net=model, loss=loss, collocation_points=collocation_pts,
        data_points=(to_tensor(X_data), to_tensor(Y_data)), weights = to_tensor(weights), grad_weights = to_tensor(grad_weights)
    )
    
    adam.optimize(nIter=num_iterations_adam)
    lbfgs = LBFGS(
        net=model, loss=loss, collocation_points=collocation_pts,
        data_points=(to_tensor(X_data), to_tensor(Y_data)), weights=to_tensor(weights), grad_weights = to_tensor(grad_weights)
    )

    lbfgs.optimize(nIter=num_iterations_lbfgs)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % elapsed) # Print training time

    # Access loss history
    total_losses = np.array(adam.loss_records.get("loss", []) + lbfgs.loss_records.get("loss", []))
    equation_losses = np.array(adam.loss_records.get("loss_equation", []) + lbfgs.loss_records.get("loss_equation", []))
    data_losses = np.array(adam.loss_records.get("loss_data", []) + lbfgs.loss_records.get("loss_data", []))

    # Add code to save model
    np.savetxt("./total_losses.csv", total_losses, delimiter=",")
    np.savetxt("./equation_losses.csv", equation_losses, delimiter=",")
    np.savetxt("./data_losses.csv", data_losses, delimiter=",")
    model.save('./my_PINN_model.h5')


if __name__ == '__main__':
    train()    
