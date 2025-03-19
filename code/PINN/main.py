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
def load_and_preprocess_data(filepath3, filepath4, filepath5, filet2m,filemsl, num_collocation_points_nodata, num_collocation_points_data):
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
    # # read satellite data
    data3 = xr.open_dataset(filepath3)
    U_star3 = data3['eastward_wind'] # X*Y
    V_star3 = data3['northward_wind']  # X*Y
    data4 = xr.open_dataset(filepath4)
    U_star4 = data4['eastward_wind'] # X*Y
    V_star4 = data4['northward_wind']  # X*Y
    data5 = xr.open_dataset(filepath5)
    U_star5 = data5['eastward_wind'] # X*Y
    V_star5 = data5['northward_wind']  # X*Y   

    U_star_combined = xr.concat([U_star3, U_star4, U_star5], dim='time')
    V_star_combined = xr.concat([V_star3, V_star4, V_star5], dim='time')


    # 对 U_star_combined and V_star_combined  进行归一化
    U_star_normalized = normalize(U_star_combined, -17, 17)
    V_star_normalized = normalize(V_star_combined, -17, 17)

    # 展平U_star_normalize和V_star_normalized
    U_star_combined_flattened = U_star_normalized.values.flatten()
    V_star_combined_flattened = V_star_normalized.values.flatten()


    # 读取2m温度t2m
    datat2m = xr.open_dataset(filet2m)
    t2m_data = datat2m['t2m']
    # 按照 latitude 维度进行排序，t2m从小到大排列
    t2m_data_sorted = t2m_data.sortby('latitude')

    # 读取海平面气压msl
    datamsl = xr.open_dataset(filemsl)
    msl_data = datamsl['msl']
    # 按照 latitude 维度进行排序，msl从小到大排列
    msl_data_sorted = msl_data.sortby('latitude')


    # msl归一化
    msl_star_normalized = normalize(msl_data_sorted, 1007*100,1024*100)
    msl_sorted_flattened = msl_star_normalized.values.flatten()
    # 将 z_sorted_flattened 中对应位置设置为 NaN
    nan_indices = np.isnan(U_star_combined_flattened)
    msl_sorted_flattened[nan_indices] = np.nan



    # 计算u_x,u_y,v_x,v_y,z_x,z_y梯度，即utrue梯度
    # 定义 x 和 y 方向上的网格间距（假设为 25000 米）
    dx = 0.01  # 经度方向的间距
    dy = 0.01  # 纬度方向的间距

    # 初始化用于存储梯度的数组
    dU_dx = np.zeros_like(U_star_normalized)
    dU_dy = np.zeros_like(U_star_normalized)
    # 在每个时间点计算梯度
    for t in range(U_star_normalized.shape[0]):
        # 计算 x 方向的梯度
        dU_dx[t, :, :] = np.gradient(U_star_normalized[t, :, :], dx, axis=0)
        # 计算 y 方向的梯度
        dU_dy[t, :, :] = np.gradient(U_star_normalized[t, :, :], dy, axis=1)

    # 初始化用于存储梯度的数组
    dV_dx = np.zeros_like(V_star_normalized)
    dV_dy = np.zeros_like(V_star_normalized)
    # 在每个时间点计算梯度
    for t in range(V_star_normalized.shape[0]):
        # 计算 x 方向的梯度
        dV_dx[t, :, :] = np.gradient(V_star_normalized[t, :, :], dx, axis=0)
        # 计算 y 方向的梯度
        dV_dy[t, :, :] = np.gradient(V_star_normalized[t, :, :], dy, axis=1)

    # 初始化用于存储梯度的数组
    dp_dx = np.zeros_like(msl_star_normalized)
    dp_dy = np.zeros_like(msl_star_normalized)
    # 在每个时间点计算梯度
    for t in range(msl_star_normalized.shape[0]):
        # 计算 x 方向的梯度
        dp_dx[t, :, :] = np.gradient(msl_star_normalized[t, :, :], dx, axis=0)
        # 计算 y 方向的梯度
        dp_dy[t, :, :] = np.gradient(msl_star_normalized[t, :, :], dy, axis=1)

    dU_dx_star_combined_flattened = dU_dx.flatten()
    dU_dy_star_combined_flattened = dU_dy.flatten()
    dV_dx_star_combined_flattened = dV_dx.flatten()
    dV_dy_star_combined_flattened = dV_dy.flatten()
    dp_dx_star_combined_flattened = dp_dx.flatten()
    dp_dy_star_combined_flattened = dp_dy.flatten()

    grad_wind_speed = np.sqrt(dU_dx_star_combined_flattened ** 2 + dU_dy_star_combined_flattened ** 2 + dV_dx_star_combined_flattened**2 + dV_dy_star_combined_flattened**2)


    nan_indices_du = np.isnan(dU_dy_star_combined_flattened)
    dU_dx_star_combined_flattened[nan_indices_du] = np.nan

    # utrueuvz在x,y方向上的梯度数组
    combined_grad = np.stack(
        [dU_dx_star_combined_flattened, dU_dy_star_combined_flattened, dV_dx_star_combined_flattened, dV_dy_star_combined_flattened, dp_dx_star_combined_flattened, dp_dy_star_combined_flattened], axis=-1)




    # 创建X_data网格
    TIME_E = [-3124, -199, 3124]
    HEIGHT_E = 10
    LAT_E = np.arange((-len(X_star)/2),(len(X_star)/2))*25000
    LON_E = np.arange((-len(Y_star)/2),(len(Y_star)/2))*25000
    TIME, HEI, LAT, LONG = np.meshgrid(TIME_E, HEIGHT_E, LAT_E, LON_E, indexing='ij')
    X_data1 = np.stack([TIME.ravel(), HEI.ravel(), LAT.ravel(), LONG.ravel()], axis=-1)

    lx = np.array([-3124, 0, -2500000, -2500000])
    ux = np.array([3124, 20, 2500000, 2500000])

    # collocation points so everything is between -1 and 1
    X_data_normalized = (X_data1 - lx)/(ux - lx) * 2 - 1

    Y_data1 = np.column_stack((U_star_combined_flattened, V_star_combined_flattened, msl_sorted_flattened))
    wind_speed = np.sqrt(U_star_combined_flattened ** 2 + V_star_combined_flattened ** 2)


    # 找出 combined_array 第一列中 NaN 值的位置
    nan_indices_combined = np.isnan(combined_grad[:, 0])
    # 找出 Y_data1 第一列中 NaN 值的位置
    nan_indices_Y_data1 = np.isnan(Y_data1[:, 0])
    # 数据点去除NAN值
    nan_indices2 = nan_indices_combined | nan_indices_Y_data1


    # 数据点过多，内存爆炸，在数据点中随意挑选6000点
    data_where = np.where(~nan_indices2)[0]
    random_data_cleaned = np.random.choice(data_where, 7000, replace=False)
    X_data_pre = X_data_normalized[random_data_cleaned]
    Y_data_pre = Y_data1[random_data_cleaned]
    weight = wind_speed[random_data_cleaned]
    grad_weights = grad_wind_speed[random_data_cleaned]
    combined_grad_pre = combined_grad[random_data_cleaned]

    X_data = tf.constant(X_data_pre, dtype=tf.float64)
    Y_data = tf.constant(Y_data_pre, dtype=tf.float64)# data points



    # 搭配点
    # 在无数据点的地方随机索引3000点
    nan_indices_row1 = np.where(nan_indices2)[0]
    # 从这些索引中随机选择 3000 个点
    random_indices1 = np.random.choice(nan_indices_row1, 10000, replace=False)
    # 在有数据点的地方随机索引500点
    nan_indices_row2 = np.where(~nan_indices2)[0]
    # 从这些索引中随机选择 3000 个点
    random_indices2 = np.random.choice(nan_indices_row2, 2000, replace=False)
    combined_idx2 = np.concatenate((random_indices1, random_indices2))

    collocation_pts1 = X_data1[combined_idx2,:]
    collocation_pts = tf.constant(collocation_pts1, dtype=tf.float64)



    return X_data, Y_data, collocation_pts, weight, combined_grad_pre, grad_weights

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

    X_data, Y_data, collocation_pts, weights, combined_grad_pre, grad_weights = load_and_preprocess_data(filepath3, filepath4, filepath5, filet2m, filemsl, num_collocation_points_nodata, num_collocation_points_data)


    # # 进行归一化
    # combined_grad_pre = (combined_grad_pre - min_vals) / (max_vals - min_vals) * 2 - 1
    #
    #
    # # scale the X and Y values of the data points and
    # # collocation points so everything is between -1 and 1
    # X_data = (X_data - lx)/(ux - lx) * 2 - 1
    # Y_data = (Y_data - ly)/(uy -ly) * 2 - 1
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
        data_points=(to_tensor(X_data), to_tensor(Y_data)), weights = to_tensor(weights), grad_weights = grad_weights
    )
    
    adam.optimize(nIter=num_iterations_adam)
    lbfgs = LBFGS(
        net=model, loss=loss, collocation_points=collocation_pts,
        data_points=(to_tensor(X_data), to_tensor(Y_data)), weights=to_tensor(weights), grad_weights = grad_weights
    )

    lbfgs.optimize(nIter=num_iterations_lbfgs)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % elapsed) # Print training time

    # Access loss history
    total_losses = np.array(adam.loss_records.get("loss", []) + lbfgs.loss_records.get("loss", []))
    equation_losses = np.array(adam.loss_records.get("loss_equation", []) + lbfgs.loss_records.get("loss_equation", []))
    data_losses = np.array(adam.loss_records.get("loss_data", []) + lbfgs.loss_records.get("loss_data", []))

    # Add code to save your model if desired
    np.savetxt("./total_losses.csv", total_losses, delimiter=",")
    np.savetxt("./equation_losses.csv", equation_losses, delimiter=",")
    np.savetxt("./data_losses.csv", data_losses, delimiter=",")
    model.save('./my_PINN_model.h5')


if __name__ == '__main__':
    train()    
