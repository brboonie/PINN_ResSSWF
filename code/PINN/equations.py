import tensorflow as tf

"""
Fundamental Equations of the Problem
"""

def Data_Equations(t, z, x, y, Y, neural_net):
   """The equations for matching neural net prediction and observed data"""

   u_data = Y[..., 0:1]
   v_data = Y[..., 1:2]
   p_data = Y[..., 2:3]
   nn_forward = neural_net(tf.concat([t,z,x,y],1))
   u_pred = nn_forward[..., 0:1]
   v_pred = nn_forward[..., 1:2]
   p_pred = nn_forward[..., 2:3]
   print("Data_Equation_run")
   return u_data - u_pred, v_data - v_pred, p_data - p_pred



def Data_Equations_grad(t, z, x, y, Y, combined_grad_pre, neural_net):

    u_data = Y[..., 0:1]
    v_data = Y[..., 1:2]
    p_data = Y[..., 2:3]

    du_dx = combined_grad_pre[..., 0:1]
    du_dy = combined_grad_pre[..., 1:2]
    dv_dx = combined_grad_pre[..., 2:3]
    dv_dy = combined_grad_pre[..., 3:4]
    dp_dx = combined_grad_pre[..., 4:5]
    dp_dy = combined_grad_pre[..., 5:6]

    with tf.GradientTape(persistent=True) as tg:

       tg.watch(t)  # define the variable with respect to which you want to take derivative
       tg.watch(z)
       tg.watch(x)
       tg.watch(y)

       nn_forward = neural_net(tf.concat([t, z, x, y], 1))
       u_pred = nn_forward[..., 0:1]
       v_pred = nn_forward[..., 1:2]
       p_pred = nn_forward[..., 2:3]



    upred_x = tg.gradient(u_pred, x)
    vpred_x = tg.gradient(v_pred, x)
    upred_y = tg.gradient(u_pred, y)
    vpred_y = tg.gradient(v_pred, y)
    ppred_x = tg.gradient(p_pred, x)
    ppred_y = tg.gradient(p_pred, y)


    return u_data - u_pred, v_data - v_pred, p_data - p_pred, du_dx - upred_x, du_dy-upred_y, dv_dx - vpred_x, dv_dy-vpred_y, dp_dx- ppred_x, dp_dy - ppred_y, u_data*v_pred-u_pred*v_data





def Inverse_1stOrder_Equations():
    
   ##########################
   # args: x_0,t_0,p_0,u_0,h_0,f,beta
   def inverse_1st_order(t, z, x, y, args, neural_net=None, drop_mass_balance: bool = False):
      
      x_0,t_0,z_0,u_0,p_0,f,beta,rho_dao = args

      with tf.GradientTape(persistent=True) as tg:
         tg.watch(t)  # define the variable with respect to which you want to take derivative
         tg.watch(z)
         tg.watch(x)
         tg.watch(y)
         
         nn_forward = neural_net(tf.concat([t,z,x,y],1))
         u = nn_forward[..., 0:1]
         v = nn_forward[..., 1:2]
         p = nn_forward[..., 2:3]
         w = nn_forward[..., 3:4]      
      
      u_t = tg.gradient(u, t)
      v_t = tg.gradient(v, t)
      u_x = tg.gradient(u, x)
      v_x = tg.gradient(v, x)
      u_y = tg.gradient(u, y)
      v_y = tg.gradient(v, y)
      p_x = tg.gradient(p, x)
      p_y = tg.gradient(p, y)
      u_z = tg.gradient(u, z)
      v_z = tg.gradient(v, z)
      w_z = tg.gradient(w, z)
      
      
      cor = f + beta*y*x_0
      # Momentum balance governing equations in horizontal direction
      ns_x = x_0/u_0/t_0*u_t + u*u_x + v*u_y + x_0/u_0/t_0*w*u_z - x_0/u_0*cor*v + (p_0/u_0**2)*rho_dao*p_x
      ns_y = x_0/u_0/t_0*v_t + u*v_x + v*v_y + x_0/u_0/t_0*w*v_z + x_0/u_0*cor*u + (p_0/u_0**2)*rho_dao*p_y
      print(inverse_1st_order)

      if drop_mass_balance:
         return ns_x, ns_y
      else:
         # mass balance governing equation
         cont = u_x + v_y + x_0/u_0/t_0*w_z
         return ns_x, ns_y, cont

   return inverse_1st_order
                