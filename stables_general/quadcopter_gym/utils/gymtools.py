import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation

# import utils
from .rotationConversion import quatToYPR_ZYX, quat2Dcm
from .. import config

rad2deg = 180.0/pi
deg2rad = pi/180.0
rads2rpm = 60.0/(2.0*pi)
rpm2rads = 2.0*pi/60.0

def makeGymFigures(params, time, pos_all, vel_all, quat_all, omega_all, euler_all, commands, wMotor_all, thrust, torque, reward):

    x    = pos_all[:,0]
    y    = pos_all[:,1]
    z    = pos_all[:,2]
    q0   = quat_all[:,0]
    q1   = quat_all[:,1]
    q2   = quat_all[:,2]
    q3   = quat_all[:,3]
    xdot = vel_all[:,0]
    ydot = vel_all[:,1]
    zdot = vel_all[:,2]
    p    = omega_all[:,0]*rad2deg
    q    = omega_all[:,1]*rad2deg
    r    = omega_all[:,2]*rad2deg

    wM1  = wMotor_all[:,0]*rads2rpm
    wM2  = wMotor_all[:,1]*rads2rpm
    wM3  = wMotor_all[:,2]*rads2rpm
    wM4  = wMotor_all[:,3]*rads2rpm

    phi   = euler_all[:,0]*rad2deg
    theta = euler_all[:,1]*rad2deg
    psi   = euler_all[:,2]*rad2deg

    # x_sp  = sDes_calc[:,0]
    # y_sp  = sDes_calc[:,1]
    # z_sp  = sDes_calc[:,2]
    # Vx_sp = sDes_calc[:,3]
    # Vy_sp = sDes_calc[:,4]
    # Vz_sp = sDes_calc[:,5]
    # x_thr_sp = sDes_calc[:,6]
    # y_thr_sp = sDes_calc[:,7]
    # z_thr_sp = sDes_calc[:,8]
    # q0Des = sDes_calc[:,9]
    # q1Des = sDes_calc[:,10]
    # q2Des = sDes_calc[:,11]
    # q3Des = sDes_calc[:,12]    
    # pDes  = sDes_calc[:,13]*rad2deg
    # qDes  = sDes_calc[:,14]*rad2deg
    # rDes  = sDes_calc[:,15]*rad2deg

    # x_tr  = sDes_traj[:,0]
    # y_tr  = sDes_traj[:,1]
    # z_tr  = sDes_traj[:,2]
    # Vx_tr = sDes_traj[:,3]
    # Vy_tr = sDes_traj[:,4]
    # Vz_tr = sDes_traj[:,5]
    # Ax_tr = sDes_traj[:,6]
    # Ay_tr = sDes_traj[:,7]
    # Az_tr = sDes_traj[:,8]
    # yaw_tr = sDes_traj[:,14]*rad2deg

    uM1 = commands[:,0]*rads2rpm
    uM2 = commands[:,1]*rads2rpm
    uM3 = commands[:,2]*rads2rpm
    uM4 = commands[:,3]*rads2rpm

    # x_err = x_sp - x
    # y_err = y_sp - y
    # z_err = z_sp - z

    # psiDes   = np.zeros(q0Des.shape[0])
    # thetaDes = np.zeros(q0Des.shape[0])
    # phiDes   = np.zeros(q0Des.shape[0])
    # for ii in range(q0Des.shape[0]):
    #     YPR = quatToYPR_ZYX(sDes_calc[ii,9:13])
    #     psiDes[ii]   = YPR[0]*rad2deg
    #     thetaDes[ii] = YPR[1]*rad2deg
    #     phiDes[ii]   = YPR[2]*rad2deg
    
    plt.show()

    plt.figure()
    plt.plot(time, x, time, y, time, z)
    plt.grid(True)
    plt.legend(['x','y','z'])
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.draw()

    plt.figure()
    plt.plot(time, xdot, time, ydot, time, zdot)
    plt.grid(True)
    plt.legend(['Vx','Vy','Vz'])
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.draw()

    plt.figure()
    plt.plot(time, phi, time, theta, time, psi)
    plt.grid(True)
    plt.legend(['roll','pitch','yaw'])
    plt.xlabel('Time (s)')
    plt.ylabel('Euler Angle (°)')
    plt.draw()
    
    plt.figure()
    plt.plot(time, p, time, q, time, r)
    plt.grid(True)
    plt.legend(['p','q','r'])
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (°/s)')
    plt.draw()

    plt.figure()
    plt.plot(time, wM1, time, wM2, time, wM3, time, wM4)
    plt.plot(time, uM1, '--', time, uM2, '--', time, uM3, '--', time, uM4, '--')
    plt.grid(True)
    plt.legend(['w1','w2','w3','w4'])
    plt.xlabel('Time (s)')
    plt.ylabel('Motor Angular Velocity (RPM)')
    plt.draw()

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(time, thrust[:,0], time, thrust[:,1], time, thrust[:,2], time, thrust[:,3])
    plt.grid(True)
    plt.legend(['thr1','thr2','thr3','thr4'], loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Rotor Thrust (N)')
    plt.draw()

    plt.subplot(2,1,2)
    plt.plot(time, torque[:,0], time, torque[:,1], time, torque[:,2], time, torque[:,3])
    plt.grid(True)
    plt.legend(['tor1','tor2','tor3','tor4'], loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Rotor Torque (N*m)')
    plt.draw()



numFrames = 8

def gymSameAxisAnimation(t_all, waypoints, pos_all, quat_all, Ts, params, xyzType, yawType, ifsave):

    x = pos_all[:,0]
    y = pos_all[:,1]
    z = pos_all[:,2]

    # xDes = sDes_tr_all[:,0]
    # yDes = sDes_tr_all[:,1]
    # zDes = sDes_tr_all[:,2]

    x_wp = waypoints[:,0]
    y_wp = waypoints[:,1]
    z_wp = waypoints[:,2]

    # if (config.orient == "NED"):
    #     z = -z
    #     zDes = -zDes
    #     z_wp = -z_wp

    fig = plt.figure()
    ax = p3.Axes3D(fig)
    line1, = ax.plot([], [], [], lw=2, color='red')
    line2, = ax.plot([], [], [], lw=2, color='blue')
    line3, = ax.plot([], [], [], '--', lw=1, color='blue')

    # Setting the axes properties
    extraEachSide = 0.5
    maxRange = 0.5*np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() + extraEachSide
    mid_x = 0.5*(x.max()+x.min())
    mid_y = 0.5*(y.max()+y.min())
    mid_z = 0.5*(z.max()+z.min())
    
    ax.set_xlim3d([mid_x-maxRange, mid_x+maxRange])
    ax.set_xlabel('X')
    if (config.orient == "NED"):
        ax.set_ylim3d([mid_y+maxRange, mid_y-maxRange])
        z_wp = -z_wp
    elif (config.orient == "ENU"):
        ax.set_ylim3d([mid_y-maxRange, mid_y+maxRange])
    ax.set_ylabel('Y')
    ax.set_zlim3d([mid_z-maxRange, mid_z+maxRange])
    ax.set_zlabel('Altitude')

    titleTime = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    trajType = ''
    yawTrajType = ''

    if (xyzType == 0):
        trajType = 'Hover'
    else:
        ax.scatter(x_wp, y_wp, z_wp, color='green', alpha=1, marker = 'o', s = 25)
        if (xyzType == 1 or xyzType == 12):
            trajType = 'Simple Waypoints'
        else:
            ax.plot(xDes, yDes, zDes, ':', lw=1.3, color='green')
            if (xyzType == 2):
                trajType = 'Simple Waypoint Interpolation'
            elif (xyzType == 3):
                trajType = 'Minimum Velocity Trajectory'
            elif (xyzType == 4):
                trajType = 'Minimum Acceleration Trajectory'
            elif (xyzType == 5):
                trajType = 'Minimum Jerk Trajectory'
            elif (xyzType == 6):
                trajType = 'Minimum Snap Trajectory'
            elif (xyzType == 7):
                trajType = 'Minimum Acceleration Trajectory - Stop'
            elif (xyzType == 8):
                trajType = 'Minimum Jerk Trajectory - Stop'
            elif (xyzType == 9):
                trajType = 'Minimum Snap Trajectory - Stop'
            elif (xyzType == 10):
                trajType = 'Minimum Jerk Trajectory - Fast Stop'
            elif (xyzType == 1):
                trajType = 'Minimum Snap Trajectory - Fast Stop'

    if (yawType == 0):
        yawTrajType = 'None'
    elif (yawType == 1):
        yawTrajType = 'Waypoints'
    elif (yawType == 2):
        yawTrajType = 'Interpolation'
    elif (yawType == 3):
        yawTrajType = 'Follow'
    elif (yawType == 4):
        yawTrajType = 'Zero'



    titleType1 = ax.text2D(0.95, 0.95, trajType, transform=ax.transAxes, horizontalalignment='right')
    titleType2 = ax.text2D(0.95, 0.91, 'Yaw: '+ yawTrajType, transform=ax.transAxes, horizontalalignment='right')   
    
    def updateLines(i):

        time = t_all[i*numFrames]
        pos = pos_all[i*numFrames]
        x = pos[0]
        y = pos[1]
        z = pos[2]

        x_from0 = pos_all[0:i*numFrames,0]
        y_from0 = pos_all[0:i*numFrames,1]
        z_from0 = pos_all[0:i*numFrames,2]
    
        dxm = params["dxm"]
        dym = params["dym"]
        dzm = params["dzm"]
        
        quat = quat_all[i*numFrames]
    
        if (config.orient == "NED"):
            z = -z
            z_from0 = -z_from0
            quat = np.array([quat[0], -quat[1], -quat[2], quat[3]])
    
        R = quat2Dcm(quat)    
        motorPoints = np.array([[dxm, -dym, dzm], [0, 0, 0], [dxm, dym, dzm], [-dxm, dym, dzm], [0, 0, 0], [-dxm, -dym, dzm]])
        motorPoints = np.dot(R, np.transpose(motorPoints))
        motorPoints[0,:] += x 
        motorPoints[1,:] += y 
        motorPoints[2,:] += z 
        
        line1.set_data(motorPoints[0,0:3], motorPoints[1,0:3])
        line1.set_3d_properties(motorPoints[2,0:3])
        line2.set_data(motorPoints[0,3:6], motorPoints[1,3:6])
        line2.set_3d_properties(motorPoints[2,3:6])
        line3.set_data(x_from0, y_from0)
        line3.set_3d_properties(z_from0)
        titleTime.set_text(u"Time = {:.2f} s".format(time))
        
        return line1, line2


    def ini_plot():

        line1.set_data(np.empty([1]), np.empty([1]))
        line1.set_3d_properties(np.empty([1]))
        line2.set_data(np.empty([1]), np.empty([1]))
        line2.set_3d_properties(np.empty([1]))
        line3.set_data(np.empty([1]), np.empty([1]))
        line3.set_3d_properties(np.empty([1]))

        return line1, line2, line3

    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, updateLines, init_func=ini_plot, frames=len(t_all[0:-2:numFrames]), interval=(Ts*1000*numFrames), blit=False)
    
    if (ifsave):
        line_ani.save('animation_{0:.0f}_{1:.0f}.gif'.format(xyzType,yawType), dpi=80, writer='imagemagick', fps=25)
        
    plt.show()
    return line_ani
