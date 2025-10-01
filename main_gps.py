import numpy as np
import numpy.random as npr
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import math
import model
from referencevectorgauge import ReferenceVectorGauge
from noisydevice import NoisyDeviceDecorator
import gyro
import kalman_gps as kalman
import copy

def quatToEuler(q):
    euler = []
    x = math.atan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]*q[1] + q[2]*q[2]))
    euler.append(x)

    x = math.asin(2*(q[0]*q[2] - q[3]*q[1]))
    euler.append(x)

    x = math.atan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]*q[2] + q[3]*q[3]))
    euler.append(x)

    return euler

def quatListToEulerArrays(qs):
    euler = np.ndarray(shape=(3, len(qs)), dtype=float)

    for (i, q) in enumerate(qs):
        e = quatToEuler(q)
        euler[0, i] = e[0]
        euler[1, i] = e[1]
        euler[2, i] = e[2]

    return euler

def eulerError(estimate, truth):
    return np.minimum(np.minimum(np.abs(estimate - truth), np.abs(2*math.pi + estimate - truth)),
                                 np.abs(-2*math.pi + estimate - truth))

def eulerArraysToErrorArrays(estimate, truth):
    errors = []
    for i in range(3):
        errors.append(eulerError(estimate[i], truth[i]))
    return errors

def quatListToErrorArrays(estimate, truth):
    return eulerArraysToErrorArrays(quatListToEulerArrays(estimate), quatListToEulerArrays(truth))

def rmse_euler(estimate, truth):
    def rmse(vec1, vec2):
        return np.sqrt(np.mean((vec1 - vec2)**2))

    return[rmse(estimate[0], truth[0]), 
           rmse(estimate[1], truth[1]),
           rmse(estimate[2], truth[2])]

if __name__ == '__main__':
    NUM_ITER = 10000
    # NUM_ITER = 30
    accel_cov = 0.001
    accel_bias = np.array([0.6, 0.02, 0.05])
    mag_cov = 0.001
    mag_bias = np.array([0.03, 0.08, 0.04])
    gyro_cov = 0.1
    gyro_bias = np.array([0.25, 0.01, 0.05])
    gyro_bias_drift = 0.0001

    # Create Gyro w/ initial bias, noise dist, bias drift
    gyro = NoisyDeviceDecorator(gyro.Gyro(), gyro_bias, gyro_cov, gyro_bias_drift)
    # CCreate accel+mag w/ initial bias, noise dist, bias drift, and a standard direction
    accelerometer = NoisyDeviceDecorator(ReferenceVectorGauge(np.array([0, 0, -1])), 
            accel_bias, accel_cov, bias_drift_covariance = 0.0) 
    magnetometer = NoisyDeviceDecorator(ReferenceVectorGauge(np.array([1, 0, 0])), 
            mag_bias, mag_cov, bias_drift_covariance = 0.0) 
    real_measurement = np.array([0.0, 0.0, 0.0])
    time_delta = 0.005

    gps_data = np.tile(NUM_ITER * np.linspace(0, 1, num=NUM_ITER).reshape(NUM_ITER, 1), (1, 3))

    
    # Create true orientation and dead reckoning accumulated state
    true_orientation = model.Model(Quaternion(axis = [1, 0, 0], angle=0))
    dead_reckoning_estimate = model.Model(Quaternion(axis = [1, 0, 0], angle=0))
    true_rotations = []
    dead_reckoning_rotation_estimates = []
    filtered_rotation_estimates = []
    filtered_position_estimates = []
    filtered_velocity_estimates = []
    # initial_est, estimate_covariance, gyro_cov, gyro_bias_cov, accel_proc_cov,
    # accel_bias_cov, mag_proc_cov, mag_bias_cov, accel_obs_cov, mag_obs_cov)
    # Create K filter w/ starting orientation and estimates for covariances, etc.
    kalman = kalman.Kalman(true_orientation.orientation, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1)
    for i in range(NUM_ITER):

        if (i % 10 == 0):
          real_measurement = npr.normal(0.0, 1.0, 3)
            # Create a measurement of actual rotation of the vehicle

        gyro_measurement = gyro.measure(time_delta, real_measurement)
        # Turn gyro measurement into noisy measurement

        dead_reckoning_estimate.update(time_delta, gyro_measurement)
        dead_reckoning_rotation_estimates.append(dead_reckoning_estimate.orientation)
        # Raw integrate for dead reckoning

        true_orientation.update(time_delta, real_measurement)
        true_rotations.append(true_orientation.orientation)
        # Keep track of state
        
        measured_acc = accelerometer.measure(time_delta, true_orientation.orientation)
        measured_mag = magnetometer.measure(time_delta, true_orientation.orientation)
        # Measure acc,mag based on current orientation estimate

        # TODO dummy gps_measurement:
        gps_meas = gps_data[i, :]

        kalman.update(gyro_measurement, measured_acc, measured_mag, gps_meas, time_delta)
        # Give all measurements and dt to kf
        filtered_rotation_estimates.append(copy.deepcopy(kalman.estimate))
        filtered_position_estimates.append(copy.deepcopy(kalman.position))
        filtered_velocity_estimates.append(copy.deepcopy(kalman.velocity))

    #print "gyro bias: ", kalman.gyro_bias
    #print "accel bias: ", kalman.accelerometer_bias
    #print "mag bias: ", kalman.magnetometer_bias

    dead_reckoning_errors = quatListToErrorArrays(dead_reckoning_rotation_estimates, true_rotations)
    filtered_errors = quatListToErrorArrays(filtered_rotation_estimates, true_rotations)

    unfiltered_roll, = plt.plot(dead_reckoning_errors[0], label='dead reckoning roll')
    unfiltered_pitch, = plt.plot(dead_reckoning_errors[1], label='dead reckoning pitch')
    unfiltered_yaw, = plt.plot(dead_reckoning_errors[2], label='dead reckoning yaw')
    filtered_roll, = plt.plot(filtered_errors[0], label='mekf roll')
    filtered_pitch, = plt.plot(filtered_errors[1], label='mekf pitch')
    filtered_yaw, = plt.plot(filtered_errors[2], label='mekf yaw')
    plt.legend(handles=[unfiltered_roll, 
                        unfiltered_pitch,
                        unfiltered_yaw, 
                        filtered_roll, 
                        filtered_pitch, 
                        filtered_yaw])
    plt.xlabel("Discrete time")
    plt.ylabel("Error (in radians)")
    plt.show()

    # print(f"Filtered position estimates: {filtered_position_estimates}")
    x_pos = [x[0] for x in filtered_position_estimates]
    y_pos = [x[1] for x in filtered_position_estimates]
    plt.clf()
    filtered_pos = plt.scatter(x_pos, y_pos)
    plt.show()


    # print(f"Filtered position estimates: {filtered_position_estimates}")
    x_vel = [x[0] for x in filtered_velocity_estimates]
    y_vel = [x[1] for x in filtered_velocity_estimates]
    plt.clf()
    filtered_vel = plt.scatter(x_vel, y_vel)
    plt.show()
