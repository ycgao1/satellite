# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 05:17:50 2024

@author: ycgao
"""

import numpy as np
from load_data.get_satellites import satellite, get_satellites
from load_data.get_ground_stations import ground_station, get_ground_station
from load_data.load_tle import parse_tle
from skyfield.api import load
from model.distribution_efficiency_sat2sat import communication_loss_sat2sat
from model.distribution_efficiency_stat2sat import communication_loss_stat2sat

tle_path = './dataset/satellite_tle.txt'
satellite_tle_data = parse_tle(tle_path)

# 提取有效时间窗口
def extract_valid_windows(data, threshold=0):
    # 找到大于阈值的索引
    indices = np.where(data > threshold)[0]
    # 通过对连续索引分组提取时间窗口
    windows = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
    # 只返回长度大于0的窗口
    valid_windows = [window for window in windows if len(window) > 0]
    return valid_windows

def calculate_key_generation_rate(eta, R_source, QBER, QBER_threshold=0.11):
    """
    Calculate the key generation rate for quantum communication using E91 protocol,
    considering a QBER threshold.

    Parameters:
    eta (float): Transmission efficiency (0 <= eta <= 1)
    R_source (float): Photon pair emission rate per second from the quantum source
    QBER (float): Quantum Bit Error Rate (0 <= QBER <= 1)
    QBER_threshold (float): The maximum acceptable QBER for key generation (default is 0.11)

    Returns:
    float: Key generation rate in bits per second. Returns 0 if QBER is above the threshold.
    """
    # Check if QBER is within the acceptable range
    if QBER > QBER_threshold:
        return 0  # No key is generated if the QBER is too high

    # Calculate the raw key rate considering the transmission efficiency and the emission rate
    S_raw = 0.5 * eta * R_source  # 50% probability of choosing each basis
    
    # Calculate the final key rate considering the QBER
    key_generation_rate = S_raw * (1 - 2 * QBER)
    
    return key_generation_rate


# 定义卫星列表
satellites = ['STARLINK-1094', 'STARLINK-1102', 'STARLINK-1144', 'STARLINK-1156']

# 创建一个字典来存储每颗卫星的效率历史数据
satellite_data = {}
valid_windows_dict = {}  # 创建一个字典来存储每颗卫星的有效时间窗口

# 循环每个卫星，读取存储的效率历史数据
for sat in satellites:
    file_path = f'E:\starlink_quantum\dataset/efficiency_history_{sat}.json.npy'  # 使用.npy文件格式
    efficiency_history = np.load(file_path)  # 使用numpy.load读取数据
    satellite_data[sat] = efficiency_history  # 存储到字典中
    valid_windows_dict[sat] = extract_valid_windows(satellite_data[sat])

import numpy as np
import random

def simulate_key_rates_with_adjusted_windows(valid_windows_dict, satellite_data, qber_no_eve, qber_with_eve, R_source, satellite_tle_data, primary_satellite='STARLINK-1094', total_attackers=1500):
    total_satellites = len(satellite_data)
    all_key_rates = np.zeros(len(satellite_data[primary_satellite]))
    switch_success_count = 0
    switch_failure_count = 0

    def attempt_reshedule(current_satellite, already_searched, start_window_id, previous_start_time, switch_success_count,  switch_failure_count):
        
        current_windows = valid_windows_dict[current_satellite]
        next_window_used = False
        for idx in range(start_window_id, len(current_windows)):
            if idx != start_window_id:
                already_searched = set()
                
            # Check if we're on the next window after a successful switch
            if idx == start_window_id + 1 :
                switch_success_count += 1
            
            targeted_satellites = random.sample(list(satellite_tle_data), total_attackers) #random attack
            
            window = current_windows[idx]
            window_start = window[0]

            if current_satellite in targeted_satellites:
                can_switch, new_satellite, new_window_id = can_switch_to_another_satellite(window_start , valid_windows_dict, current_satellite, already_searched)
                if can_switch:
                    print(f"Switching from {current_satellite} to {new_satellite} at second {window_start + 30}")
                    already_searched.add(new_satellite) 
                    #valid_windows_dict[new_satellite][0]
                    return attempt_reshedule(new_satellite, already_searched, new_window_id, max(valid_windows_dict[new_satellite][new_window_id][0], previous_start_time+15, window_start+15), switch_success_count,  switch_failure_count)
                else:
                    qber = qber_with_eve  # No available satellite to switch or all are monitored
                    switch_failure_count +=1
            else:
                qber = qber_no_eve  # Current satellite not targeted

             # Update key rates for the current window
            for second in range(max(window_start, previous_start_time), window[-1] + 1):
                eta = satellite_data[current_satellite][second]
                key_rate = calculate_key_generation_rate(eta, R_source, qber, QBER_threshold)
                all_key_rates[second] = key_rate

        return all_key_rates, switch_success_count,  switch_failure_count  # Return all_key_rates

    # Start the simulation with the initial satellite and an empty set for already searched satellites
    return attempt_reshedule(primary_satellite, set(), 0, 0, switch_success_count,  switch_failure_count)

def can_switch_to_another_satellite(start_second, valid_windows_dict, current_satellite, already_searched):
    switch_time_start = start_second + 10
    switch_time_end = start_second + 60
    for sat, windows in valid_windows_dict.items():
        if sat == current_satellite or sat in already_searched:
            continue
        for idx, window in enumerate(windows):
            window_start = window[0]
            window_end = window[-1]
            if window_start <= switch_time_end and window_end >= switch_time_start:
                return True, sat, idx  # 返回卫星名、索引
    return False, None, None  # 如果找不到合适的卫星，返回False和None


qber_no_eve = 0.045
qber_with_eve = 0.25
R_source = 10000000000
QBER_threshold = 0.11
# Assuming the other necessary functions and variables are defined as needed.
all_key_rates, switch_success_count,  switch_failure_count = simulate_key_rates_with_adjusted_windows(valid_windows_dict, satellite_data, qber_no_eve, qber_with_eve, R_source, satellite_tle_data, primary_satellite='STARLINK-1094', total_attackers=1500)
