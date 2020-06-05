import numpy as np
import xlrd
import datetime
import matplotlib.pyplot as plt
import math
import torch
########################################################################################################################
#aux methods
########################################################################################################################
# energy Managment actuator  #((year month day hours mins seconds) Real)
def read_data_from_energy_file(given_path):
    database = xlrd.open_workbook(given_path)
    worksheet = database.sheet_by_index(0)
    shape = (2)
    dates = []
    times = []
    database = []
    for current_row in range(worksheet.nrows):
        current_triple = []
        device = str((worksheet.cell(current_row, 0).value))
        date_and_time = str((worksheet.cell(current_row, 1).value))
        current_value = str((worksheet.cell(current_row, 2).value))
        current_date, current_time = date_and_time.split(" ")
        dates.append(current_date)
        times.append(current_time)
        current_triple.append(current_date)
        current_triple.append(current_time)
        current_triple.append(current_value)
        year, month, day = current_triple[0].split("-")
        hours, mins, seconds = current_triple[1].split(":")
        value = current_triple[2]
        if value == "NaN":
            value = -1
        current_row = np.zeros(shape)
        seconds_only, micro_seconds = seconds.split(".")
        date_time_object = datetime.datetime.combine(datetime.date(year=int(year), month=int(month), day=int(day)),
                                                     datetime.time(hour=int(hours), minute=int(mins),
                                                                   second=int(seconds_only),
                                                                   microsecond=int(micro_seconds)))
        time_stamp = datetime.datetime.timestamp(date_time_object)

        current_row[0] = time_stamp
        # current_row[0] = (float(year))
        # current_row[1] = (float(month))
        # current_row[2] = (float(day))
        # current_row[3] = (float(hours))
        # current_row[4] = (float(mins))
        # current_row[5] = (float(seconds))
        current_row[1] = (float(value))
        database.append(current_row)

    return np.asarray(database)

# read data from alaram aux channel,climate_control_valveActutaor, group 0 alarm
# number of colums 7
# ((year month day hours mins seconds) Real value)
# if NaN then -1
def read_data_from_thermal(given_path):
    database = xlrd.open_workbook(given_path)
    worksheet = database.sheet_by_index(0)
    shape = (2)
    dates = []
    times = []
    database = []
    for current_row in range(worksheet.nrows):
        current_triple = []
        device = str((worksheet.cell(current_row, 0).value))
        date_and_time = str((worksheet.cell(current_row, 1).value))
        current_value = str((worksheet.cell(current_row, 2).value))
        current_date, current_time = date_and_time.split(" ")
        dates.append(current_date)
        times.append(current_time)
        current_triple.append(current_date)
        current_triple.append(current_time)
        current_triple.append(current_value)
        year, month, day = current_triple[0].split("-")
        hours, mins, seconds = current_triple[1].split(":")
        value = current_triple[2]
        if value == "NaN":
            value = -1
        current_row = np.zeros(shape)
        seconds_only, micro_seconds = seconds.split(".")
        date_time_object = datetime.datetime.combine(datetime.date(year=int(year), month=int(month), day=int(day)),
                                                     datetime.time(hour=int(hours), minute=int(mins),
                                                                   second=int(seconds_only),
                                                                   microsecond=int(micro_seconds)))
        time_stamp = datetime.datetime.timestamp(date_time_object)
        current_row[0] = time_stamp
        current_row[1] = (float(value))
        database.append(current_row)

    return np.asarray(database)


def make_intersection_energy_manag_thermal_prob(energy, thermal):
    week = 86400
    array_list_thermal = []
    for i in range(math.ceil(np.amax(thermal, axis=0)[0] / week)):
        array_list_thermal.append([x for x in thermal if (i + 1) * week > x[0] >= i * week])

    mat_list_thermal = []
    tensor_list_thermal = []
    for i in range(len(array_list_thermal)):
        mat_list_thermal.append(np.asarray(array_list_thermal[i]))
        tensor_list_thermal.append(torch.from_numpy(np.asarray(array_list_thermal[i])))

    array_list_energy = []
    for i in range(math.ceil(np.amax(energy, axis=0)[0] / week)):
        array_list_energy.append([x for x in energy if (i + 1) * week > x[0] >= i * week])

    mat_list_energy = []
    tensor_list_energy = []
    for i in range(len(array_list_energy)):
        mat_list_energy.append(np.asarray(array_list_energy[i]))
        tensor_list_energy.append(torch.from_numpy(np.asarray(array_list_energy[i])))
    # print("number of mats in energy list " +str(len(mat_list_energy)))
    # print("number of mats in thermal list " + str(len(mat_list_thermal)))
    # count = 0
    # for current_mat in mat_list_energy:
    #     if len(current_mat) is 0:
    #         print( "week number " + str(count) +" is empty.")
    #         count+=1
    #         continue
    #     plt.scatter(current_mat[:, 0], current_mat[:, 1], s=3, c='blue')
    #     title_str = "Energy , graph number: " +str(count)
    #     plt.title(title_str)
    #     plt.xlabel('Time')
    #     plt.ylabel('Value')
    #     plt.show()
    #     count += 1

    count = 0
    # for current_mat in mat_list_thermal:
    #     if len(current_mat) is 0:
    #         print( "week number " + str(count) +" is empty.")
    #         count+=1
    #         continue
    #     plt.scatter(current_mat[:, 0], current_mat[:, 1], s=1, c='blue')
    #     title_str = "Thermal, day number " +str(count)
    #     plt.title(title_str)
    #     plt.xlabel('Time')
    #     plt.ylabel('Value')
    #     title_str_png = title_str + ".png"
    #     plt.savefig(title_str_png, dpi=300, bbox_inches='tight')
    #     plt.show()
    #     title_str_png = title_str +".png"
    #     #plt.imsave(title_str_png,)
    #     count += 1
    #for saving



    # mat_list_energy = list of numpy arrays, each array is a week of time.
    # count = 0
    # for count in range(len(mat_list_thermal)):
    #     if len(mat_list_energy[count]) is 0 or len(mat_list_thermal[count]) is 0:
    #         print( "week number " + str(count) +" is empty.")
    #         count+=1
    #         continue
    #     current_thermal, current_energy = mat_list_thermal[int(count)], mat_list_energy[int(count)]
    #     plt.scatter(current_thermal[:, 0], current_thermal[:, 1], s=3,c='blue')
    #     plt.scatter(current_energy[:, 0], current_energy[:, 1], s=3, c='black')
    #     title_str = "Thermal intersect energy , graph number: " +str(count)
    #     plt.title(title_str)
    #     plt.xlabel('Time')
    #     plt.ylabel('Value')
    #     plt.show()
    #     count += 1

    return mat_list_thermal, mat_list_energy, tensor_list_thermal, tensor_list_energy



########################################################################################################################
#energy preparation
########################################################################################################################
eng_mana_actuator_1 = read_data_from_energy_file(
    "data/Copy of Devices.EnergyManagement.EnergyManagementActuator.1.xls")
eng_mana_actuator_2 = read_data_from_energy_file(
    "data/Copy of Devices.EnergyManagement.EnergyManagementActuator.2.xls")
eng_mana_actuator_3 = read_data_from_energy_file(
    "data/Copy of Devices.EnergyManagement.EnergyManagementActuator.3.xls")
eng_mana_actuator_4 = read_data_from_energy_file(
    "data/Copy of Devices.EnergyManagement.EnergyManagementActuator.4.xls")
eng_mana_actuator_5 = read_data_from_energy_file(
    "data/Copy of Devices.EnergyManagement.EnergyManagementActuator.5.xls")
eng_mana_actuator_6 = read_data_from_energy_file(
    "data/Copy of Devices.EnergyManagement.EnergyManagementActuator.6.xls")
eng_mana_actuator_7 = read_data_from_energy_file(
    "data/Copy of Devices.EnergyManagement.EnergyManagementActuator.7.xls")
eng_mana_actuator_8 = read_data_from_energy_file(
    "data/Copy of Devices.EnergyManagement.EnergyManagementActuator.8.xls")
eng_mana_actuator_9 = read_data_from_energy_file(
    "data/Copy of Devices.EnergyManagement.EnergyManagementActuator.9.xls")
eng_mana_actuator_merge = np.concatenate((eng_mana_actuator_1, eng_mana_actuator_2, eng_mana_actuator_3,
                                          eng_mana_actuator_4, eng_mana_actuator_5, eng_mana_actuator_6,
                                           eng_mana_actuator_7, eng_mana_actuator_8, eng_mana_actuator_9), axis=0)
indices_eng_mana_actuator_merge = []
rows = eng_mana_actuator_merge.shape[0]
cols = eng_mana_actuator_merge.shape[1]
for x in range(0, rows):
    for y in range(0, cols):
        if eng_mana_actuator_merge[x][y] == -1:
            indices_eng_mana_actuator_merge.append(x)
eng_mana_actuator_merge = np.delete(eng_mana_actuator_merge, indices_eng_mana_actuator_merge, axis=0)

# find the minimum value of time
minimum_time = np.amin(eng_mana_actuator_merge, axis=0)
eng_mana_actuator_merge[:, 0] -= float(minimum_time[0])
print("minimum time value for energy is " + str(minimum_time[0]))
eng_mana_actuator_merge_sorted = eng_mana_actuator_merge[np.argsort(eng_mana_actuator_merge[:, 0])]
########################################################################################################################
#thermal preparation
########################################################################################################################
thermal_probe_1 = read_data_from_thermal("data/Copy of Devices.ClimateControl.ThermalProbe.1.xls")
thermal_probe_2 = read_data_from_thermal("data/Copy of Devices.ClimateControl.ThermalProbe.2.xls")
thermal_probe_3 = read_data_from_thermal("data/Copy of Devices.ClimateControl.ThermalProbe.3.xls")
Thermal_probe_merge = np.concatenate((thermal_probe_1, thermal_probe_2, thermal_probe_3), axis=0)
indices_Thermal_probe_merge = []
rows = Thermal_probe_merge.shape[0]
cols = Thermal_probe_merge.shape[1]
for x in range(0, rows):
    for y in range(0, cols):
        if Thermal_probe_merge[x][y] == -1:
            indices_Thermal_probe_merge.append(x)
Thermal_probe_merge = np.delete(Thermal_probe_merge, indices_Thermal_probe_merge, axis=0)
# find the minimum value of time
minimum_time_Thermal_probe_merge = np.amin(Thermal_probe_merge, axis=0)
#print("minimum time value for thermal is " + str(minimum_time_Thermal_probe_merge[0]))
#print("difference between minimum time between thermal and energy is " + str(minimum_time_Thermal_probe_merge[0] -minimum_time[0] ) )
Thermal_probe_merge[:, 0] -= float(minimum_time_Thermal_probe_merge[0])
thermal_probe_merge_sorted = Thermal_probe_merge[np.argsort(Thermal_probe_merge[:, 0])]
#Thermal_probe_merge_splited = np.array_split(Thermal_probe_merge, 1700, axis=0)  # 38 entries (rows)
########################################################################################################################
#intersection preparation
########################################################################################################################
#list_mat_thermal, list_mat_energy = make_intersection_energy_manag_thermal_prob(eng_mana_actuator_merge_sorted,
                                                                                #thermal_probe_merge_sorted)
########################################################################################################################
#transformation to tensor flow
########################################################################################################################
torch_thermal_merge_tensor = torch.from_numpy(thermal_probe_merge_sorted)
torch_eng_mana_actuator_merge_tensor = torch.from_numpy(eng_mana_actuator_merge_sorted)
# print(torch_eng_mana_actuator_merge_tensor)

def pad_tensor_list(list):
    max_len = max([len(i) for i in list])
    return [torch.cat((l, torch.zeros(max_len - len(l), 2, dtype=torch.float64)), dim=0 ) for l in list], max_len
###
    week = 86400
    array_list_thermal = []
    for i in range(math.ceil(np.amax(thermal, axis=0)[0] / week)):
        array_list_thermal.append([x for x in thermal if (i + 1) * week > x[0] >= i * week])


def prefix_sum(data, legnth):
    time_unit = 86400/legnth
    res = np.zeros(int(np.ceil(legnth)),dtype=float)
    if len(data) == 0:
        return res
    minimum_time_local = min(data[:,0])
    for i in range(len(data)): #reducing the minimum time
        data[i,0] -= float(minimum_time_local)

    for row in data:
        if 0 <= row[0] <= time_unit:
            res[0] +=  row[1]

    i=1
    for current_time_slot in range(int(time_unit),86400,int(time_unit)):
        flag = 0
        if i == legnth: break
        res[i] = res[i - 1]
        for row in data:
            if  float(current_time_slot)  <= row[0] <= float(current_time_slot + time_unit):
                #flag =1
                res[i] += row[1]
        #if flag == 0:
         #   res[i] = res[i-1]
        i += 1
    return res



def return_data():
    list_mat_thermal, list_mat_energy, tensor_list_thermal, tensor_list_energy = make_intersection_energy_manag_thermal_prob(eng_mana_actuator_merge_sorted,
    thermal_probe_merge_sorted)
    avg = 0
    for i in range(len(list_mat_energy)):
        avg += len(list_mat_energy)
    avg /= len(list_mat_energy)
    D1_arrays_energy = []
    for i in range(len(list_mat_energy)):
        D1_arrays_energy.append(prefix_sum(list_mat_energy[i], int(np.ceil(avg))))
    D1_arrays_thermal = []
    for i in range(len(list_mat_energy)):
        D1_arrays_thermal.append(prefix_sum(list_mat_thermal[i], int(np.ceil(avg))))

    return D1_arrays_energy, D1_arrays_thermal
    #max(list_mat_energy[3][:,0])- min(list_mat_energy[3][:,0])
    # tensor_list_thermal_padded, thermal_max= pad_tensor_list(tensor_list_thermal)
    # tensor_list_energy_padded, energy_max = pad_tensor_list(tensor_list_energy)
    # return tensor_list_thermal, tensor_list_thermal_padded,thermal_max, tensor_list_energy, tensor_list_energy_padded, energy_max


########################################################################################################################
# split the matrix into submatrix.
# eng_mana_actuator_merge_splitted = np.array_split(eng_mana_actuator_merge, 4000, axis=0)  # 38 entries (rows)
# count = 0
# for current_mat in eng_mana_actuator_merge_splitted:
#     if count is 50: break
#     plt.scatter(current_mat[:, 0], current_mat[:, 1], s=3)
#     plt.title("Thermal Probe")
#     empty_str = 'Time, the count is: ' + str(count)
#     plt.xlabel(empty_str)
#     plt.ylabel('Value')
#     plt.show()
#     count += 1

#
# plt.scatter(Thermal_probe_merge[:, 0], Thermal_probe_merge[:, 1], s=0.1)
# plt.title("Thermal Probe Actuator Merge")
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.show()



#find maximum
# maximum_time_thermal_prob = np.amax(Thermal_probe_merge, axis=0)[0]
# print("max time for thermal problem after reduction is: " + str(maximum_time_thermal_prob))