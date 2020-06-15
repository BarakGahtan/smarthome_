import numpy as np
import xlrd
import datetime
import matplotlib.pyplot as plt
import math


# def get_dates_from_file_for_AAC(given_file_path):
#     data = given_file_path
#     tree = etree.parse(data)
#     root = tree.getroot()
#     list_of_dates = []
#     for current_line in root:
#         line = str(current_line.attrib)
#         splited = line.split(" ")
#         splited_pure_date = splited[1]
#         splited_pure_date = splited_pure_date[1:]
#         list_of_dates.append(splited_pure_date)
#     list_of_dates = list(dict.fromkeys(list_of_dates))
#     return list_of_dates
#

# read data from alaram aux channel,climate_control_valveActutaor, group 0 alarm
# number of colums 7
# ((year month day hours mins seconds) value)
# if null then -1
# valve Actuator group #(year month day hours mins seconds percent)
def read_data_seven_coloums(given_path):
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
        if value == "null":
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


def renomve_null_subtruct_min_sort(matrix):
    indices_alarm_aux_merge = []
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    for x in range(0, rows):
        for y in range(0, cols):
            if matrix[x][y] == -1:
                indices_alarm_aux_merge.append(x)
    updated_matrix = np.delete(matrix, indices_alarm_aux_merge, axis=0)
    minimum_ = np.amin(updated_matrix, axis=0)
    updated_matrix[:, 0] -= float(minimum_[0])
    sorted_updated_matrix = updated_matrix[np.argsort(updated_matrix[:, 0])]
    return sorted_updated_matrix


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
#read the file
alarm_aux_channel_1 = read_data_seven_coloums("data/Copy of Devices.Alarm.AuxChannel.1.xls")
alarm_aux_channel_2 = read_data_seven_coloums("data/Copy of Devices.Alarm.AuxChannel.2.xls")
alarm_aux_channel_3 = read_data_seven_coloums("data/Copy of Devices.Alarm.AuxChannel.3.xls")
alarm_aux_channel_4 = read_data_seven_coloums("data/Copy of Devices.Alarm.AuxChannel.4.xls")
alarm_aux_channel_5 = read_data_seven_coloums("data/Copy of Devices.Alarm.AuxChannel.5.xls")
alarm_aux_channel_6 = read_data_seven_coloums("data/Copy of Devices.Alarm.AuxChannel.6.xls")
alarm_aux_channel_7 = read_data_seven_coloums("data/Copy of Devices.Alarm.AuxChannel.7.xls")
alarm_aux_channel_8 = read_data_seven_coloums("data/Copy of Devices.Alarm.AuxChannel.8.xls")
alarm_aux_channel_9 = read_data_seven_coloums("data/Copy of Devices.Alarm.AuxChannel.9.xls")
# remove null, remove minimum and sort
alarm_aux_channel_1_updated = renomve_null_subtruct_min_sort(alarm_aux_channel_1)
alarm_aux_channel_2_updated = renomve_null_subtruct_min_sort(alarm_aux_channel_2)
alarm_aux_channel_3_updated = renomve_null_subtruct_min_sort(alarm_aux_channel_3)
alarm_aux_channel_4_updated = renomve_null_subtruct_min_sort(alarm_aux_channel_4)
alarm_aux_channel_5_updated = renomve_null_subtruct_min_sort(alarm_aux_channel_5)
alarm_aux_channel_6_updated = renomve_null_subtruct_min_sort(alarm_aux_channel_6)
alarm_aux_channel_7_updated = renomve_null_subtruct_min_sort(alarm_aux_channel_7)
alarm_aux_channel_8_updated = renomve_null_subtruct_min_sort(alarm_aux_channel_8)
alarm_aux_channel_9_updated = renomve_null_subtruct_min_sort(alarm_aux_channel_9)
week = 86400
# day = 86400, 604800
# list_count_max = math.ceil(np.amax(alarm_aux_channel_1_updated, axis=0)[0] / week)
list_count_max = max(math.ceil(np.amax(alarm_aux_channel_1_updated, axis=0)[0] / week),math.ceil(np.amax(alarm_aux_channel_2_updated, axis=0)[0] / week),math.ceil(np.amax(alarm_aux_channel_3_updated, axis=0)[0] / week),
                     math.ceil(np.amax(alarm_aux_channel_4_updated, axis=0)[0] / week),math.ceil(np.amax(alarm_aux_channel_5_updated, axis=0)[0] / week),
                     math.ceil(np.amax(alarm_aux_channel_6_updated, axis=0)[0] / week),math.ceil(np.amax(alarm_aux_channel_7_updated, axis=0)[0] / week)
                     ,math.ceil(np.amax(alarm_aux_channel_8_updated, axis=0)[0] / week),math.ceil(np.amax(alarm_aux_channel_9_updated, axis=0)[0] / week))
#intiallizing empty lists
array_list_aux_1 = [ [] for i in range(list_count_max) ]
array_list_aux_1_numpy = [ [] for i in range(list_count_max) ]
array_list_aux_2 = [ [] for i in range(list_count_max) ]
array_list_aux_2_numpy = [ [] for i in range(list_count_max) ]
array_list_aux_3 = [ [] for i in range(list_count_max) ]
array_list_aux_3_numpy = [ [] for i in range(list_count_max) ]
array_list_aux_4 = [ [] for i in range(list_count_max) ]
array_list_aux_4_numpy = [ [] for i in range(list_count_max) ]
array_list_aux_5 = [ [] for i in range(list_count_max) ]
array_list_aux_5_numpy = [ [] for i in range(list_count_max) ]
array_list_aux_6 = [ [] for i in range(list_count_max) ]
array_list_aux_6_numpy = [ [] for i in range(list_count_max) ]
array_list_aux_7 = [ [] for i in range(list_count_max) ]
array_list_aux_7_numpy = [ [] for i in range(list_count_max) ]
array_list_aux_8 = [ [] for i in range(list_count_max) ]
array_list_aux_8_numpy = [ [] for i in range(list_count_max) ]
array_list_aux_9 = [ [] for i in range(list_count_max) ]
array_list_aux_9_numpy = [ [] for i in range(list_count_max) ]

for i in range(list_count_max):
    array_list_aux_1[i].append([x for x in alarm_aux_channel_1_updated if (i + 1) * week > x[0] >= i * week])
    array_list_aux_1_numpy[i] = (np.concatenate(np.asarray(array_list_aux_1[i]), axis=0))

    array_list_aux_2[i].append([x for x in alarm_aux_channel_2_updated if (i + 1) * week > x[0] >= i * week])
    array_list_aux_2_numpy[i] = (np.concatenate(np.asarray(array_list_aux_2[i]), axis=0))

    array_list_aux_3[i].append([x for x in alarm_aux_channel_3_updated if (i + 1) * week > x[0] >= i * week])
    array_list_aux_3_numpy[i] = (np.concatenate(np.asarray(array_list_aux_3[i]), axis=0))

    array_list_aux_4[i].append([x for x in alarm_aux_channel_2_updated if (i + 1) * week > x[0] >= i * week])
    array_list_aux_4_numpy[i] = (np.concatenate(np.asarray(array_list_aux_4[i]), axis=0))

    array_list_aux_5[i].append([x for x in alarm_aux_channel_5_updated if (i + 1) * week > x[0] >= i * week])
    array_list_aux_5_numpy[i] = (np.concatenate(np.asarray(array_list_aux_5[i]), axis=0))

    array_list_aux_6[i].append([x for x in alarm_aux_channel_6_updated if (i + 1) * week > x[0] >= i * week])
    array_list_aux_6_numpy[i] = (np.concatenate(np.asarray(array_list_aux_6[i]), axis=0))

    array_list_aux_7[i].append([x for x in alarm_aux_channel_7_updated if (i + 1) * week > x[0] >= i * week])
    array_list_aux_7_numpy[i] = (np.concatenate(np.asarray(array_list_aux_7[i]), axis=0))

    array_list_aux_8[i].append([x for x in alarm_aux_channel_8_updated if (i + 1) * week > x[0] >= i * week])
    array_list_aux_8_numpy[i] = (np.concatenate(np.asarray(array_list_aux_8[i]), axis=0))

    array_list_aux_9[i].append([x for x in alarm_aux_channel_9_updated if (i + 1) * week > x[0] >= i * week])
    array_list_aux_9_numpy[i] = (np.concatenate(np.asarray(array_list_aux_9[i]), axis=0))


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

def calculate_avg_of_list(list):
    avg = 0
    count_without_zero = 0
    for i in range(len(list)):
        if (len(list[i]) != 0):
            avg += len(list[i])
            count_without_zero += 1

    avg /= count_without_zero
    return avg

def make_1D_array_per_one_aux_sensor(list): #recieves a list of list.
    avg = calculate_avg_of_list(list)
    if avg < len(list):
        avg = len(list)
    D1_arrays_aux_sensor = []
    for i in range(len(list)):
        D1_arrays_aux_sensor.append(prefix_sum(list[i], int(np.ceil(314))))
    return D1_arrays_aux_sensor

def return_data():
    D1_arrays_aux_sensor_1 = make_1D_array_per_one_aux_sensor(array_list_aux_1_numpy)
    D1_arrays_aux_sensor_2 = make_1D_array_per_one_aux_sensor(array_list_aux_2_numpy)
    D1_arrays_aux_sensor_3 = make_1D_array_per_one_aux_sensor(array_list_aux_3_numpy)
    D1_arrays_aux_sensor_4 = make_1D_array_per_one_aux_sensor(array_list_aux_4_numpy)
    D1_arrays_aux_sensor_5 = make_1D_array_per_one_aux_sensor(array_list_aux_5_numpy)
    D1_arrays_aux_sensor_6 = make_1D_array_per_one_aux_sensor(array_list_aux_6_numpy)
    D1_arrays_aux_sensor_7 = make_1D_array_per_one_aux_sensor(array_list_aux_7_numpy)
    D1_arrays_aux_sensor_8 = make_1D_array_per_one_aux_sensor(array_list_aux_8_numpy)
    D1_arrays_aux_sensor_9 = make_1D_array_per_one_aux_sensor(array_list_aux_9_numpy)
    result_list = []
    result_list.append(D1_arrays_aux_sensor_1)
    result_list.append(D1_arrays_aux_sensor_2)
    result_list.append(D1_arrays_aux_sensor_3)
    result_list.append(D1_arrays_aux_sensor_4)
    result_list.append(D1_arrays_aux_sensor_5)
    result_list.append(D1_arrays_aux_sensor_6)
    result_list.append(D1_arrays_aux_sensor_7)
    result_list.append(D1_arrays_aux_sensor_8)
    result_list.append(D1_arrays_aux_sensor_9)
    return result_list

return_data()





# for count in range(list_count_max):
#     flag = 0
#     if array_list_aux_1_numpy[count].size != 0:
#         plt.scatter(array_list_aux_1_numpy[count][:, 0], array_list_aux_1_numpy[count][:, 1], s=0.5, c="blue")
#         flag = 1
#     if array_list_aux_2_numpy[count].size != 0:
#         plt.scatter(array_list_aux_2_numpy[count][:, 0], array_list_aux_2_numpy[count][:, 1], s=0.5, c="black")
#         flag = 1
#     if array_list_aux_3_numpy[count].size != 0:
#         plt.scatter(array_list_aux_3_numpy[count][:, 0], array_list_aux_3_numpy[count][:, 1], s=0.5, c="yellow")
#         flag = 1
#     if array_list_aux_4_numpy[count].size != 0:
#         plt.scatter(array_list_aux_4_numpy[count][:, 0], array_list_aux_4_numpy[count][:, 1], s=0.5, c="cyan")
#         flag = 1
#     if array_list_aux_5_numpy[count].size != 0:
#         plt.scatter(array_list_aux_5_numpy[count][:, 0], array_list_aux_5_numpy[count][:, 1], s=0.5, c="red")
#         flag = 1
#     if array_list_aux_6_numpy[count].size != 0:
#         plt.scatter(array_list_aux_6_numpy[count][:, 0], array_list_aux_6_numpy[count][:, 1], s=0.5, c="green")
#         flag = 1
#     if array_list_aux_7_numpy[count].size != 0:
#         plt.scatter(array_list_aux_7_numpy[count][:, 0], array_list_aux_7_numpy[count][:, 1], s=0.5, c="magenta")
#         flag = 1
#     if array_list_aux_8_numpy[count].size != 0:
#         plt.scatter(array_list_aux_8_numpy[count][:, 0], array_list_aux_8_numpy[count][:, 1], s=0.5, c='#A2142F')
#         flag = 1
#     if array_list_aux_9_numpy[count].size != 0:
#         plt.scatter(array_list_aux_9_numpy[count][:, 0], array_list_aux_9_numpy[count][:, 1], s=0.5, c='#77AC30')
#         flag = 1
#
#     if flag is 1:
#         title_str = "Aux Stat , week number " + str(count)
#         plt.title(title_str)
#         plt.xlabel('Time')
#         plt.ylabel('Value')
#         title_str_png = title_str + ".png"
#         plt.savefig(title_str_png, dpi=300, bbox_inches='tight')
#         plt.show()
#         title_str_png = title_str + ".png"
    # plt.imsave(title_str_png,)



