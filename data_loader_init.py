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


# number of colums 10
# read data from alaram zone, alarm zone central
# ((year month day hours mins seconds) active panic intrusion tamper)
# ((year month day hours mins seconds) active armed Battery Network) - alarm zone central
# if null then -1
def read_data_alarm_zone_(given_path):
    database = xlrd.open_workbook(given_path)
    worksheet = database.sheet_by_index(0)
    shape = (5)
    dates = []
    times = []
    database = []
    for current_row in range(worksheet.nrows):
        current_sixth = []
        device = str((worksheet.cell(current_row, 0).value))
        date_and_time = str((worksheet.cell(current_row, 1).value))
        if worksheet.cell(current_row, 2).value == 'null':
            active = -1
        else:
            active = worksheet.cell(current_row, 2).value
        if worksheet.cell(current_row, 3).value == 'null':
            anti_panic = -1
        else:
            anti_panic = worksheet.cell(current_row, 3).value
        if worksheet.cell(current_row, 4).value == 'null':
            intrusion = -1
        else:
            intrusion = worksheet.cell(current_row, 4).value
        if worksheet.cell(current_row, 5).value == 'null':
            tamper = -1
        else:
            tamper = worksheet.cell(current_row, 5).value

        current_date, current_time = date_and_time.split(" ")
        dates.append(current_date)
        times.append(current_time)
        current_sixth.append(current_date)
        current_sixth.append(current_time)
        current_sixth.append(active)
        current_sixth.append(anti_panic)
        current_sixth.append(intrusion)
        current_sixth.append(tamper)
        year, month, day = current_sixth[0].split("-")
        hours, mins, seconds = current_sixth[1].split(":")
        current_row = np.zeros(shape)
        seconds_only, micro_seconds = seconds.split(".")
        date_time_object = datetime.datetime.combine(datetime.date(year=int(year), month=int(month), day=int(day)),
                                                     datetime.time(hour=int(hours), minute=int(mins),
                                                                   second=int(seconds_only),
                                                                   microsecond=int(micro_seconds)))
        time_stamp = datetime.datetime.timestamp(date_time_object)
        # current_row[0] = (float(year))
        # current_row[1] = (float(month))
        # current_row[2] = (float(day))
        # current_row[3] = (float(hours))
        # current_row[4] = (float(mins))
        # current_row[5] = (float(seconds))
        current_row[0] = time_stamp
        current_row[1] = (float(active))
        current_row[2] = (float(anti_panic))
        current_row[3] = (float(intrusion))
        current_row[4] = (float(tamper))
        database.append(current_row)

    return np.asarray(database)



# number of colums 10
# read data from alaram zone, alarm zone central
# ((year month day hours mins seconds) active panic intrusion tamper)
# ((year month day hours mins seconds) active armed Battery Network) - alarm zone central
# if null then -1
def read_data_alarm_zone_(given_path):
    database = xlrd.open_workbook(given_path)
    worksheet = database.sheet_by_index(0)
    shape = (5)
    dates = []
    times = []
    database = []
    for current_row in range(worksheet.nrows):
        current_sixth = []
        device = str((worksheet.cell(current_row, 0).value))
        date_and_time = str((worksheet.cell(current_row, 1).value))
        if worksheet.cell(current_row, 2).value == 'null':
            active = -1
        else:
            active = worksheet.cell(current_row, 2).value
        if worksheet.cell(current_row, 3).value == 'null':
            anti_panic = -1
        else:
            anti_panic = worksheet.cell(current_row, 3).value
        if worksheet.cell(current_row, 4).value == 'null':
            intrusion = -1
        else:
            intrusion = worksheet.cell(current_row, 4).value
        if worksheet.cell(current_row, 5).value == 'null':
            tamper = -1
        else:
            tamper = worksheet.cell(current_row, 5).value

        current_date, current_time = date_and_time.split(" ")
        dates.append(current_date)
        times.append(current_time)
        current_sixth.append(current_date)
        current_sixth.append(current_time)
        current_sixth.append(active)
        current_sixth.append(anti_panic)
        current_sixth.append(intrusion)
        current_sixth.append(tamper)
        year, month, day = current_sixth[0].split("-")
        hours, mins, seconds = current_sixth[1].split(":")
        current_row = np.zeros(shape)
        seconds_only, micro_seconds = seconds.split(".")
        date_time_object = datetime.datetime.combine(datetime.date(year=int(year), month=int(month), day=int(day)),
                                                     datetime.time(hour=int(hours), minute=int(mins),
                                                                   second=int(seconds_only),
                                                                   microsecond=int(micro_seconds)))
        time_stamp = datetime.datetime.timestamp(date_time_object)

        current_row[0] = time_stamp
        # current_row[1] = (float(month))
        # current_row[2] = (float(day))
        # current_row[3] = (float(hours))
        # current_row[4] = (float(mins))
        # current_row[5] = (float(seconds))
        current_row[1] = (float(active))
        current_row[2] = (float(anti_panic))
        current_row[3] = (float(intrusion))
        current_row[4] = (float(tamper))
        database.append(current_row)

    return np.asarray(database)


# energy Managment actuator  #((year month day hours mins seconds) Real)
def read_data_seven_coloums_management_actuator(given_path):
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
alarm_zone_0 = read_data_alarm_zone_("data/Copy of Devices.Alarm.AlarmZone.0.xls")
alarm_zone_1 = read_data_alarm_zone_("data/Copy of Devices.Alarm.AlarmZone.1.xls")
alarm_zone_2 = read_data_alarm_zone_("data/Copy of Devices.Alarm.AlarmZone.2.xls")
alarm_zone_3 = read_data_alarm_zone_("data/Copy of Devices.Alarm.AlarmZone.3.xls")
alarm_zone_12 = read_data_alarm_zone_("data/Copy of Devices.Alarm.AlarmZone.12.xls")
alarm_zone_merge = np.concatenate((alarm_zone_0, alarm_zone_1, alarm_zone_2,
                                   alarm_zone_3, alarm_zone_12), axis=0)

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
ClimateControl_ValveActuator_0_1 = read_data_seven_coloums("data/Copy of Devices.ClimateControl.ValveActuator.0.1.xls")
ClimateControl_ValveActuator_1_1 = read_data_seven_coloums("data/Copy of Devices.ClimateControl.ValveActuator.1.1.xls")
ClimateControl_ValveActuator_1_2 = read_data_seven_coloums("data/Copy of Devices.ClimateControl.ValveActuator.1.2.xls")
ClimateControl_ValveActuator_1_3 = read_data_seven_coloums("data/Copy of Devices.ClimateControl.ValveActuator.1.3.xls")
ClimateControl_ValveActuator_2_4 = read_data_seven_coloums("data/Copy of Devices.ClimateControl.ValveActuator.2.4.xls")
ClimateControl_ValveActuator_3_1 = read_data_seven_coloums("data/Copy of Devices.ClimateControl.ValveActuator.3.1.xls")
ClimateControl_ValveActuator_3_2 = read_data_seven_coloums("data/Copy of Devices.ClimateControl.ValveActuator.3.2.xls")
ClimateControl_ValveActuator_3_3 = read_data_seven_coloums("data/Copy of Devices.ClimateControl.ValveActuator.3.3.xls")
ClimateControl_ValveActuator_3_4 = read_data_seven_coloums("data/Copy of Devices.ClimateControl.ValveActuator.3.4.xls")
valve_actuator_merge = np.concatenate((ClimateControl_ValveActuator_0_1, ClimateControl_ValveActuator_1_1,
                                       ClimateControl_ValveActuator_1_2, ClimateControl_ValveActuator_1_3,
                                       ClimateControl_ValveActuator_2_4, ClimateControl_ValveActuator_3_1,
                                       ClimateControl_ValveActuator_3_2, ClimateControl_ValveActuator_3_3,
                                       ClimateControl_ValveActuator_3_4), axis=0)
indices_valve_actuator_merge = []
rows = valve_actuator_merge.shape[0]
cols = valve_actuator_merge.shape[1]
for x in range(0, rows):
    for y in range(0, cols):
        if valve_actuator_merge[x][y] == -1:
            indices_valve_actuator_merge.append(x)
valve_actuator_merge = np.delete(valve_actuator_merge, indices_valve_actuator_merge, axis=0)
# find the minimum value of time
minimum_time_valve_actuator_merge = np.amin(valve_actuator_merge, axis=0)
valve_actuator_merge[:, 0] -= float(minimum_time_valve_actuator_merge[0])

plt.scatter(valve_actuator_merge[:, 0], valve_actuator_merge[:, 1], s=0.001)
plt.title("Climate Control Valve Actuator Merge")
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
actuator_group_1 = read_data_seven_coloums("data/Copy of Devices.ClimateControl.ValveActuatorGroup.1.xls")
actuator_group_3 = read_data_seven_coloums("data/Copy of Devices.ClimateControl.ValveActuatorGroup.3.xls")
actuator_group_merge_probe_merge = np.concatenate((actuator_group_1, actuator_group_3), axis=0)
indices_actuator_group_merge_probe_merge = []
rows = actuator_group_merge_probe_merge.shape[0]
cols = actuator_group_merge_probe_merge.shape[1]
for x in range(0, rows):
    for y in range(0, cols):
        if actuator_group_merge_probe_merge[x][y] == -1:
            indices_actuator_group_merge_probe_merge.append(x)
actuator_group_merge_probe_merge = np.delete(actuator_group_merge_probe_merge, indices_actuator_group_merge_probe_merge,
                                             axis=0)
# find the minimum value of time
minimum_time_ac = np.amin(actuator_group_merge_probe_merge, axis=0)
actuator_group_merge_probe_merge[:, 0] -= float(minimum_time_ac[0])
size_is = np.size(actuator_group_merge_probe_merge, 0)
plt.scatter(actuator_group_merge_probe_merge[:, 0], actuator_group_merge_probe_merge[:, 1], s=1)
plt.title("Actuator Gropu Merge")
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
alarm_zone_central = read_data_alarm_zone_("data/Copy of Devices.Alarm.AlarmCentral.xls")
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
alarm_zone_group_0 = read_data_seven_coloums("data/Copy of Devices.Alarm.AlarmZoneGroup.0.xls")
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
alarm_sensor_1_4 = read_data_seven_coloums("data/Copy of Devices.Alarm.Sensor.1.4.xls")
