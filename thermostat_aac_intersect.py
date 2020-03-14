import numpy as np
import xlrd
import datetime
import matplotlib.pyplot as plt
import math

# number of colums 10
# ((year month day hours mins seconds) fan active temperature vane) aacControl
# if Undefined = -2
# if null = -1
# if 0ff = 0
# if auto = 1
# if Cooling = 2
# if Heating = 3
# if Dehumidification = 4
def read_data_AAC_control_(given_path):
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
            fan = -1
        elif str(worksheet.cell(current_row, 2).value) == 'Undefined':
            fan = -2
        elif worksheet.cell(current_row, 2).value == 'Auto':
            fan = 1
        elif worksheet.cell(current_row, 2).value == 'Cooling':
            fan = 2
        elif worksheet.cell(current_row, 2).value == 'Heating':
            fan = 3
        elif worksheet.cell(current_row, 2).value == 'Dehumidification':
            fan = 4
        elif worksheet.cell(current_row, 2).value == 'Off':
            fan = 0
        else:
            fan = worksheet.cell(current_row, 2).value

        if worksheet.cell(current_row, 3).value == 'null':
            mode = -1
        elif worksheet.cell(current_row, 3).value == 'Undefined':
            mode = -2
        elif worksheet.cell(current_row, 3).value == 'Auto':
            mode = 1
        elif worksheet.cell(current_row, 3).value == 'Cooling':
            mode = 2
        elif worksheet.cell(current_row, 3).value == 'Heating':
            mode = 3
        elif worksheet.cell(current_row, 3).value == 'Dehumidification':
            mode = 4
        elif worksheet.cell(current_row, 3).value == 'Off':
            mode = 0
        else:
            mode = worksheet.cell(current_row, 3).value

        if worksheet.cell(current_row, 4).value == 'null':
            Temperature = -1
        elif worksheet.cell(current_row, 4).value == 'Undefined':
            Temperature = -2
        elif worksheet.cell(current_row, 4).value == 'Auto':
            Temperature = 1
        elif worksheet.cell(current_row, 4).value == 'Cooling':
            Temperature = 2
        elif worksheet.cell(current_row, 4).value == 'Heating':
            Temperature = 3
        elif worksheet.cell(current_row, 4).value == 'Dehumidification':
            Temperature = 4
        elif worksheet.cell(current_row, 4).value == 'Off':
            Temperature = 0
        else:
            Temperature = worksheet.cell(current_row, 4).value

        if worksheet.cell(current_row, 5).value == 'null':
            Vane = -1
        elif worksheet.cell(current_row, 5).value == 'Undefined':
            Vane = -2
        elif worksheet.cell(current_row, 5).value == 'Auto':
            Vane = 1
        elif worksheet.cell(current_row, 5).value == 'Cooling':
            Vane = 2
        elif worksheet.cell(current_row, 5).value == 'Heating':
            Vane = 3
        elif worksheet.cell(current_row, 5).value == 'Dehumidification':
            Vane = 4
        elif worksheet.cell(current_row, 5).value == 'Off':
            Vane = 0
        else:
            Vane = worksheet.cell(current_row, 5).value

        current_date, current_time = date_and_time.split(" ")
        dates.append(current_date)
        times.append(current_time)
        current_sixth.append(current_date)
        current_sixth.append(current_time)
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
        current_row[1] = (float(fan))
        current_row[2] = (float(mode))
        current_row[3] = (float(Temperature))
        current_row[4] = (float(Vane))
        database.append(current_row)

    return np.asarray(database)


# number of colums 10
# ((year month day hours mins seconds) mode  temperature ) thermostat
# if Undefined = -2
# if null = -1
# if 0ff = 0
# if AutoEco = 1
# if AutoComfort = 2
# if Heating = 3
# if Antifreeze = 4
def read_data_ThermoStat_(given_path):
    database = xlrd.open_workbook(given_path)
    worksheet = database.sheet_by_index(0)
    shape = (3)
    dates = []
    times = []
    database = []
    for current_row in range(worksheet.nrows):
        current_sixth = []
        device = str((worksheet.cell(current_row, 0).value))
        date_and_time = str((worksheet.cell(current_row, 1).value))
        if worksheet.cell(current_row, 2).value == 'null':
            mode = -1
        elif str(worksheet.cell(current_row, 2).value) == 'Undefined':
            mode = -2
        elif worksheet.cell(current_row, 2).value == 'AutoEco':
            mode = 1
        elif worksheet.cell(current_row, 2).value == 'AutoComfort':
            mode = 2
        elif worksheet.cell(current_row, 2).value == 'Heating':
            mode = 3
        elif worksheet.cell(current_row, 2).value == 'Antifreeze':
            mode = 4
        elif worksheet.cell(current_row, 2).value == 'Off':
            mode = 0
        else:
            mode = worksheet.cell(current_row, 2).value

        if worksheet.cell(current_row, 3).value == 'null':
            temperature = -1
        elif worksheet.cell(current_row, 3).value == 'Undefined':
            temperature = -2
        elif worksheet.cell(current_row, 3).value == 'AutoEco':
            temperature = 1
        elif worksheet.cell(current_row, 3).value == 'AutoComfort':
            temperature = 2
        elif worksheet.cell(current_row, 3).value == 'Heating':
            temperature = 3
        elif worksheet.cell(current_row, 3).value == 'Antifreeze':
            temperature = 4
        elif worksheet.cell(current_row, 3).value == 'Off':
            temperature = 0
        else:
            temperature = worksheet.cell(current_row, 3).value

        current_date, current_time = date_and_time.split(" ")
        dates.append(current_date)
        times.append(current_time)
        current_sixth.append(current_date)
        current_sixth.append(current_time)
        year, month, day = current_sixth[0].split("-")
        hours, mins, seconds = current_sixth[1].split(":")
        current_row = np.zeros(shape)
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
        current_row[1] = (float(mode))
        current_row[2] = (float(temperature))
        database.append(current_row)

    return np.asarray(database)





########################################################################################################################
########################################################################################################################
##################################################AAC###################################################################
########################################################################################################################
########################################################################################################################
AAC_Control_1_9 = read_data_AAC_control_("data/Copy of Devices.ClimateControl.ACControl.1.9.xls")
AAC_Control_3_2 = read_data_AAC_control_("data/Copy of Devices.ClimateControl.ACControl.3.2.xls")
AAC_Control_6_5 = read_data_AAC_control_("data/Copy of Devices.ClimateControl.ACControl.6.5.xls")
AAC_Control_7_3 = read_data_AAC_control_("data/Copy of Devices.ClimateControl.ACControl.7.3.xls")
AAC_merge = np.concatenate((AAC_Control_1_9, AAC_Control_3_2, AAC_Control_6_5,
                            AAC_Control_7_3), axis=0)
# find the minimum value of time
minimum_time_AAC_merge = np.amin(AAC_merge, axis=0)
AAC_merge[:, 0] -= float(minimum_time_AAC_merge[0])
sorted_merge_acc = AAC_merge[np.argsort(AAC_merge[:, 0])]
thermoStat_1 = read_data_ThermoStat_("data/Copy of Devices.ClimateControl.Thermostat.1.xls")
thermoStat_2 = read_data_ThermoStat_("data/Copy of Devices.ClimateControl.Thermostat.2.xls")
thermoStat_3 = read_data_ThermoStat_("data/Copy of Devices.ClimateControl.Thermostat.3.xls")
thermoStat_merge = np.concatenate((thermoStat_1, thermoStat_2, thermoStat_3), axis=0)
minimum_time_thermoStat_merge = np.amin(thermoStat_merge, axis=0)
thermoStat_merge[:, 0] -= float(minimum_time_thermoStat_merge[0])
sorted_thermoStat_merge = thermoStat_merge[np.argsort(thermoStat_merge[:, 0])]
week = 604800
lists_count_aac = math.ceil(np.amax(sorted_merge_acc, axis=0)[0] / week)
lists_count_thermostat = math.ceil(np.amax(sorted_thermoStat_merge, axis=0)[0] / week)
list_max = max(lists_count_aac,lists_count_thermostat)


array_list_aac = [ [] for i in range(list_max) ]
array_list_aac_numpy = [ [] for i in range(list_max) ]
array_list_thermostat = [ [] for i in range(list_max) ]
array_list_thermostat_numpy = [ [] for i in range(list_max) ]
for i in range(list_max):
    array_list_aac[i].append([x for x in sorted_merge_acc if (i + 1) * week > x[0] >= i * week])
    array_list_aac_numpy[i] = (np.concatenate(np.asarray(array_list_aac[i]), axis=0))

for i in range(list_max):
    array_list_thermostat[i].append([x for x in sorted_thermoStat_merge if (i + 1) * week > x[0] >= i * week])
    array_list_thermostat_numpy[i] = (np.concatenate(np.asarray(array_list_thermostat[i]), axis=0))

for count in range(list_max):
    flag = 0
    if array_list_aac_numpy[count].size != 0:
        plt.scatter(array_list_aac_numpy[count][:, 0], array_list_aac_numpy[count][:, 1], s=0.1, c="blue")
        plt.scatter(array_list_aac_numpy[count][:, 0], array_list_aac_numpy[count][:, 2], s=0.1, c="green")
        plt.scatter(array_list_aac_numpy[count][:, 0], array_list_aac_numpy[count][:, 3], s=0.1, c="red")
        plt.scatter(array_list_aac_numpy[count][:, 0], array_list_aac_numpy[count][:, 4], s=0.1, c="orange")
        flag = 1
    if array_list_thermostat_numpy[count].size != 0:
        plt.scatter(array_list_thermostat_numpy[count][:, 0], array_list_thermostat_numpy[count][:, 1], s=0.5, c="black")
        plt.scatter(array_list_thermostat_numpy[count][:, 0], array_list_thermostat_numpy[count][:, 2], s=0.5, c="yellow")
        flag = 1
    if flag is 1:
        title_str = "Intersection aac and thermostat, week number " + str(count)
        plt.title(title_str)
        plt.xlabel('Time')
        plt.ylabel('Value')
        title_str_png = title_str + ".png"
        plt.savefig(title_str_png, dpi=300, bbox_inches='tight')
        plt.show()
        title_str_png = title_str + ".png"
########################################################################################################################
########################################################################################################################
####################### AAC ############################################################################################
########################################################################################################################
########################################################################################################################
# count = 0
# count_weeks_not_empty = 0
# for current_mat in array_list_aac_numpy:
#     if len(current_mat) is 0:
#         print("week number " + str(count) + " is empty.")
#         count += 1
#         continue
#     plt.scatter(current_mat[:, 0], current_mat[:, 1], s=0.1, c="blue")
#     plt.scatter(current_mat[:, 0], current_mat[:, 2], s=0.1, c="green")
#     plt.scatter(current_mat[:, 0], current_mat[:, 3], s=0.1, c="red")
#     plt.scatter(current_mat[:, 0], current_mat[:, 4], s=0.1, c="orange")
#     title_str = "AAC Control , day number " + str(count)
#     plt.title(title_str)
#     plt.xlabel('Time')
#     plt.ylabel('Value')
#     title_str_png = title_str + ".png"
#     plt.savefig(title_str_png, dpi=300, bbox_inches='tight')
#     plt.show()
#     title_str_png = title_str + ".png"
#     # plt.imsave(title_str_png,)
#     count += 1
#     count_weeks_not_empty+=1



########################################################################################################################
########################################################################################################################
####################### ThermoStat #####################################################################################
########################################################################################################################
########################################################################################################################

# count = 0
# count_weeks_not_empty = 0
# for current_mat in array_list_thermo_stat_numpy:
#     if len(current_mat) is 0:
#         print("week number " + str(count) + " is empty.")
#         count += 1
#         continue
#     plt.scatter(current_mat[:, 0], current_mat[:, 1], s=0.5, c="blue")
#     plt.scatter(current_mat[:, 0], current_mat[:, 2], s=0.5, c="black")
#     title_str = "ThermoStat, week number " + str(count)
#     plt.title(title_str)
#     plt.xlabel('Time')
#     plt.ylabel('Value')
#     title_str_png = title_str + ".png"
#     plt.savefig(title_str_png, dpi=300, bbox_inches='tight')
#     plt.show()
#     title_str_png = title_str + ".png"
#     # plt.imsave(title_str_png,)
#     count += 1
#     count_weeks_not_empty+=1
