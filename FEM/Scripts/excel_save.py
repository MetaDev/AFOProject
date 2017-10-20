# save results in excel
# import xlwings as xw
# wb = xw.Book(result_path)  # this will create a new workbook
# import datetime
# sht = wb.sheets.add(datetime.datetime.now().strftime(
#     '%m-%d-%h-%M') + " axisdim-" + str(n_sensor_axis))
# sht.range('A1').value = np.concatenate( 
#     (np.mean(degree_test,axis=0).reshape(7,-1),
#     np.std(degree_test,axis=0).reshape(7,-1)[:,2:4]),axis=1)