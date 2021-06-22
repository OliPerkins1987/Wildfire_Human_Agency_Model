# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 16:16:40 2021

@author: Oli
"""

def load_dat(path):

    import os
    os.chdir(path)

    from Access_sharepoint import read_shpt_data, shpt_file_list, write_shpt_data

### import data

    file_list = shpt_file_list()
    file_list

    current_files = {'Attend': 'Data Corp/AttendDatav2.csv', 'Activities': 'Data Corp/ClientActivitiesv3a.csv', 
                 'Client': 'Data Corp/ClientDatav3.csv', 'Reg': 'Data Corp/RegDatav3.csv', 'IMD': 'processed/IMDData_transformed.csv'}

    dat = dict(zip([x for x in current_files.keys()], 
               [read_shpt_data(x) for x in current_files.values()]))

    return(dat)

    #os.chdir(r"E:\Data Corps\Dat\Current_SL_Dat")

    #for i in range(len(dat)):
    #
    #    name = [x for x in dat.keys()][i]
    #    file = [x for x in dat.values()][i]
    
        #file.to_csv(str(name+".csv"))
    
    