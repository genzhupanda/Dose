# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 08:33:19 2018
Purpose: clean the medication table for dose.
Input files:
           1. sah_pts_enc_med_sl_ad_cl2.tsv (created by Laila on 07/31/2018)
           2. sah_pts_all_enc_med_cl3.tsv (created by Laila on 08/01/2018)
           3. PAT_file_ENC.csv (created by Vahed on 11/02/2018)
           4. SAH_ENC_screened_24h_rule.csv (created by Vahed on 11/02/2018)
@author: Gen Zhu email: gen.zhu@uth.tmc.edu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def bar_annotation(ax, option='number'):
    '''
    This function is to add according number or percentage on every bar
    option: number or percentage, default is number
    ax: the figure object
    '''
    try:
        if option == 'number':
            for i in ax.patches:
                ax1.text(i.get_x(), i.get_height()+.5, i.get_height(),
                         fontsize=15, color='dimgrey')
        if option == 'percentage':
            totals = []
            for i in ax.patches:
                totals.append(i.get_height())
                total = sum(totals)
            for i in ax.patches:
                ax.text(i.get_x(), i.get_height()+.5,
                        str(round((i.get_height()/total)*100, 2))+'%',
                        fontsize=15, color='dimgrey')
    except:
        return('inputs are wrong')


def total_seconds(time):
    '''
    Convert the resolution of time to hours
    '''
    return time.total_seconds()/3600


# -----------------------------------------------------------------------------------
# Load the medication tables
# -----------------------------------------------------------------------------------

pd.options.display.max_rows = 50
pd.options.display.max_columns = 10
root_dir = 'H:Research/Hand_dirty/SAH_data/'
enc_med_cl2 = pd.read_csv(root_dir+'medication/sah_pts_enc_med_sl_ad_cl2.tsv',
                          delimiter='\t')
enc_med_cl3 = pd.read_csv(root_dir+'medication/sah_pts_all_enc_med_cl3.tsv',
                          delimiter='\t')
# In order to get the dose quantity,
# we need to merge some columns between enc_med_cl2 and enc_med_cl3
enc_med_cl2 = pd.merge(enc_med_cl2, enc_med_cl3[['INDEX_ID', 'DOSE_QUANTITY',
                       'ORDER_NO', 'TOTAL_DISPENSED_DOSES', 'CREDIT_QUANTITY',
                       'CHARGE_QUANTITY', 'INFUSION_RATE', 'INFUSION_TIME',
                       'DOSE_FORM_DESCRIPTION', 'ROUTE_DESCRIPTION',
                       'INITIAL_DOSE_QUANTITY']],
                        on='INDEX_ID')

# -----------------------------------------------------------------------------------
# Summary statistis for some variables
# -----------------------------------------------------------------------------------

# Encounter id
enc_med_cl2['ENCOUNTER_ID'].nunique()  # 4881

# Medication id
enc_med_cl2['MEDICATION_ID'].nunique()  # 70
enc_med_cl2.groupby('GENERIC_NAME')['MEDICATION_ID'].nunique()
# Load NDC_codes information
vasopressor_ndc_codes = pd.read_csv(
        root_dir+'medication/list of vasopressors with ndc codes.tsv',
        delimiter='\t')
vasopressor_ndc_codes = vasopressor_ndc_codes[[
        'MEDICATION_ID', 'NDC_CODE']].drop_duplicates()
vasopressor_ndc_codes['MEDICATION_ID'].nunique()
vasopressor_ndc_codes['NDC_CODE'].nunique()
enc_med_cl2 = pd.merge(vasopressor_ndc_codes, enc_med_cl2, on='MEDICATION_ID',
                       how='inner')
# NDC code:0517-1805-25 to verify my theory of unit strength

# Dose units
enc_med_cl2['DOSE_UNITS'].unique()
enc_med_cl2['DOSE_UNITS'].value_counts(dropna=False)
# Dose form description
enc_med_cl2['DOSE_FORM_DESCRIPTION'].unique()
enc_med_cl2['DOSE_FORM_DESCRIPTION'].value_counts(dropna=False)
# Route description
enc_med_cl2['ROUTE_DESCRIPTION'].unique()
enc_med_cl2['ROUTE_DESCRIPTION'].value_counts(dropna=False)


# Product strength description
enc_med_cl2['PRODUCT_STRENGTH_DESCRIPTION'].unique()
# Unify the unit in PRODUCT_STRENGTH_DESRIPTION,
# because some with ml but some with mL
enc_med_cl2['PRODUCT_STRENGTH_DESCRIPTION'] = enc_med_cl2[
        'PRODUCT_STRENGTH_DESCRIPTION'].str.replace("mL", "ml")
enc_med_cl2['PRODUCT_STRENGTH_DESCRIPTION'].unique()
enc_med_cl2['PRODUCT_STRENGTH_DESCRIPTION'].value_counts(dropna=False)
enc_med_cl2.groupby(['GENERIC_NAME', 'PRODUCT_STRENGTH_DESCRIPTION'])[
        'NDC_CODE'].value_counts(dropna=False)
enc_med_cl2.groupby(['GENERIC_NAME', 'ROUTE_DESCRIPTION'])[
        'PRODUCT_STRENGTH_DESCRIPTION'].value_counts(dropna=False)

# Time difference between the med start and med stop time
enc_med_cl2['MED_STARTED_DT_TM'] = pd.to_datetime(
        enc_med_cl2['MED_STARTED_DT_TM'])
start_tm_na_ind = (enc_med_cl2['MED_STARTED_DT_TM'].apply(str)
                   == '01-JAN-1000 00:00:00')
sum(start_tm_na_ind)  # 0
stop_tm_na_ind = (enc_med_cl2['MED_STOPPED_DT_TM'].apply(str)
                  == '01-JAN-1000 00:00:00')
sum(stop_tm_na_ind)  # 135
# Set the 01-JAN-1000 00:00:00 as NA
enc_med_cl2['MED_STOPPED_DT_TM'][stop_tm_na_ind] = np.nan
enc_med_cl2['MED_STOPPED_DT_TM'] = pd.to_datetime(
        enc_med_cl2['MED_STOPPED_DT_TM'])
enc_med_cl2['stop_start_gap'] = enc_med_cl2['MED_STOPPED_DT_TM'] -\
                 enc_med_cl2['MED_STARTED_DT_TM']
enc_med_cl2['stop_start_gap'].describe()
enc_med_cl2['stop_start_gap'].value_counts(dropna=False)
enc_med_cl2['stop_start_gap'] = enc_med_cl2['stop_start_gap'].\
                                apply(total_seconds)
bins = [-24*2, -0.0000001, 0, 1*24, 2*24, 5*24, 10*30, 30*30, 704*24]
time_interval = ['-2 days - 0', '0', '0-1 day', '1 day-2 days',
                 '2 days-5 days', '5 days -10 days', '10 days-30 days',
                 '>30 days']
gap_cutted = pd.cut(enc_med_cl2['stop_start_gap'], bins, labels=time_interval)
ax = pd.value_counts(gap_cutted, sort=False).plot(kind='bar')
bar_annotation(ax, 'percentage')
ax.set_xlabel('Time gap between medication started time and stopped time')
ax.set_ylabel('Counts')
stop_start_gap_0 = enc_med_cl2.query('stop_start_gap==0')  # 8830
stop_start_gap_0[['CONSUMED_QUANTITY', 'ORDER_STRENGTH',
                  'DOSE_QUANTITY']].describe()
stop_start_gap_negative = enc_med_cl2.query('stop_start_gap < 0')  # 28
stop_start_gap_negative[['MED_STARTED_DT_TM', 'MED_STOPPED_DT_TM',
                         'MED_ENTERED_DT_TM', 'MED_DISCONTINUED_DT_TM',
                         'GENERIC_NAME', 'NDC_CODE', 'MED_ORDER_STATUS_ID']]
stop_start_gap_exceed2d = enc_med_cl2.query('stop_start_gap > 2*24')
stop_start_gap_exceed2d.shape[0]  # 2149
stop_start_gap_exceed2d[['CONSUMED_QUANTITY', 'ORDER_STRENGTH',
                         'DOSE_QUANTITY']].describe()
# Roughly check consumed quantity / dose quantity
stop_start_gap_exceed2d['dose_frequency'] = stop_start_gap_exceed2d[
        'CONSUMED_QUANTITY']/stop_start_gap_exceed2d['DOSE_QUANTITY']
stop_start_gap_exceed2d['dose_frequency'].describe()
# Roughly guess the infusion time
stop_start_gap_exceed2d['infusion_period'] = stop_start_gap_exceed2d.eval(
        'stop_start_gap/dose_frequency')
stop_start_gap_exceed2d['infusion_period'].describe()

# Total dispensed doses
enc_med_cl2['TOTAL_DISPENSED_DOSES'].describe()
num_na = sum(enc_med_cl2['TOTAL_DISPENSED_DOSES'] == 0)  # 10
enc_med_cl2['TOTAL_DISPENSED_DOSES'].hist(bins=100)
bins = [0, 6, 20, 100, 500, 1100]
bins_counts = pd.cut(enc_med_cl2['TOTAL_DISPENSED_DOSES'], bins)
ax1 = pd.value_counts(bins_counts, sort=False).plot(kind='bar')
bar_annotation(ax1, 'number')
ax1.set_xlabel('Total dispensed doses')
ax1.set_ylabel('Counts')
# How many abservations that dose quantity is larger than total dispensed doses
dq_exceed_tdd_ind = enc_med_cl2['DOSE_QUANTITY'] >\
                     enc_med_cl2['TOTAL_DISPENSED_DOSES']
dq_exceed_tdd = enc_med_cl2[dq_exceed_tdd_ind]
dq_exceed_tdd.shape[0]  # 1468
dq_exceed_tdd['stop_start_gap'].describe()
dq_exceed_tdd['stop_start_gap'].hist(bins=100)
tdd_over500 = enc_med_cl2.query('TOTAL_DISPENSED_DOSES>500')
tdd_over500[['NDC_CODE', 'DOSE_QUANTITY', 'CONSUMED_QUANTITY',
             'TOTAL_DISPENSED_DOSES', 'ORDER_STRENGTH', 'ORDER_VOLUME',
             'PRODUCT_STRENGTH_DESCRIPTION', 'ROUTE_DESCRIPTION',
             'stop_start_gap']]

# Consumed_quantity(no missing value)
enc_med_cl2['CONSUMED_QUANTITY'].describe()
ax0=enc_med_cl2['CONSUMED_QUANTITY'].hist(bins=100)
ax0.set_xlabel('Consumed quantity')
ax0.set_ylabel('Counts')
enc_med_cl2['CONSUMED_QUANTITY'].value_counts(dropna=False)  # count of 0 is 0
print("Top ten consumed quantity in terms of counts: \n{}".format(
        enc_med_cl2['CONSUMED_QUANTITY'].value_counts(dropna=False)[0:10, ]))
bins = [0, 5, 10, 20, 100, 200, 1000, 13760]
bins_counts = pd.cut(enc_med_cl2['CONSUMED_QUANTITY'], bins)
ax1 = pd.value_counts(bins_counts, sort=False).plot(kind='bar')
bar_annotation(ax1, 'number')
ax1.set_xlabel('Consumed quantity')
ax1.set_ylabel('Counts')
cq_over1000 = enc_med_cl2.query('CONSUMED_QUANTITY>1000')
cq_over1000.shape[0]  # 121
cq_over1000[['NDC_CODE', 'DOSE_QUANTITY', 'CONSUMED_QUANTITY',
             'ORDER_STRENGTH', 'ORDER_VOLUME', 'PRODUCT_STRENGTH_DESCRIPTION',
             'ROUTE_DESCRIPTION', 'TOTAL_DISPENSED_DOSES',
             'stop_start_gap']]
cq_over1000.groupby('PRODUCT_STRENGTH_DESCRIPTION')['NDC_CODE'].value_counts()
cq_over1000['TOTAL_DISPENSED_DOSES'].hist()
cq_over1000['stop_start_gap'].hist(bins=200)
cq_over1000['DOSE_QUANTITY'].hist(bins=20)

# Order strength, now assume the unit for strength is mg, later I will prove
# it.
enc_med_cl2['ORDER_STRENGTH'].value_counts(dropna=False)
sns.boxplot(enc_med_cl2['ORDER_STRENGTH'])
enc_med_cl2['ORDER_STRENGTH'].hist(bins=100)
enc_med_cl2['ORDER_STRENGTH'].describe()
order_strength_max = enc_med_cl2.loc[enc_med_cl2['ORDER_STRENGTH'].idxmax()]
print(order_strength_max)
bins = [0, 10, 60, 200, 400, 800, 5600]
enc_med_cl2['order_strength_cutted'] = pd.cut(
        enc_med_cl2['ORDER_STRENGTH'], bins)
ax1 = pd.value_counts(enc_med_cl2['order_strength_cutted'],
                      sort=False).plot(kind='bar')
bar_annotation(ax1, 'percentage')
ax1.set_xlabel('Order_strength')
ax1.set_ylabel('Counts')
# See the relationship between order strength and route description
enc_med_cl2.groupby(['order_strength_cutted'])[
        'ROUTE_DESCRIPTION'].value_counts(dropna=False)
# See the extreme order strength
threshold = 800
# order strength > 800
os_over800 = enc_med_cl2.query('ORDER_STRENGTH > @threshold')
print('The number of observations whose order strength is over 800: {}'.format(
        os_over800.shape[0]))
os_over800[['NDC_CODE', 'DOSE_QUANTITY', 'CONSUMED_QUANTITY', 'ORDER_STRENGTH',
            'ORDER_VOLUME', 'PRODUCT_STRENGTH_DESCRIPTION',
            'ROUTE_DESCRIPTION']]
os_over800['CONSUMED_QUANTITY'].describe()
sns.boxplot(os_over800['CONSUMED_QUANTITY'])
os_over800['CONSUMED_QUANTITY'].hist(bins=100)
bins = [0, 3, 10, 20, 100, 720]
bins_vector = pd.cut(os_over800['CONSUMED_QUANTITY'], bins)
ax1 = pd.value_counts(bins_vector, sort=False).plot(kind='bar')
bar_annotation(ax1, 'number')
ax1.set_xlabel('Consumed quantity')
ax1.set_ylabel('Counts')
os_over800['TOTAL_DISPENSED_DOSES'].describe()
os_over800['TOTAL_DISPENSED_DOSES'].hist(bins=100)
os_over800['stop_start_gap'].describe()
# order strength = 800
os_800 = enc_med_cl2.query('ORDER_STRENGTH == @threshold')
print('The number of observations whose order strength is 800: {}'.format(
        os_800.shape[0]))
os_800[['NDC_CODE', 'DOSE_QUANTITY', 'CONSUMED_QUANTITY', 'ORDER_STRENGTH',
        'ORDER_VOLUME', 'PRODUCT_STRENGTH_DESCRIPTION', 'ROUTE_DESCRIPTION']]
os_800['PRODUCT_STRENGTH_DESCRIPTION'].value_counts()
os_800.groupby(['PRODUCT_STRENGTH_DESCRIPTION', 'NDC_CODE'])[
        'ORDER_VOLUME'].value_counts(dropna=False)
os_800_pst320 = os_800.query(
        'PRODUCT_STRENGTH_DESCRIPTION =="5%-320 mg/100 ml"')
os_800_pst320['DOSE_QUANTITY'].value_counts()
NDC_CODE_74781022 = os_800.query('NDC_CODE==74781022')
NDC_CODE_74781022[['NDC_CODE', 'DOSE_QUANTITY', 'CONSUMED_QUANTITY',
                   'ORDER_STRENGTH', 'ORDER_VOLUME', 'TOTAL_VOLUME',
                   'PRODUCT_STRENGTH_DESCRIPTION', 'ROUTE_DESCRIPTION']]
# Check the consumed quantity(Use initial_dose_quantity doulb check and
# time gap of med_stop and med_started)
os_800['CONSUMED_QUANTITY'].describe()
sns.boxplot(os_800['CONSUMED_QUANTITY'])
os_800['CONSUMED_QUANTITY'].hist(bins=100)
bins = [0, 3, 10, 20, 100, 720]
bins_vector = pd.cut(os_800['CONSUMED_QUANTITY'], bins)
ax1 = pd.value_counts(bins_vector, sort=False).plot(kind='bar')
bar_annotation(ax1, 'number')
ax1.set_xlabel('Consumed quantity')
ax1.set_ylabel('Counts')
os_800['TOTAL_DISPENSED_DOSES'].describe()
os_800['stop_start_gap'].describe()
fig, ax = plt.subplots(1, 2)
ax[0] = os_800['TOTAL_DISPENSED_DOSES'].hist(bins=100)
ax[0].set_xlabel('Total dispensed doses')
ax[0].set_ylabel('Counts')
ax[1] = os_800['stop_start_gap'].hist(bins=100)
ax[1].set_xlabel('stop time - start_time')
ax[1].set_ylabel('Counts')

# Dose_quantity
enc_med_cl2['DOSE_QUANTITY'].describe()
enc_med_cl2['DOSE_QUANTITY'].hist(bins=100)
enc_med_cl2['DOSE_QUANTITY'].value_counts(dropna=False)  # count of 0 is 16
missing_num = sum(enc_med_cl2['DOSE_QUANTITY'].isnull() |
                  enc_med_cl2['DOSE_QUANTITY'] == 0)
print('The missing number of dose quantity is: {}'.format(missing_num))
print("Top ten dose quantity in terms of counts: \n{}".format(
        enc_med_cl2['DOSE_QUANTITY'].value_counts(dropna=False)[0:10, ]))
bins = [0, 5, 10, 100, 500]
bins_counts = pd.cut(enc_med_cl2['DOSE_QUANTITY'], bins)
ax1 = pd.value_counts(bins_counts, sort=False).plot(kind='bar')
bar_annotation(ax1, 'number')
ax1.set_xlabel('Dose quantity')
ax1.set_ylabel('Counts')
# Take insight into the detail of extreme values of dose quantity
threshold = 10
dq_over10 = enc_med_cl2.query('DOSE_QUANTITY > @threshold')
print('The number of extreme values of dose quantity is : {}'.format(
        dq_over10.shape[0]))
dq_over10['DOSE_QUANTITY'].value_counts()
# Pay speciall attention to dose quantity 250
dq_250 = dq_over10.query('DOSE_QUANTITY>=250')
dq_250[['ENCOUNTER_ID', 'NDC_CODE','PRODUCT_STRENGTH_DESCRIPTION',
                'ORDER_STRENGTH', 'CONSUMED_QUANTITY', 'TOTAL_DISPENSED_DOSES',
                'ORDER_VOLUME', 'DOSE_QUANTITY','ROUTE_DESCRIPTION']]
dq_250_ndc = dq_250['NDC_CODE'].unique()
fulltable_with_dq_250_ndc = enc_med_cl2[enc_med_cl2['NDC_CODE'].isin(dq_250_ndc)]
fulltable_with_dq_250_ndc[['ENCOUNTER_ID', 'NDC_CODE','PRODUCT_STRENGTH_DESCRIPTION',
                'ORDER_STRENGTH', 'CONSUMED_QUANTITY', 'TOTAL_DISPENSED_DOSES',
                'ORDER_VOLUME', 'DOSE_QUANTITY','ROUTE_DESCRIPTION']]
ax = fulltable_with_dq_250_ndc['DOSE_QUANTITY'].hist()
ax.set_xlabel('Dose quantity')
ax.set_ylabel('Counts')

# Strength units
enc_med_cl2['STRENGTH_UNITS'].value_counts(dropna=False)
NAs = enc_med_cl2[(enc_med_cl2["STRENGTH_UNITS"] == "Not Mapped") |
                  enc_med_cl2["STRENGTH_UNITS"].isnull()]
ug = enc_med_cl2[enc_med_cl2["STRENGTH_UNITS"] == "ug"]
ug['PRODUCT_STRENGTH_DESCRIPTION'].value_counts(dropna=False)
mg = enc_med_cl2[enc_med_cl2["STRENGTH_UNITS"] == "mg"]
# With the unit as ug or NAs, compare the consumed quantity with the mg group
ug_mg_na_cq = pd.DataFrame({'consumed_quantity_NA': NAs['CONSUMED_QUANTITY'],
                        'consumed_quantity_ug': ug['CONSUMED_QUANTITY'],
                        'consumed_quantity_mg': mg['CONSUMED_QUANTITY']})
ug_mg_na_cq.describe()
# With the unit as ug or NAs, compare the order strength with mg group
# Note that many missing order strength
ug_mg_na_os = pd.DataFrame({'order_strength_NA': NAs['ORDER_STRENGTH'],
                        'order_strength_ug': ug['ORDER_STRENGTH'],
                        'order_strength_mg': mg['ORDER_STRENGTH']})
ug_mg_na_os.describe()


# How to impute the NAs
enc_med_cl2.groupby('PRODUCT_STRENGTH_DESCRIPTION')[
        'STRENGTH_UNITS'].value_counts(dropna=False)

# Analyze the missing mechanism of order strength and strength unit
# Get the table where ORDER_STRENGTH is 0 or Null
order_strength_null = enc_med_cl2[(enc_med_cl2["ORDER_STRENGTH"] == 0) |
                                  enc_med_cl2["ORDER_STRENGTH"].isnull()]
order_strength_null.shape  # (6100,29)
# to get the table where STRENGTH_UNITS is Not Mapped or Null
strength_unit_null = enc_med_cl2[(enc_med_cl2["STRENGTH_UNITS"] ==
                     "Not Mapped") | enc_med_cl2["STRENGTH_UNITS"].isnull()]
strength_unit_null.shape   # (6103,29)
# See the missing mechanism of STRENGTH_UNIT and ORDER_STRENGTH,
# actually to see whether they miss simultaneously
overlap_indic = np.in1d(strength_unit_null.index, order_strength_null.index)
overlap_indic2 = np.in1d(order_strength_null.index, strength_unit_null.index)
order_strength_null[~overlap_indic2]
strength_unit_null[~overlap_indic]
# For the three observations with STRENGTH_UNITS 'Not Mapped',
# they have ORDER_STRENGTH. But for the rest of the observations with
# missing STRENGTH_UNIT or ORDER_STRENGTH, these two variables are missed
# simultaneously

# Order volume
enc_med_cl2['ORDER_VOLUME'].describe()
ov_na = sum(enc_med_cl2['ORDER_VOLUME'].isnull())  # 4500
print(ov_na)
ax = enc_med_cl2['ORDER_VOLUME'].hist(bins=100)
ax.set_xlabel('Order volume')
ax.set_ylabel('Count')
print("Top ten order volume value in terms of counts: \n{}".format(
        enc_med_cl2['ORDER_VOLUME'].value_counts(dropna=False)[0:10, ]))
bins = [0, 5, 10, 20, 100, 200, 13760]
bins_counts = pd.cut(enc_med_cl2['CONSUMED_QUANTITY'], bins)
ax1 = pd.value_counts(bins_counts, sort=False).plot(kind='bar')
bar_annotation(ax1, 'percentage')
ax1.set_xlabel('Consumed_quantity')
ax1.set_ylabel('Counts')

# Total volume(may be useful for deriving fluid quantity)
enc_med_cl2['TOTAL_VOLUME'].describe()
missing_rate = sum(enc_med_cl2['TOTAL_VOLUME'].isnull())/enc_med_cl2.shape[0]  # 0
total_volume_0 = enc_med_cl2.query('TOTAL_VOLUME == 0')
total_volume_0.shape[0]
print('The number of zero value  of total volume is :{}'.format(
        total_volume_0.shape[0]))
ax = enc_med_cl2['TOTAL_VOLUME'].hist(bins=100)
ax.set_xlabel('TOTAL_VOLUME')
ax.set_ylabel('Counts')
print("Top ten total volume value in terms of counts: \n{}".format(
        enc_med_cl2['TOTAL_VOLUME'].value_counts()[0:10, ]))


# Infusion rate(mm/hour)
enc_med_cl2['INFUSION_RATE'].describe()
infusion_rate_null_ind = (enc_med_cl2['INFUSION_RATE'].isnull()) |\
                                 (enc_med_cl2['INFUSION_RATE'] == 0)
# Missing number
sum(infusion_rate_null_ind)  # 13628
enc_med_cl2['INFUSION_RATE'][~infusion_rate_null_ind].describe()

# Infusion time
enc_med_cl2['INFUSION_TIME'].describe()
infusion_time_null_ind = (enc_med_cl2['INFUSION_TIME'].isnull()) |\
                                 (enc_med_cl2['INFUSION_TIME'] == 0)
# Missing number
sum(infusion_time_null_ind)  # 13350
enc_med_cl2['INFUSION_TIME'][~infusion_time_null_ind].describe()
# I need to check the infusion time unit

# -----------------------------------------------------------------------------
# Derive the variable unit strength and concentration
# -----------------------------------------------------------------------------

# Calculate the unit strength for every medication ID,
# which is equal to order strength devided by dose quantity
enc_med_cl2['UNIT_STRENGTH'] = enc_med_cl2.eval('ORDER_STRENGTH/DOSE_QUANTITY')
enc_med_cl2.groupby(['PRODUCT_STRENGTH_DESCRIPTION', 'NDC_CODE'])[
        'UNIT_STRENGTH'].value_counts(dropna=False)
# Define a new variable called concentration, to see whether it is consistent
# with product strength description
enc_med_cl2['concentration'] = enc_med_cl2.eval('ORDER_STRENGTH/ORDER_VOLUME')
enc_med_cl2.groupby(['PRODUCT_STRENGTH_DESCRIPTION', 'MEDICATION_ID'])[
        'concentration'].value_counts(dropna=False)

# I have thought of a strategy to determine the unique unit strength: for
# multiple unit strength for a specific medication, examine case by case
# whether their concentration is consistent with product strength description
one_mg_ml_16980_16 = enc_med_cl2.query(
        'MEDICATION_ID==16980 & UNIT_STRENGTH==16')
# concentration is not consistent

one_mg_ml_16980_4 = enc_med_cl2.query(
        'MEDICATION_ID == 16980 & UNIT_STRENGTH==4')
one_mg_ml_25737_4 = enc_med_cl2.query(
        'MEDICATION_ID = =25737 & UNIT_STRENGTH==4')
one_mg_ml_25737_2 = enc_med_cl2.query(
        'MEDICATION_ID == 25737 & UNIT_STRENGTH==2')
# concentration is correct

one_mg_ml_7542819_8 = enc_med_cl2.query(
        'MEDICATION_ID==7542819 & UNIT_STRENGTH==8')
# concentration is not consistent

one_mg_ml_2235022_3 = enc_med_cl2.query(
        'MEDICATION_ID == 2235022 & UNIT_STRENGTH==3')
# concentration is consistent

ten_mg_ml_12894_1 = enc_med_cl2.query(
        'MEDICATION_ID==12894 & UNIT_STRENGTH==1')
# concentration is not consistent

ten_mg_ml_12894_2 = enc_med_cl2.query(
        'MEDICATION_ID==12894 & UNIT_STRENGTH==2')
# concentration is not consistent

ten_mg_ml_12896_40 = enc_med_cl2.query(
        'MEDICATION_ID==12896 & UNIT_STRENGTH==40')
# concentration is not consistent, no order volume

ten_mg_ml_12896_10 = enc_med_cl2.query(
        'MEDICATION_ID==12896 & UNIT_STRENGTH==10')
# concentration is not consistent, no order volume

ten_mg_ml_22082_2 = enc_med_cl2.query(
        'MEDICATION_ID==22082 & UNIT_STRENGTH==2')
# concentration is consistent

ten_mg_ml_22082_10 = enc_med_cl2.query(
        'MEDICATION_ID==22082 & UNIT_STRENGTH==10')
# concentration is consistent





