import pandas as pd
import numpy as np

def load_data():
    cols_to_use = ['case_status', 'class_of_admission', 'country_of_citizenship', 'country_of_citzenship',
                   'employer_name', 'employer_state', 'pw_amount_9089', 'pw_soc_code']
    dtypes = {'pw_amount_9089', np.float64}
    data = pd.read_csv('data/us_perm_visas.csv', skiprows=0, usecols=cols_to_use, index_col=False)

    return data

us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'British Columbia': 'BC',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Marshall Islands': 'MH',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
    'Washington, DC': 'DC',
    'Northern Mariana Islands': 'MP',
    'Palau': 'PW',
    'Puerto Rico': 'PR',
    'Virgin Islands': 'VI',
    'District of Columbia': 'DC'
}

upper_us_state_dict = {}
def populate_upper_us_state_dict():
    for k, v in us_state_abbrev.items():
        upper_us_state_dict[k.upper()] = v

def convert_to_state_code(state):
    if state in upper_us_state_dict:
        return upper_us_state_dict.get(state)
    else:
        return state


def prepare_data():

    print('Reading file ..')
    data = load_data()
    print('File reading complete')

    print('Processing file ...')
    # processing the visa type columns
    data = data[data.class_of_admission == 'H-1B']
    # print(data.c)
    data.drop(['class_of_admission'], inplace=True, axis=1)

    # Combining the citizenship columns as they were exclusive duplicates with different spellings
    data["country_of_citizenship_combined"] = \
    data[['country_of_citizenship', 'country_of_citzenship']].fillna('').sum(axis=1)
    data.drop(columns=['country_of_citizenship', 'country_of_citzenship'], inplace=True)

    # drop rows that have any values missing in the selected columns
    reduced_data = data.mask(data.astype(object).eq('None')).dropna(axis=0)

    # processing column "pw_amount_9089"
    # pw_amount = reduced_data[['pw_amount_9089']]
    # pw_amount = pw_amount.replace({',': ''}, regex=True)
    # pw_amount = pw_amount.astype(float)
    # print(pw_amount.dtypes)

    # transforming individual columns

    # pw_amount_9089
    reduced_data['pw_amount_9089_processed'] = (reduced_data['pw_amount_9089']).replace({',': ''}, regex=True).astype(float)
    reduced_data.drop(['pw_amount_9089'], inplace=True, axis=1)

    # pw_soc_code
    reduced_data['pw_soc_code_processed'] = (reduced_data['pw_soc_code']).replace({'-': ''}, regex=True).astype(float)
    reduced_data.drop(['pw_soc_code'], inplace=True, axis=1)

    # country_of_citizenship_combined, will not remove collumns for manual verification
    labels, uniques = pd.factorize(reduced_data['country_of_citizenship_combined'], sort=True)
    reduced_data['country_of_citizenship_processed'] = labels

    # employer_state, will not remove collumns for manual verification
    populate_upper_us_state_dict()
    reduced_data['employer_state'] = reduced_data['employer_state'].apply(lambda x: convert_to_state_code(x))
    labels, uniques = pd.factorize(reduced_data['employer_state'], sort=True)
    reduced_data['employer_state_processed'] = labels

    # employer_name
    labels, uniques = pd.factorize(reduced_data['employer_name'], sort=True)
    reduced_data['employer_name_processed'] = labels

    # case_status
    labels, uniques = pd.factorize(reduced_data['case_status'], sort=True)
    print('unique statues - ')
    print(uniques)
    reduced_data['case_status_processed'] = labels
    # print(reduced_data['case_status'].value_counts())

    # print(reduced_data['class_of_admission'].value_counts())

    # change data types to neumric
    # print(reduced_data.dtypes)
    # print(reduced_data.shape)

    # reduced_data.to_csv('data/reduced_data.csv')

    print('File processing complete')
    return reduced_data, ['employer_name_processed', 'employer_state_processed', 'country_of_citizenship_processed',
                          'pw_soc_code_processed', 'pw_amount_9089_processed'], ['case_status_processed']

def save_processed_file_bd():
    reduced_data, c1, c2 = prepare_data()
    reduced_data.to_csv('data/reduced_data.csv')
