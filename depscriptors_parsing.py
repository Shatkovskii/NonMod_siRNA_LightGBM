import pandas as pd
import requests
from bs4 import BeautifulSoup

from sequant_funcs import aa_dict


class SCBDD:

    """All the descriptors options one can find on: http://www.scbdd.com/chemdes/"""

    Chemopy = 'chemopy_desc'
    CDK = 'cdk_desc'
    RDKit = 'rdk_desc'
    Pybel = 'pybel_desc'
    BlueDesc = 'blue_desc'
    PaDEL = 'padel_desc'


def parse_scbdd(descriptors, monomer_dict=None):
    if monomer_dict is None:
        monomer_dict = aa_dict

    descriptors_allowed = {SCBDD.Chemopy: 'Chemopy',
                           SCBDD.CDK: 'CDK',
                           SCBDD.RDKit: 'RDKit',
                           SCBDD.Pybel: 'Pybel',
                           SCBDD.BlueDesc: 'BlueDesc',
                           SCBDD.PaDEL: 'PaDEL'}

    if descriptors not in descriptors_allowed:
        log = ('Parse SCBDD Error:  unknown descriptors type\n'
               '\t\texpected:   SCBDD.Chemopy, SCBDD.CDK, SCBDD.RDKit, SCBDD.Pybel, SCBDD.BlueDesc or SCBDD.PaDEL\n'
               f'\t\tgot:        {descriptors}')
        print('\033[91m' + log + '\033[0m')
        exit(1)
    else:
        print('Start parsing {} descriptors'.format(descriptors_allowed[descriptors]), end='\n\n')

    url = 'http://www.scbdd.com/%s/index/' % descriptors
    filename = 'Datasets/Descriptors/%s_descriptors.xlsx' % descriptors_allowed[descriptors].lower()

    # Parse descriptors names
    response = requests.post(url, data={'Smiles': 'c1ccccc1', 'check_box_d': '3D'})
    soup = BeautifulSoup(response.text, 'html.parser')
    info_table = soup.find('table', class_='table table-bordered table-condensed')
    desc_lines = info_table.find_all('tr', class_='altrow')
    desc_names = [line.find_all('td')[1].get_text(strip=True) for line in desc_lines]

    # Create blank dataframe
    desc_df = pd.DataFrame(columns=desc_names, index=monomer_dict.keys())

    # Parse descriptors from monomer_dict
    for monomer_name, smiles in monomer_dict.items():
        data = {'Smiles': smiles, 'check_box_d': '3D'}
        response = requests.post(url, data=data)
        soup = BeautifulSoup(response.text, 'html.parser')

        info_table = soup.find('table', class_='table table-bordered table-condensed')
        desc_lines = info_table.find_all('tr', class_='altrow')

        for line in desc_lines:
            columns = [col.get_text(strip=True) for col in line.find_all('td')]
            desc_name = columns[1]
            desc_value = columns[2]
            desc_df.loc[monomer_name, desc_name] = desc_value

        # Save parsed descriptors dataset
        writer = pd.ExcelWriter(filename)
        desc_df.to_excel(writer, index=True)
        writer._save()

        log = '{}/{}. {} '.format(list(monomer_dict.keys()).index(monomer_name)+1, len(monomer_dict), monomer_name)
        print(log + '\033[92m' + 'done' + '\033[0m')
