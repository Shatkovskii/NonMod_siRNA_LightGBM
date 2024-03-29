import pandas as pd
import numpy as np
from force_field import ForceField


def minmax_normalization(values):
    min_x, max_x = min(values), max(values)
    delta = max_x - min_x
    return [(x-min_x)/delta * 2 - 1 for x in values]


def get_dataset(filename, descriptors,
                sense_column='Sense',
                antisense_column='AntiSense',
                concentration_column='Concentration, nM',
                efficacy_column='Efficacy, %',
                max_sequence_length=None,
                force_field=None):

    descriptors_allowed = ['RDKit',
                           'PyBioMed',
                           'CDK_',
                           'Mordred',
                           'Chemopy',
                           'BlueDesc',
                           'PaDEL']

    if not(hasattr(descriptors, '__name__') and descriptors.__name__ in descriptors_allowed):
        desc_names = ', '.join([
            'Descriptors.' + name.replace('_', '') for name in descriptors_allowed[:-1]
        ])
        desc_names += ' or Descriptors.' + descriptors_allowed[-1]

        log = ('Get Dataset Error:  unknown descriptors type\n'
               f'\t\t expected:  {desc_names}\n'
               f'\t\t got:       {descriptors}')
        print('\033[91m' + log + '\033[0m')
        exit(1)

    elif descriptors.__name__ == 'BlueDesc' and force_field not in [
        ForceField.mmff94, ForceField.mmff94s, ForceField.ghemical, ForceField.gaff, ForceField.uff
    ]:
        log = ('Get Dataset Error:  unknown force_field value\n'
               '\t\t expected:  ForceField.mmff94, ForceField.mmff94s, ForceField.ghemical, ForceField.gaff or '
               'ForceField.uff\n'
               f'\t\t got:       {force_field}')
        print('\033[91m' + log + '\033[0m')
        exit(1)

    elif descriptors.__name__ == 'PaDEL' and force_field not in [ForceField.mmff94, ForceField.mm2, None]:
        log = ('Get Dataset Error:  unknown force_field value\n'
               '\t\t expected:  ForceField.mmff94, ForceField.mm2 or None\n'
               f'\t\t got:       {force_field}')
        print('\033[91m' + log + '\033[0m')
        exit(1)
    else:
        print('Descriptors:', descriptors.__name__)
        if descriptors.__name__ in ['BlueDesc', 'PaDEL']:
            print('ForceField: ', force_field)

    df = pd.read_csv(filename)

    senses = df[sense_column].to_list()
    antisenses = df[antisense_column].to_list()
    concs = df[concentration_column].to_list()
    effs = df[efficacy_column].to_list()

    norm_effs = minmax_normalization(effs)
    norm_concs = minmax_normalization(concs)

    if not max_sequence_length:
        max_sequence_length = get_max_sequence_length(dataframe=df, sense_column=sense_column,
                                                      antisense_column=antisense_column)

    if descriptors.__name__ in ['BlueDesc', 'PaDEL']:
        x_senses = descriptors.encode_sequences(sequences_list=senses, max_length=max_sequence_length,
                                                force_field=force_field)
        x_antisenses = descriptors.encode_sequences(sequences_list=antisenses, max_length=max_sequence_length,
                                                    force_field=force_field)
    else:
        x_senses = descriptors.encode_sequences(sequences_list=senses, max_length=max_sequence_length)
        x_antisenses = descriptors.encode_sequences(sequences_list=antisenses, max_length=max_sequence_length)

    x = list()
    for i in range(len(x_senses)):
        m = np.hstack([x_senses[i], x_antisenses[i]])
        m = np.append(m, norm_concs[i])
        x.append(m)
    x = np.vstack(x)
    y = np.array(norm_effs)

    print('\nX.shape', x.shape)
    print('y.shape', y.shape, end='\n\n')

    return x, y


def get_max_sequence_length(database_filename=None, dataframe=None,
                            sense_column='Sense', antisense_column='AntiSense'):

    if database_filename is None and dataframe is None:
        return

    if dataframe is None:
        dataframe = pd.read_csv(database_filename, header=0)

    seqs = dataframe[sense_column].to_list() + dataframe[antisense_column].to_list()
    seqs = [len(i) for i in seqs]
    return max(seqs)
