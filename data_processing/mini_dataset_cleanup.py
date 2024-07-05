import re
import pandas as pd


# Ivan's dataset contains lines with extra characters in sequences (allowed chars are: AUGTC). Let's clean it up
def extra_characters_cleanup(filename, output_filename=None):
    df = pd.read_csv(filename, header=0, index_col=0)

    clean_df = pd.DataFrame(columns=df.columns)
    for i, line in df.iterrows():
        seqs = line['Sense'] + line['AntiSense']
        if not re.findall(r'[^A|^C|^G|^T|^U]', seqs):
            clean_df.loc[clean_df.shape[0]] = line

    if output_filename is None:
        output_filename = filename.replace('.xlsx', '').replace('.csv', '') + '_clean.csv'

    clean_df.to_csv(output_filename, index=False)


extra_characters_cleanup(filename='Datasets/Ivan-non-mod_3.csv',
                         output_filename='../data/datasets/unmodified/original_data/Ivan-non-mod_4.csv')

