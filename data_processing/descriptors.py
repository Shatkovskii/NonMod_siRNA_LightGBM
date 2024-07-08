from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from PyBioMed.PyMolecule import connectivity
from CDK_pywrapper import CDK

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data_processing.sequant_funcs import aa_dict, seq_to_matrix_, SeQuant_encoding, generate_latent_representations
from data_processing.depscriptors_parsing import SCBDD, ForceField, parse_scbdd
from data_processing.load_unqiue import load_unique_descriptors


class RDKit:

    @staticmethod
    def generate_descriptors(normalize: tuple = (-1, 1), monomer_dict_: dict = None):
        if monomer_dict_ is None:
            monomer_dict_ = aa_dict

        descriptor_names = list(rdMolDescriptors.Properties.GetAvailableProperties())
        get_descriptors = rdMolDescriptors.Properties(descriptor_names)
        num_descriptors = len(descriptor_names)

        descriptors_set = np.empty((0, num_descriptors), float)

        for _, value in monomer_dict_.items():
            molecule = Chem.MolFromSmiles(value)
            descriptors = np.array(get_descriptors.ComputeProperties(molecule))
            descriptors = descriptors.reshape((-1, num_descriptors))
            descriptors_set = np.append(descriptors_set, descriptors, axis=0)

        sc = MinMaxScaler(feature_range=normalize)
        scaled_array = sc.fit_transform(descriptors_set)
        return pd.DataFrame(scaled_array, columns=descriptor_names, index=list(monomer_dict_.keys()))

    @staticmethod
    def encode_sequences(sequences_list, max_length=96):

        descriptors_set = RDKit.generate_descriptors()

        if not isinstance(sequences_list, list):
            sequences_list = [sequences_list]

        encoded_senses = SeQuant_encoding(sequences_list=sequences_list,
                                          polymer_type='RNA',
                                          descriptors=descriptors_set,
                                          num=96)

        x_senses = generate_latent_representations(sequences_list=encoded_senses,
                                                   sequant_encoded_sequences=encoded_senses,
                                                   polymer_type='RNA',
                                                   path_to_model_folder='../data/saved_models/nucleic_acids')

        return x_senses

    @staticmethod
    def encode_modified_sequences(sequences_list, max_length=27, normalize=True):

        descriptor_names = list(rdMolDescriptors.Properties.GetAvailableProperties())
        get_descriptors = rdMolDescriptors.Properties(descriptor_names)
        num_descriptors = len(descriptor_names)

        sc = MinMaxScaler(feature_range=(-1, 1))
        x_senses = np.empty((0, num_descriptors*max_length), float)

        for seq in sequences_list:
            descriptors_set = np.empty((0, num_descriptors), float)
            for smiles in seq:
                molecule = Chem.MolFromSmiles(smiles)
                try:
                    descriptors = np.array(get_descriptors.ComputeProperties(molecule))
                except:
                    print(smiles)
                    continue
                descriptors = descriptors.reshape((-1, num_descriptors))
                descriptors_set = np.append(descriptors_set, descriptors, axis=0)

            if normalize is True:
                scaled_array = sc.fit_transform(descriptors_set)
            else:
                scaled_array = descriptors_set

            scaled_df = pd.DataFrame(scaled_array, columns=descriptor_names)
            while scaled_df.shape[0] < 27:
                scaled_df.loc[scaled_df.shape[0]] = [0] * 43

            array = np.array(scaled_df)
            array = array.flatten()
            x_senses = np.vstack((x_senses, array))
        return x_senses


class PyBioMed:

    # Get names of PyBioMed descriptors, obtain their amount
    descriptor_names = list(connectivity._connectivity.keys())
    num_descriptors = len(descriptor_names)

    @staticmethod
    def generate_descriptors(normalize: tuple = (-1, 1), monomer_dict_: dict = None):
        if monomer_dict_ is None:
            monomer_dict_ = aa_dict

        # Get names of PyBioMed descriptors, obtain their amount
        descriptor_names = list(connectivity._connectivity.keys())
        num_descriptors = len(descriptor_names)

        # Create empty array where monomer descriptors will be kept
        descriptors_set = np.empty((0, num_descriptors), float)

        for _, value in monomer_dict_.items():
            molecule = Chem.MolFromSmiles(value)
            descriptors = list(connectivity.GetConnectivity(molecule).values())
            descriptors = np.array(descriptors)
            descriptors = descriptors.reshape((-1, num_descriptors))
            descriptors_set = np.append(descriptors_set, descriptors, axis=0)

        sc = MinMaxScaler(feature_range=normalize)
        scaled_array = sc.fit_transform(descriptors_set)
        return pd.DataFrame(scaled_array, columns=descriptor_names, index=list(monomer_dict_.keys()))

    @staticmethod
    def load_descriptors(normalize: tuple = (-1, 1), monomer_dict_: dict = None):
        """PyBioMed иногда падает с ошибкой, поскольку не поддерживает новый numpy. В этом случае сгенерировать
        дескрипторы не получится, так что придётся загружать из файла
        """

        if monomer_dict_ is None:
            monomer_dict_ = aa_dict

        desc_df = pd.read_excel('Datasets/Descriptors/pybiomed_descriptors.xlsx', index_col=0)

        descriptor_names = desc_df.columns.to_list()
        descriptors_set = [line.values for _, line in desc_df.iterrows()]
        descriptors_set = np.array(descriptors_set)

        sc = MinMaxScaler(feature_range=normalize)
        scaled_array = sc.fit_transform(descriptors_set)

        return pd.DataFrame(scaled_array, columns=descriptor_names, index=list(monomer_dict_.keys()))

    @staticmethod
    def encode_sequences(sequences_list, max_length=27):

        try:
            descriptors_set = PyBioMed.generate_descriptors()
        except:
            descriptors_set = PyBioMed.load_descriptors()

        if not isinstance(sequences_list, list):
            sequences_list = [sequences_list]

        container = []
        for sequence in sequences_list:
            seq_matrix = seq_to_matrix_(sequence=sequence,
                                        polymer_type='RNA',
                                        descriptors=descriptors_set,
                                        num=max_length)

            seq_matrix = np.array(seq_matrix).flatten()
            container.append(seq_matrix)

        return np.array(container)

    @staticmethod
    def descriptors_from_smiles(smiles):
        molecule = Chem.MolFromSmiles(smiles)
        try:
            descriptors = list(connectivity.GetConnectivity(molecule).values())
        except Exception as ex:
            print(ex)
            print(smiles, end='\n\n')
            return None

        descriptors = list(descriptors)
        # descriptors = descriptors.reshape((-1, PyBioMed.num_descriptors))
        return descriptors

    @staticmethod
    def encode_modified_sequences(sequences_list, max_length=27, normalize=True):

        # Create empty array where monomer descriptors will be kept
        x_senses = np.empty((0, max_length * PyBioMed.num_descriptors), float)

        for seq in sequences_list:
            descriptors_set = np.empty((0, PyBioMed.num_descriptors), float)
            for smiles in seq:
                molecule = Chem.MolFromSmiles(smiles)
                try:
                    descriptors = list(connectivity.GetConnectivity(molecule).values())
                    descriptors = np.array(descriptors)
                except Exception as ex:
                    print(ex)
                    print(smiles, end='\n\n')
                    continue
                descriptors = descriptors.reshape((-1, PyBioMed.num_descriptors))
                descriptors_set = np.append(descriptors_set, descriptors, axis=0)

            if normalize is True:
                sc = MinMaxScaler(feature_range=(-1, 1))
                scaled_array = sc.fit_transform(descriptors_set)
            else:
                scaled_array = descriptors_set

            scaled_df = pd.DataFrame(scaled_array, columns=PyBioMed.descriptor_names)
            while scaled_df.shape[0] < max_length:
                scaled_df.loc[scaled_df.shape[0]] = [0] * PyBioMed.num_descriptors

            array = np.array(scaled_df)
            array = array.flatten()
            x_senses = np.vstack((x_senses, array))
        return x_senses

    @staticmethod
    def encode_sequences_from_unique(sequences_list, max_length=27, normalize=True):

        unique_desc = load_unique_descriptors(descriptors=PyBioMed)

        # Create empty array where monomer descriptors will be kept
        x_senses = np.empty((0, max_length * PyBioMed.num_descriptors), float)

        for i, seq in enumerate(sequences_list, 1):
            descriptors_set = np.empty((0, PyBioMed.num_descriptors), float)
            print(f"{i}.", end=' ')
            for smiles in seq:
                try:
                    descriptors = unique_desc[smiles]
                    descriptors = np.array(descriptors)
                except Exception as ex:
                    print(ex)
                    print(smiles, end='\n\n')
                    break
                descriptors = descriptors.reshape((-1, PyBioMed.num_descriptors))
                descriptors_set = np.append(descriptors_set, descriptors, axis=0)
            print("\033[92m" + "done" + "\033[0m")
            if normalize is True:
                sc = MinMaxScaler(feature_range=(-1, 1))
                scaled_array = sc.fit_transform(descriptors_set)
            else:
                scaled_array = descriptors_set

            scaled_df = pd.DataFrame(scaled_array, columns=PyBioMed.descriptor_names)
            while scaled_df.shape[0] < max_length:
                scaled_df.loc[scaled_df.shape[0]] = [0] * PyBioMed.num_descriptors

            array = np.array(scaled_df)
            array = array.flatten()
            x_senses = np.vstack((x_senses, array))
        return x_senses


class CDK_:

    num_descriptors = 223
    descriptor_names = ['Fsp3', 'nSmallRings', 'nAromRings', 'nRingBlocks', 'nAromBlocks', 'nRings3', 'nRings4', 'nRings5', 'nRings6', 'nRings7', 'nRings8', 'nRings9', 'tpsaEfficiency', 'Zagreb', 'XLogP', 'WPATH', 'WPOL', 'WTPT-1', 'WTPT-2', 'WTPT-3', 'WTPT-4', 'WTPT-5', 'MW', 'VAdjMat', 'VABC', 'TopoPSA', 'LipinskiFailures', 'nRotB', 'topoShape', 'PetitjeanNumber', 'MDEC-11', 'MDEC-12', 'MDEC-13', 'MDEC-14', 'MDEC-22', 'MDEC-23', 'MDEC-24', 'MDEC-33', 'MDEC-34', 'MDEC-44', 'MDEO-11', 'MDEO-12', 'MDEO-22', 'MDEN-11', 'MDEN-12', 'MDEN-13', 'MDEN-22', 'MDEN-23', 'MDEN-33', 'MLogP', 'nAtomLAC', 'nAtomP', 'nAtomLC', 'khs.sLi', 'khs.ssBe', 'khs.ssssBe', 'khs.ssBH', 'khs.sssB', 'khs.ssssB', 'khs.sCH3', 'khs.dCH2', 'khs.ssCH2', 'khs.tCH', 'khs.dsCH', 'khs.aaCH', 'khs.sssCH', 'khs.ddC', 'khs.tsC', 'khs.dssC', 'khs.aasC', 'khs.aaaC', 'khs.ssssC', 'khs.sNH3', 'khs.sNH2', 'khs.ssNH2', 'khs.dNH', 'khs.ssNH', 'khs.aaNH', 'khs.tN', 'khs.sssNH', 'khs.dsN', 'khs.aaN', 'khs.sssN', 'khs.ddsN', 'khs.aasN', 'khs.ssssN', 'khs.sOH', 'khs.dO', 'khs.ssO', 'khs.aaO', 'khs.sF', 'khs.sSiH3', 'khs.ssSiH2', 'khs.sssSiH', 'khs.ssssSi', 'khs.sPH2', 'khs.ssPH', 'khs.sssP', 'khs.dsssP', 'khs.sssssP', 'khs.sSH', 'khs.dS', 'khs.ssS', 'khs.aaS', 'khs.dssS', 'khs.ddssS', 'khs.sCl', 'khs.sGeH3', 'khs.ssGeH2', 'khs.sssGeH', 'khs.ssssGe', 'khs.sAsH2', 'khs.ssAsH', 'khs.sssAs', 'khs.sssdAs', 'khs.sssssAs', 'khs.sSeH', 'khs.dSe', 'khs.ssSe', 'khs.aaSe', 'khs.dssSe', 'khs.ddssSe', 'khs.sBr', 'khs.sSnH3', 'khs.ssSnH2', 'khs.sssSnH', 'khs.ssssSn', 'khs.sI', 'khs.sPbH3', 'khs.ssPbH2', 'khs.sssPbH', 'khs.ssssPb', 'Kier1', 'Kier2', 'Kier3', 'HybRatio', 'nHBDon', 'nHBAcc', 'fragC', 'FMF', 'ECCEN', 'SP-0', 'SP-1', 'SP-2', 'SP-3', 'SP-4', 'SP-5', 'SP-6', 'SP-7', 'VP-0', 'VP-1', 'VP-2', 'VP-3', 'VP-4', 'VP-5', 'VP-6', 'VP-7', 'SPC-4', 'SPC-5', 'SPC-6', 'VPC-4', 'VPC-5', 'VPC-6', 'SC-3', 'SC-4', 'SC-5', 'SC-6', 'VC-3', 'VC-4', 'VC-5', 'VC-6', 'SCH-3', 'SCH-4', 'SCH-5', 'SCH-6', 'SCH-7', 'VCH-3', 'VCH-4', 'VCH-5', 'VCH-6', 'VCH-7', 'C1SP1', 'C2SP1', 'C1SP2', 'C2SP2', 'C3SP2', 'C1SP3', 'C2SP3', 'C3SP3', 'C4SP3', 'bpol', 'nB', 'BCUTw-1l', 'BCUTw-1h', 'BCUTc-1l', 'BCUTc-1h', 'BCUTp-1l', 'BCUTp-1h', 'nBase', 'ATSp1', 'ATSp2', 'ATSp3', 'ATSp4', 'ATSp5', 'ATSm1', 'ATSm2', 'ATSm3', 'ATSm4', 'ATSm5', 'ATSc1', 'ATSc2', 'ATSc3', 'ATSc4', 'ATSc5', 'nAtom', 'nAromBond', 'naAromAtom', 'apol', 'ALogP', 'ALogp2', 'AMR', 'nAcid', 'JPLogP']

    @staticmethod
    def generate_descriptors(normalize: tuple = (-1, 1), monomer_dict_: dict = None):
        if monomer_dict_ is None:
            monomer_dict_ = aa_dict

        smiles_list = list(monomer_dict_.values())
        mols = [Chem.AddHs(Chem.MolFromSmiles(smiles)) for smiles in smiles_list]

        cdk = CDK()
        desc_df = cdk.calculate(mols)

        descriptor_names = desc_df.columns.to_list()
        descriptors_set = [line.values for _, line in desc_df.iterrows()]
        descriptors_set = np.array(descriptors_set)

        sc = MinMaxScaler(feature_range=normalize)
        scaled_array = sc.fit_transform(descriptors_set)
        return pd.DataFrame(scaled_array, columns=descriptor_names, index=list(monomer_dict_.keys()))

    @staticmethod
    def encode_sequences(sequences_list, max_length=27):

        descriptors_set = CDK_.generate_descriptors()

        if not isinstance(sequences_list, list):
            sequences_list = [sequences_list]

        container = []
        for sequence in sequences_list:
            seq_matrix = seq_to_matrix_(sequence=sequence,
                                        polymer_type='RNA',
                                        descriptors=descriptors_set,
                                        num=max_length)

            seq_matrix = np.array(seq_matrix).flatten()
            container.append(seq_matrix)

        return np.array(container)

    @staticmethod
    def descriptors_from_smiles(smiles):

        mol = [Chem.AddHs(Chem.MolFromSmiles(smiles))]
        cdk = CDK()

        try:
            descriptors = cdk.calculate(mol).loc[0].values
        except Exception as ex:
            print(ex)
            print(smiles, end='\n\n')
            return None

        descriptors = list(descriptors)
        return descriptors

    @staticmethod
    def encode_sequences_from_unique(sequences_list, max_length=27, normalize=True):

        unique_desc = load_unique_descriptors(descriptors=CDK_)

        # Create empty array where monomer descriptors will be kept
        x_senses = np.empty((0, max_length * CDK_.num_descriptors), float)

        for i, seq in enumerate(sequences_list, 1):
            descriptors_set = np.empty((0, CDK_.num_descriptors), float)
            print(f"{i}.", end=' ')
            for smiles in seq:
                try:
                    descriptors = unique_desc[smiles]
                    descriptors = np.array(descriptors)
                except Exception as ex:
                    print(ex)
                    print(smiles, end='\n\n')
                    break
                descriptors = descriptors.reshape((-1, CDK_.num_descriptors))
                descriptors_set = np.append(descriptors_set, descriptors, axis=0)
            print("\033[92m" + "done" + "\033[0m")
            if normalize is True:
                sc = MinMaxScaler(feature_range=(-1, 1))
                scaled_array = sc.fit_transform(descriptors_set)
            else:
                scaled_array = descriptors_set

            scaled_df = pd.DataFrame(scaled_array, columns=CDK_.descriptor_names)
            while scaled_df.shape[0] < max_length:
                scaled_df.loc[scaled_df.shape[0]] = [0] * CDK_.num_descriptors

            array = np.array(scaled_df)
            array = array.flatten()
            x_senses = np.vstack((x_senses, array))
        return x_senses


class Mordred:
    @staticmethod
    def generate_descriptors(normalize: tuple = (-1, 1), monomer_dict_: dict = None):
        """
        Mordred не завёлся локально, но нормально заработал в google.colab, поэтому просто сгенерил дескрипторы там.
        Дескрипторы уже нормализованы на (-1, 1). Код:

                import pandas as pd
                import numpy as np
                from sklearn.preprocessing import MinMaxScaler

                smiles_list = list(aa_dict.values())
                mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

                calc = Calculator(descriptors, ignore_3D=False)
                desc_df = calc.pandas(mols)

                descriptor_names = desc_df.columns.to_list()
                descriptors_set = [line.values for _, line in desc_df.iterrows()]
                descriptors_set = np.array(descriptors_set)

                normalize = (-1, 1)
                sc = MinMaxScaler(feature_range=normalize)
                scaled_array = sc.fit_transform(descriptors_set)
                df = pd.DataFrame(scaled_array, columns=descriptor_names, index=list(aa_dict.keys()))
        """

        return pd.read_excel('Datasets/Descriptors/mordred_descriptors.xlsx', index_col=0)

    @staticmethod
    def encode_sequences(sequences_list, max_length=27):

        descriptors_set = Mordred.generate_descriptors()

        if not isinstance(sequences_list, list):
            sequences_list = [sequences_list]

        container = []
        for sequence in sequences_list:
            seq_matrix = seq_to_matrix_(sequence=sequence,
                                        polymer_type='RNA',
                                        descriptors=descriptors_set,
                                        num=max_length)

            seq_matrix = np.array(seq_matrix).flatten()
            container.append(seq_matrix)

        return np.array(container)


class Chemopy:
    @staticmethod
    def parse_descriptors():
        """ Библиотека chemopy2 требует для работы библиотеку openbabel, её трудно установить с ходу,
        поэтому решил пока что просто запарсить дескрипторы с сайта http://www.scbdd.com/chemopy_desc/index/
        см. модуль descriptors_parsing.py
        """

        parse_scbdd(descriptors=SCBDD.Chemopy)

    @staticmethod
    def generate_descriptors(normalize: tuple = (-1, 1), monomer_dict_: dict = None):

        """Chemopy дескрипторы для словаря мономеров aa_dict запаршены с сайта с сайта:
        http://www.scbdd.com/chemopy_desc/index/
        Здесь мы просто их нормируем на диапазоне normalize
        """

        if monomer_dict_ is None:
            monomer_dict_ = aa_dict

        desc_df = pd.read_excel('Datasets/Descriptors/chemopy_descriptors.xlsx', index_col=0)

        descriptor_names = desc_df.columns.to_list()
        descriptors_set = [line.values for _, line in desc_df.iterrows()]
        descriptors_set = np.array(descriptors_set)

        sc = MinMaxScaler(feature_range=normalize)
        scaled_array = sc.fit_transform(descriptors_set)

        return pd.DataFrame(scaled_array, columns=descriptor_names, index=list(monomer_dict_.keys()))

    @staticmethod
    def encode_sequences(sequences_list, max_length=27):

        descriptors_set = Chemopy.generate_descriptors()

        if not isinstance(sequences_list, list):
            sequences_list = [sequences_list]

        container = []
        for sequence in sequences_list:
            seq_matrix = seq_to_matrix_(sequence=sequence,
                                        polymer_type='RNA',
                                        descriptors=descriptors_set,
                                        num=max_length)

            seq_matrix = np.array(seq_matrix).flatten()
            container.append(seq_matrix)

        return np.array(container)


class BlueDesc:
    @staticmethod
    def parse_descriptors():
        """ BlueDesc изначально написан на java, чтобы долго не возиться с имплементацией было решено пока,
        в качестве временного решения, просто запарсить дескрипторы с сайта http://www.scbdd.com/blue_desc/index/
        см. модуль descriptors_parsing.py

        В качестве доп. параметра при расчёте дескрипторов нужно передать значение ForceField
        """

        parse_scbdd(descriptors=SCBDD.BlueDesc, force_field=ForceField.mmff94)
        parse_scbdd(descriptors=SCBDD.BlueDesc, force_field=ForceField.mmff94s)
        parse_scbdd(descriptors=SCBDD.BlueDesc, force_field=ForceField.ghemical)
        parse_scbdd(descriptors=SCBDD.BlueDesc, force_field=ForceField.gaff)
        parse_scbdd(descriptors=SCBDD.BlueDesc, force_field=ForceField.uff)

    @staticmethod
    def generate_descriptors(normalize: tuple = (-1, 1), monomer_dict_: dict = None, force_field=None):

        """BlueDesc дескрипторы для словаря мономеров aa_dict запаршены с сайта с сайта:
        http://www.scbdd.com/blue_desc/index/ в пяти вариациях, с ForceField mmff94, mmff94s, ghemical, gaff, uff
        Здесь мы их нормируем на диапазоне normalize
        """

        if monomer_dict_ is None:
            monomer_dict_ = aa_dict

        filename = 'Datasets/Descriptors/bluedesc_descriptors_%s.xlsx' % force_field
        desc_df = pd.read_excel(filename, index_col=0)

        descriptor_names = desc_df.columns.to_list()
        descriptors_set = [line.values for _, line in desc_df.iterrows()]
        descriptors_set = np.array(descriptors_set)
        monomer_dict_ = {
            monomer: monomer_dict_[monomer]
            for monomer in monomer_dict_ if monomer in desc_df.index
        }

        sc = MinMaxScaler(feature_range=normalize)
        scaled_array = sc.fit_transform(descriptors_set)

        return pd.DataFrame(scaled_array, columns=descriptor_names, index=list(monomer_dict_.keys()))

    @staticmethod
    def encode_sequences(sequences_list, max_length=27, force_field=None):

        descriptors_set = BlueDesc.generate_descriptors(force_field=force_field)

        if not isinstance(sequences_list, list):
            sequences_list = [sequences_list]

        container = []
        for sequence in sequences_list:
            seq_matrix = seq_to_matrix_(sequence=sequence,
                                        polymer_type='RNA',
                                        descriptors=descriptors_set,
                                        num=max_length)

            seq_matrix = np.array(seq_matrix).flatten()
            container.append(seq_matrix)

        return np.array(container)


class PaDEL:
    @staticmethod
    def parse_descriptors():
        """ BlueDesc изначально написан на java, чтобы долго не возиться с имплементацией было решено пока,
        в качестве временного решения, просто запарсить дескрипторы с сайта http://www.scbdd.com/padel_desc/index/
        см. модуль descriptors_parsing.py

        В качестве доп. параметра при расчёте дескрипторов нужно передать значение ForceField
        """

        parse_scbdd(descriptors=SCBDD.PaDEL, force_field=ForceField.mmff94)
        parse_scbdd(descriptors=SCBDD.PaDEL, force_field=ForceField.mm2)
        parse_scbdd(descriptors=SCBDD.PaDEL, force_field=None)

    @staticmethod
    def generate_descriptors(normalize: tuple = (-1, 1), monomer_dict_: dict = None, force_field=None):

        """BlueDesc дескрипторы для словаря мономеров aa_dict запаршены с сайта с сайта:
        http://www.scbdd.com/blue_desc/index/ в трёх вариациях, с ForceField mmff94, mm2, None
        Здесь мы их нормируем на диапазоне normalize
        """

        if monomer_dict_ is None:
            monomer_dict_ = aa_dict

        if force_field is not None:
            filename = 'Datasets/Descriptors/padel_descriptors_%s.xlsx' % force_field
        else:
            filename = '../data/datasets/unmodified/descriptors/padel_descriptors_none_ff.xlsx'
        desc_df = pd.read_excel(filename, index_col=0)

        descriptor_names = desc_df.columns.to_list()
        descriptors_set = [line.values for _, line in desc_df.iterrows()]
        descriptors_set = np.array(descriptors_set)

        sc = MinMaxScaler(feature_range=normalize)
        scaled_array = sc.fit_transform(descriptors_set)

        return pd.DataFrame(scaled_array, columns=descriptor_names, index=list(monomer_dict_.keys()))

    @staticmethod
    def encode_sequences(sequences_list, max_length=27, force_field=None):

        descriptors_set = PaDEL.generate_descriptors(force_field=force_field)

        if not isinstance(sequences_list, list):
            sequences_list = [sequences_list]

        container = []
        for sequence in sequences_list:
            seq_matrix = seq_to_matrix_(sequence=sequence,
                                        polymer_type='RNA',
                                        descriptors=descriptors_set,
                                        num=max_length)

            seq_matrix = np.array(seq_matrix).flatten()
            container.append(seq_matrix)

        return np.array(container)


class Descriptors:

    RDKit = RDKit
    PyBioMed = PyBioMed
    CDK = CDK_
    Mordred = Mordred
    Chemopy = Chemopy
    BlueDesc = BlueDesc
    PaDEL = PaDEL
