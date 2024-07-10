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

    descriptor_names = "nAcid,nBase,SpAbs_A,SpMax_A,SpDiam_A,SpAD_A,SpMAD_A,LogEE_A,VE1_A,VE2_A,VE3_A,VR1_A,VR2_A,VR3_A,nAromAtom,nAromBond,nAtom,nHeavyAtom,nSpiro,nBridgehead,nHetero,nH,nB,nC,nN,nO,nS,nP,nF,nCl,nBr,nI,nX,ATS0dv,ATS1dv,ATS2dv,ATS3dv,ATS4dv,ATS5dv,ATS6dv,ATS7dv,ATS8dv,ATS0d,ATS1d,ATS2d,ATS3d,ATS4d,ATS5d,ATS6d,ATS7d,ATS8d,ATS0s,ATS1s,ATS2s,ATS3s,ATS4s,ATS5s,ATS6s,ATS7s,ATS8s,ATS0Z,ATS1Z,ATS2Z,ATS3Z,ATS4Z,ATS5Z,ATS6Z,ATS7Z,ATS8Z,ATS0m,ATS1m,ATS2m,ATS3m,ATS4m,ATS5m,ATS6m,ATS7m,ATS8m,ATS0v,ATS1v,ATS2v,ATS3v,ATS4v,ATS5v,ATS6v,ATS7v,ATS8v,ATS0se,ATS1se,ATS2se,ATS3se,ATS4se,ATS5se,ATS6se,ATS7se,ATS8se,ATS0pe,ATS1pe,ATS2pe,ATS3pe,ATS4pe,ATS5pe,ATS6pe,ATS7pe,ATS8pe,ATS0are,ATS1are,ATS2are,ATS3are,ATS4are,ATS5are,ATS6are,ATS7are,ATS8are,ATS0p,ATS1p,ATS2p,ATS3p,ATS4p,ATS5p,ATS6p,ATS7p,ATS8p,ATS0i,ATS1i,ATS2i,ATS3i,ATS4i,ATS5i,ATS6i,ATS7i,ATS8i,AATS0dv,AATS1dv,AATS2dv,AATS3dv,AATS4dv,AATS5dv,AATS6dv,AATS7dv,AATS8dv,AATS0d,AATS1d,AATS2d,AATS3d,AATS4d,AATS5d,AATS6d,AATS7d,AATS8d,AATS0s,AATS1s,AATS2s,AATS3s,AATS4s,AATS5s,AATS6s,AATS7s,AATS8s,AATS0Z,AATS1Z,AATS2Z,AATS3Z,AATS4Z,AATS5Z,AATS6Z,AATS7Z,AATS8Z,AATS0m,AATS1m,AATS2m,AATS3m,AATS4m,AATS5m,AATS6m,AATS7m,AATS8m,AATS0v,AATS1v,AATS2v,AATS3v,AATS4v,AATS5v,AATS6v,AATS7v,AATS8v,AATS0se,AATS1se,AATS2se,AATS3se,AATS4se,AATS5se,AATS6se,AATS7se,AATS8se,AATS0pe,AATS1pe,AATS2pe,AATS3pe,AATS4pe,AATS5pe,AATS6pe,AATS7pe,AATS8pe,AATS0are,AATS1are,AATS2are,AATS3are,AATS4are,AATS5are,AATS6are,AATS7are,AATS8are,AATS0p,AATS1p,AATS2p,AATS3p,AATS4p,AATS5p,AATS6p,AATS7p,AATS8p,AATS0i,AATS1i,AATS2i,AATS3i,AATS4i,AATS5i,AATS6i,AATS7i,AATS8i,ATSC0c,ATSC1c,ATSC2c,ATSC3c,ATSC4c,ATSC5c,ATSC6c,ATSC7c,ATSC8c,ATSC0dv,ATSC1dv,ATSC2dv,ATSC3dv,ATSC4dv,ATSC5dv,ATSC6dv,ATSC7dv,ATSC8dv,ATSC0d,ATSC1d,ATSC2d,ATSC3d,ATSC4d,ATSC5d,ATSC6d,ATSC7d,ATSC8d,ATSC0s,ATSC1s,ATSC2s,ATSC3s,ATSC4s,ATSC5s,ATSC6s,ATSC7s,ATSC8s,ATSC0Z,ATSC1Z,ATSC2Z,ATSC3Z,ATSC4Z,ATSC5Z,ATSC6Z,ATSC7Z,ATSC8Z,ATSC0m,ATSC1m,ATSC2m,ATSC3m,ATSC4m,ATSC5m,ATSC6m,ATSC7m,ATSC8m,ATSC0v,ATSC1v,ATSC2v,ATSC3v,ATSC4v,ATSC5v,ATSC6v,ATSC7v,ATSC8v,ATSC0se,ATSC1se,ATSC2se,ATSC3se,ATSC4se,ATSC5se,ATSC6se,ATSC7se,ATSC8se,ATSC0pe,ATSC1pe,ATSC2pe,ATSC3pe,ATSC4pe,ATSC5pe,ATSC6pe,ATSC7pe,ATSC8pe,ATSC0are,ATSC1are,ATSC2are,ATSC3are,ATSC4are,ATSC5are,ATSC6are,ATSC7are,ATSC8are,ATSC0p,ATSC1p,ATSC2p,ATSC3p,ATSC4p,ATSC5p,ATSC6p,ATSC7p,ATSC8p,ATSC0i,ATSC1i,ATSC2i,ATSC3i,ATSC4i,ATSC5i,ATSC6i,ATSC7i,ATSC8i,AATSC0c,AATSC1c,AATSC2c,AATSC3c,AATSC4c,AATSC5c,AATSC6c,AATSC7c,AATSC8c,AATSC0dv,AATSC1dv,AATSC2dv,AATSC3dv,AATSC4dv,AATSC5dv,AATSC6dv,AATSC7dv,AATSC8dv,AATSC0d,AATSC1d,AATSC2d,AATSC3d,AATSC4d,AATSC5d,AATSC6d,AATSC7d,AATSC8d,AATSC0s,AATSC1s,AATSC2s,AATSC3s,AATSC4s,AATSC5s,AATSC6s,AATSC7s,AATSC8s,AATSC0Z,AATSC1Z,AATSC2Z,AATSC3Z,AATSC4Z,AATSC5Z,AATSC6Z,AATSC7Z,AATSC8Z,AATSC0m,AATSC1m,AATSC2m,AATSC3m,AATSC4m,AATSC5m,AATSC6m,AATSC7m,AATSC8m,AATSC0v,AATSC1v,AATSC2v,AATSC3v,AATSC4v,AATSC5v,AATSC6v,AATSC7v,AATSC8v,AATSC0se,AATSC1se,AATSC2se,AATSC3se,AATSC4se,AATSC5se,AATSC6se,AATSC7se,AATSC8se,AATSC0pe,AATSC1pe,AATSC2pe,AATSC3pe,AATSC4pe,AATSC5pe,AATSC6pe,AATSC7pe,AATSC8pe,AATSC0are,AATSC1are,AATSC2are,AATSC3are,AATSC4are,AATSC5are,AATSC6are,AATSC7are,AATSC8are,AATSC0p,AATSC1p,AATSC2p,AATSC3p,AATSC4p,AATSC5p,AATSC6p,AATSC7p,AATSC8p,AATSC0i,AATSC1i,AATSC2i,AATSC3i,AATSC4i,AATSC5i,AATSC6i,AATSC7i,AATSC8i,MATS1c,MATS2c,MATS3c,MATS4c,MATS5c,MATS6c,MATS7c,MATS8c,MATS1dv,MATS2dv,MATS3dv,MATS4dv,MATS5dv,MATS6dv,MATS7dv,MATS8dv,MATS1d,MATS2d,MATS3d,MATS4d,MATS5d,MATS6d,MATS7d,MATS8d,MATS1s,MATS2s,MATS3s,MATS4s,MATS5s,MATS6s,MATS7s,MATS8s,MATS1Z,MATS2Z,MATS3Z,MATS4Z,MATS5Z,MATS6Z,MATS7Z,MATS8Z,MATS1m,MATS2m,MATS3m,MATS4m,MATS5m,MATS6m,MATS7m,MATS8m,MATS1v,MATS2v,MATS3v,MATS4v,MATS5v,MATS6v,MATS7v,MATS8v,MATS1se,MATS2se,MATS3se,MATS4se,MATS5se,MATS6se,MATS7se,MATS8se,MATS1pe,MATS2pe,MATS3pe,MATS4pe,MATS5pe,MATS6pe,MATS7pe,MATS8pe,MATS1are,MATS2are,MATS3are,MATS4are,MATS5are,MATS6are,MATS7are,MATS8are,MATS1p,MATS2p,MATS3p,MATS4p,MATS5p,MATS6p,MATS7p,MATS8p,MATS1i,MATS2i,MATS3i,MATS4i,MATS5i,MATS6i,MATS7i,MATS8i,GATS1c,GATS2c,GATS3c,GATS4c,GATS5c,GATS6c,GATS7c,GATS8c,GATS1dv,GATS2dv,GATS3dv,GATS4dv,GATS5dv,GATS6dv,GATS7dv,GATS8dv,GATS1d,GATS2d,GATS3d,GATS4d,GATS5d,GATS6d,GATS7d,GATS8d,GATS1s,GATS2s,GATS3s,GATS4s,GATS5s,GATS6s,GATS7s,GATS8s,GATS1Z,GATS2Z,GATS3Z,GATS4Z,GATS5Z,GATS6Z,GATS7Z,GATS8Z,GATS1m,GATS2m,GATS3m,GATS4m,GATS5m,GATS6m,GATS7m,GATS8m,GATS1v,GATS2v,GATS3v,GATS4v,GATS5v,GATS6v,GATS7v,GATS8v,GATS1se,GATS2se,GATS3se,GATS4se,GATS5se,GATS6se,GATS7se,GATS8se,GATS1pe,GATS2pe,GATS3pe,GATS4pe,GATS5pe,GATS6pe,GATS7pe,GATS8pe,GATS1are,GATS2are,GATS3are,GATS4are,GATS5are,GATS6are,GATS7are,GATS8are,GATS1p,GATS2p,GATS3p,GATS4p,GATS5p,GATS6p,GATS7p,GATS8p,GATS1i,GATS2i,GATS3i,GATS4i,GATS5i,GATS6i,GATS7i,GATS8i,BCUTc-1h,BCUTc-1l,BCUTdv-1h,BCUTdv-1l,BCUTd-1h,BCUTd-1l,BCUTs-1h,BCUTs-1l,BCUTZ-1h,BCUTZ-1l,BCUTm-1h,BCUTm-1l,BCUTv-1h,BCUTv-1l,BCUTse-1h,BCUTse-1l,BCUTpe-1h,BCUTpe-1l,BCUTare-1h,BCUTare-1l,BCUTp-1h,BCUTp-1l,BCUTi-1h,BCUTi-1l,BalabanJ,SpAbs_DzZ,SpMax_DzZ,SpDiam_DzZ,SpAD_DzZ,SpMAD_DzZ,LogEE_DzZ,SM1_DzZ,VE1_DzZ,VE2_DzZ,VE3_DzZ,VR1_DzZ,VR2_DzZ,VR3_DzZ,SpAbs_Dzm,SpMax_Dzm,SpDiam_Dzm,SpAD_Dzm,SpMAD_Dzm,LogEE_Dzm,SM1_Dzm,VE1_Dzm,VE2_Dzm,VE3_Dzm,VR1_Dzm,VR2_Dzm,VR3_Dzm,SpAbs_Dzv,SpMax_Dzv,SpDiam_Dzv,SpAD_Dzv,SpMAD_Dzv,LogEE_Dzv,SM1_Dzv,VE1_Dzv,VE2_Dzv,VE3_Dzv,VR1_Dzv,VR2_Dzv,VR3_Dzv,SpAbs_Dzse,SpMax_Dzse,SpDiam_Dzse,SpAD_Dzse,SpMAD_Dzse,LogEE_Dzse,SM1_Dzse,VE1_Dzse,VE2_Dzse,VE3_Dzse,VR1_Dzse,VR2_Dzse,VR3_Dzse,SpAbs_Dzpe,SpMax_Dzpe,SpDiam_Dzpe,SpAD_Dzpe,SpMAD_Dzpe,LogEE_Dzpe,SM1_Dzpe,VE1_Dzpe,VE2_Dzpe,VE3_Dzpe,VR1_Dzpe,VR2_Dzpe,VR3_Dzpe,SpAbs_Dzare,SpMax_Dzare,SpDiam_Dzare,SpAD_Dzare,SpMAD_Dzare,LogEE_Dzare,SM1_Dzare,VE1_Dzare,VE2_Dzare,VE3_Dzare,VR1_Dzare,VR2_Dzare,VR3_Dzare,SpAbs_Dzp,SpMax_Dzp,SpDiam_Dzp,SpAD_Dzp,SpMAD_Dzp,LogEE_Dzp,SM1_Dzp,VE1_Dzp,VE2_Dzp,VE3_Dzp,VR1_Dzp,VR2_Dzp,VR3_Dzp,SpAbs_Dzi,SpMax_Dzi,SpDiam_Dzi,SpAD_Dzi,SpMAD_Dzi,LogEE_Dzi,SM1_Dzi,VE1_Dzi,VE2_Dzi,VE3_Dzi,VR1_Dzi,VR2_Dzi,VR3_Dzi,BertzCT,nBonds,nBondsO,nBondsS,nBondsD,nBondsT,nBondsA,nBondsM,nBondsKS,nBondsKD,PNSA1,PNSA2,PNSA3,PNSA4,PNSA5,PPSA1,PPSA2,PPSA3,PPSA4,PPSA5,DPSA1,DPSA2,DPSA3,DPSA4,DPSA5,FNSA1,FNSA2,FNSA3,FNSA4,FNSA5,FPSA1,FPSA2,FPSA3,FPSA4,FPSA5,WNSA1,WNSA2,WNSA3,WNSA4,WNSA5,WPSA1,WPSA2,WPSA3,WPSA4,WPSA5,RNCG,RPCG,RNCS,RPCS,TASA,TPSA,RASA,RPSA,C1SP1,C2SP1,C1SP2,C2SP2,C3SP2,C1SP3,C2SP3,C3SP3,C4SP3,HybRatio,FCSP3,Xch-3d,Xch-4d,Xch-5d,Xch-6d,Xch-7d,Xch-3dv,Xch-4dv,Xch-5dv,Xch-6dv,Xch-7dv,Xc-3d,Xc-4d,Xc-5d,Xc-6d,Xc-3dv,Xc-4dv,Xc-5dv,Xc-6dv,Xpc-4d,Xpc-5d,Xpc-6d,Xpc-4dv,Xpc-5dv,Xpc-6dv,Xp-0d,Xp-1d,Xp-2d,Xp-3d,Xp-4d,Xp-5d,Xp-6d,Xp-7d,AXp-0d,AXp-1d,AXp-2d,AXp-3d,AXp-4d,AXp-5d,AXp-6d,AXp-7d,Xp-0dv,Xp-1dv,Xp-2dv,Xp-3dv,Xp-4dv,Xp-5dv,Xp-6dv,Xp-7dv,AXp-0dv,AXp-1dv,AXp-2dv,AXp-3dv,AXp-4dv,AXp-5dv,AXp-6dv,AXp-7dv,SZ,Sm,Sv,Sse,Spe,Sare,Sp,Si,MZ,Mm,Mv,Mse,Mpe,Mare,Mp,Mi,SpAbs_Dt,SpMax_Dt,SpDiam_Dt,SpAD_Dt,SpMAD_Dt,LogEE_Dt,SM1_Dt,VE1_Dt,VE2_Dt,VE3_Dt,VR1_Dt,VR2_Dt,VR3_Dt,DetourIndex,SpAbs_D,SpMax_D,SpDiam_D,SpAD_D,SpMAD_D,LogEE_D,VE1_D,VE2_D,VE3_D,VR1_D,VR2_D,VR3_D,NsLi,NssBe,NssssBe,NssBH,NsssB,NssssB,NsCH3,NdCH2,NssCH2,NtCH,NdsCH,NaaCH,NsssCH,NddC,NtsC,NdssC,NaasC,NaaaC,NssssC,NsNH3,NsNH2,NssNH2,NdNH,NssNH,NaaNH,NtN,NsssNH,NdsN,NaaN,NsssN,NddsN,NaasN,NssssN,NsOH,NdO,NssO,NaaO,NsF,NsSiH3,NssSiH2,NsssSiH,NssssSi,NsPH2,NssPH,NsssP,NdsssP,NsssssP,NsSH,NdS,NssS,NaaS,NdssS,NddssS,NsCl,NsGeH3,NssGeH2,NsssGeH,NssssGe,NsAsH2,NssAsH,NsssAs,NsssdAs,NsssssAs,NsSeH,NdSe,NssSe,NaaSe,NdssSe,NddssSe,NsBr,NsSnH3,NssSnH2,NsssSnH,NssssSn,NsI,NsPbH3,NssPbH2,NsssPbH,NssssPb,SsLi,SssBe,SssssBe,SssBH,SsssB,SssssB,SsCH3,SdCH2,SssCH2,StCH,SdsCH,SaaCH,SsssCH,SddC,StsC,SdssC,SaasC,SaaaC,SssssC,SsNH3,SsNH2,SssNH2,SdNH,SssNH,SaaNH,StN,SsssNH,SdsN,SaaN,SsssN,SddsN,SaasN,SssssN,SsOH,SdO,SssO,SaaO,SsF,SsSiH3,SssSiH2,SsssSiH,SssssSi,SsPH2,SssPH,SsssP,SdsssP,SsssssP,SsSH,SdS,SssS,SaaS,SdssS,SddssS,SsCl,SsGeH3,SssGeH2,SsssGeH,SssssGe,SsAsH2,SssAsH,SsssAs,SsssdAs,SsssssAs,SsSeH,SdSe,SssSe,SaaSe,SdssSe,SddssSe,SsBr,SsSnH3,SssSnH2,SsssSnH,SssssSn,SsI,SsPbH3,SssPbH2,SsssPbH,SssssPb,MAXsLi,MAXssBe,MAXssssBe,MAXssBH,MAXsssB,MAXssssB,MAXsCH3,MAXdCH2,MAXssCH2,MAXtCH,MAXdsCH,MAXaaCH,MAXsssCH,MAXddC,MAXtsC,MAXdssC,MAXaasC,MAXaaaC,MAXssssC,MAXsNH3,MAXsNH2,MAXssNH2,MAXdNH,MAXssNH,MAXaaNH,MAXtN,MAXsssNH,MAXdsN,MAXaaN,MAXsssN,MAXddsN,MAXaasN,MAXssssN,MAXsOH,MAXdO,MAXssO,MAXaaO,MAXsF,MAXsSiH3,MAXssSiH2,MAXsssSiH,MAXssssSi,MAXsPH2,MAXssPH,MAXsssP,MAXdsssP,MAXsssssP,MAXsSH,MAXdS,MAXssS,MAXaaS,MAXdssS,MAXddssS,MAXsCl,MAXsGeH3,MAXssGeH2,MAXsssGeH,MAXssssGe,MAXsAsH2,MAXssAsH,MAXsssAs,MAXsssdAs,MAXsssssAs,MAXsSeH,MAXdSe,MAXssSe,MAXaaSe,MAXdssSe,MAXddssSe,MAXsBr,MAXsSnH3,MAXssSnH2,MAXsssSnH,MAXssssSn,MAXsI,MAXsPbH3,MAXssPbH2,MAXsssPbH,MAXssssPb,MINsLi,MINssBe,MINssssBe,MINssBH,MINsssB,MINssssB,MINsCH3,MINdCH2,MINssCH2,MINtCH,MINdsCH,MINaaCH,MINsssCH,MINddC,MINtsC,MINdssC,MINaasC,MINaaaC,MINssssC,MINsNH3,MINsNH2,MINssNH2,MINdNH,MINssNH,MINaaNH,MINtN,MINsssNH,MINdsN,MINaaN,MINsssN,MINddsN,MINaasN,MINssssN,MINsOH,MINdO,MINssO,MINaaO,MINsF,MINsSiH3,MINssSiH2,MINsssSiH,MINssssSi,MINsPH2,MINssPH,MINsssP,MINdsssP,MINsssssP,MINsSH,MINdS,MINssS,MINaaS,MINdssS,MINddssS,MINsCl,MINsGeH3,MINssGeH2,MINsssGeH,MINssssGe,MINsAsH2,MINssAsH,MINsssAs,MINsssdAs,MINsssssAs,MINsSeH,MINdSe,MINssSe,MINaaSe,MINdssSe,MINddssSe,MINsBr,MINsSnH3,MINssSnH2,MINsssSnH,MINssssSn,MINsI,MINsPbH3,MINssPbH2,MINsssPbH,MINssssPb,ECIndex,ETA_alpha,AETA_alpha,ETA_shape_p,ETA_shape_y,ETA_shape_x,ETA_beta,AETA_beta,ETA_beta_s,AETA_beta_s,ETA_beta_ns,AETA_beta_ns,ETA_beta_ns_d,AETA_beta_ns_d,ETA_eta,AETA_eta,ETA_eta_L,AETA_eta_L,ETA_eta_R,AETA_eta_R,ETA_eta_RL,AETA_eta_RL,ETA_eta_F,AETA_eta_F,ETA_eta_FL,AETA_eta_FL,ETA_eta_B,AETA_eta_B,ETA_eta_BR,AETA_eta_BR,ETA_dAlpha_A,ETA_dAlpha_B,ETA_epsilon_1,ETA_epsilon_2,ETA_epsilon_3,ETA_epsilon_4,ETA_epsilon_5,ETA_dEpsilon_A,ETA_dEpsilon_B,ETA_dEpsilon_C,ETA_dEpsilon_D,ETA_dBeta,AETA_dBeta,ETA_psi_1,ETA_dPsi_A,ETA_dPsi_B,fragCpx,fMF,GeomDiameter,GeomRadius,GeomShapeIndex,GeomPetitjeanIndex,GRAV,GRAVH,GRAVp,GRAVHp,nHBAcc,nHBDon,IC0,IC1,IC2,IC3,IC4,IC5,TIC0,TIC1,TIC2,TIC3,TIC4,TIC5,SIC0,SIC1,SIC2,SIC3,SIC4,SIC5,BIC0,BIC1,BIC2,BIC3,BIC4,BIC5,CIC0,CIC1,CIC2,CIC3,CIC4,CIC5,MIC0,MIC1,MIC2,MIC3,MIC4,MIC5,ZMIC0,ZMIC1,ZMIC2,ZMIC3,ZMIC4,ZMIC5,Kier1,Kier2,Kier3,Lipinski,GhoseFilter,FilterItLogS,VMcGowan,Mor01,Mor02,Mor03,Mor04,Mor05,Mor06,Mor07,Mor08,Mor09,Mor10,Mor11,Mor12,Mor13,Mor14,Mor15,Mor16,Mor17,Mor18,Mor19,Mor20,Mor21,Mor22,Mor23,Mor24,Mor25,Mor26,Mor27,Mor28,Mor29,Mor30,Mor31,Mor32,Mor01m,Mor02m,Mor03m,Mor04m,Mor05m,Mor06m,Mor07m,Mor08m,Mor09m,Mor10m,Mor11m,Mor12m,Mor13m,Mor14m,Mor15m,Mor16m,Mor17m,Mor18m,Mor19m,Mor20m,Mor21m,Mor22m,Mor23m,Mor24m,Mor25m,Mor26m,Mor27m,Mor28m,Mor29m,Mor30m,Mor31m,Mor32m,Mor01v,Mor02v,Mor03v,Mor04v,Mor05v,Mor06v,Mor07v,Mor08v,Mor09v,Mor10v,Mor11v,Mor12v,Mor13v,Mor14v,Mor15v,Mor16v,Mor17v,Mor18v,Mor19v,Mor20v,Mor21v,Mor22v,Mor23v,Mor24v,Mor25v,Mor26v,Mor27v,Mor28v,Mor29v,Mor30v,Mor31v,Mor32v,Mor01se,Mor02se,Mor03se,Mor04se,Mor05se,Mor06se,Mor07se,Mor08se,Mor09se,Mor10se,Mor11se,Mor12se,Mor13se,Mor14se,Mor15se,Mor16se,Mor17se,Mor18se,Mor19se,Mor20se,Mor21se,Mor22se,Mor23se,Mor24se,Mor25se,Mor26se,Mor27se,Mor28se,Mor29se,Mor30se,Mor31se,Mor32se,Mor01p,Mor02p,Mor03p,Mor04p,Mor05p,Mor06p,Mor07p,Mor08p,Mor09p,Mor10p,Mor11p,Mor12p,Mor13p,Mor14p,Mor15p,Mor16p,Mor17p,Mor18p,Mor19p,Mor20p,Mor21p,Mor22p,Mor23p,Mor24p,Mor25p,Mor26p,Mor27p,Mor28p,Mor29p,Mor30p,Mor31p,Mor32p,LabuteASA,PEOE_VSA1,PEOE_VSA2,PEOE_VSA3,PEOE_VSA4,PEOE_VSA5,PEOE_VSA6,PEOE_VSA7,PEOE_VSA8,PEOE_VSA9,PEOE_VSA10,PEOE_VSA11,PEOE_VSA12,PEOE_VSA13,SMR_VSA1,SMR_VSA2,SMR_VSA3,SMR_VSA4,SMR_VSA5,SMR_VSA6,SMR_VSA7,SMR_VSA8,SMR_VSA9,SlogP_VSA1,SlogP_VSA2,SlogP_VSA3,SlogP_VSA4,SlogP_VSA5,SlogP_VSA6,SlogP_VSA7,SlogP_VSA8,SlogP_VSA9,SlogP_VSA10,SlogP_VSA11,EState_VSA1,EState_VSA2,EState_VSA3,EState_VSA4,EState_VSA5,EState_VSA6,EState_VSA7,EState_VSA8,EState_VSA9,EState_VSA10,VSA_EState1,VSA_EState2,VSA_EState3,VSA_EState4,VSA_EState5,VSA_EState6,VSA_EState7,VSA_EState8,VSA_EState9,MDEC-11,MDEC-12,MDEC-13,MDEC-14,MDEC-22,MDEC-23,MDEC-24,MDEC-33,MDEC-34,MDEC-44,MDEO-11,MDEO-12,MDEO-22,MDEN-11,MDEN-12,MDEN-13,MDEN-22,MDEN-23,MDEN-33,MID,AMID,MID_h,AMID_h,MID_C,AMID_C,MID_N,AMID_N,MID_O,AMID_O,MID_X,AMID_X,MOMI-X,MOMI-Y,MOMI-Z,PBF,MPC2,MPC3,MPC4,MPC5,MPC6,MPC7,MPC8,MPC9,MPC10,TMPC10,piPC1,piPC2,piPC3,piPC4,piPC5,piPC6,piPC7,piPC8,piPC9,piPC10,TpiPC10,apol,bpol,nRing,n3Ring,n4Ring,n5Ring,n6Ring,n7Ring,n8Ring,n9Ring,n10Ring,n11Ring,n12Ring,nG12Ring,nHRing,n3HRing,n4HRing,n5HRing,n6HRing,n7HRing,n8HRing,n9HRing,n10HRing,n11HRing,n12HRing,nG12HRing,naRing,n3aRing,n4aRing,n5aRing,n6aRing,n7aRing,n8aRing,n9aRing,n10aRing,n11aRing,n12aRing,nG12aRing,naHRing,n3aHRing,n4aHRing,n5aHRing,n6aHRing,n7aHRing,n8aHRing,n9aHRing,n10aHRing,n11aHRing,n12aHRing,nG12aHRing,nARing,n3ARing,n4ARing,n5ARing,n6ARing,n7ARing,n8ARing,n9ARing,n10ARing,n11ARing,n12ARing,nG12ARing,nAHRing,n3AHRing,n4AHRing,n5AHRing,n6AHRing,n7AHRing,n8AHRing,n9AHRing,n10AHRing,n11AHRing,n12AHRing,nG12AHRing,nFRing,n4FRing,n5FRing,n6FRing,n7FRing,n8FRing,n9FRing,n10FRing,n11FRing,n12FRing,nG12FRing,nFHRing,n4FHRing,n5FHRing,n6FHRing,n7FHRing,n8FHRing,n9FHRing,n10FHRing,n11FHRing,n12FHRing,nG12FHRing,nFaRing,n4FaRing,n5FaRing,n6FaRing,n7FaRing,n8FaRing,n9FaRing,n10FaRing,n11FaRing,n12FaRing,nG12FaRing,nFaHRing,n4FaHRing,n5FaHRing,n6FaHRing,n7FaHRing,n8FaHRing,n9FaHRing,n10FaHRing,n11FaHRing,n12FaHRing,nG12FaHRing,nFARing,n4FARing,n5FARing,n6FARing,n7FARing,n8FARing,n9FARing,n10FARing,n11FARing,n12FARing,nG12FARing,nFAHRing,n4FAHRing,n5FAHRing,n6FAHRing,n7FAHRing,n8FAHRing,n9FAHRing,n10FAHRing,n11FAHRing,n12FAHRing,nG12FAHRing,nRot,RotRatio,SLogP,SMR,TopoPSA(NO),TopoPSA,GGI1,GGI2,GGI3,GGI4,GGI5,GGI6,GGI7,GGI8,GGI9,GGI10,JGI1,JGI2,JGI3,JGI4,JGI5,JGI6,JGI7,JGI8,JGI9,JGI10,JGT10,Diameter,Radius,TopoShapeIndex,PetitjeanIndex,Vabc,VAdjMat,MWC01,MWC02,MWC03,MWC04,MWC05,MWC06,MWC07,MWC08,MWC09,MWC10,TMWC10,SRW02,SRW03,SRW04,SRW05,SRW06,SRW07,SRW08,SRW09,SRW10,TSRW10,MW,AMW,WPath,WPol,Zagreb1,Zagreb2,mZagreb1,mZagreb2".split(",")
    num_descriptors = len(descriptor_names)

    @staticmethod
    def generate_descriptors(normalize: tuple = (-1, 1), monomer_dict_: dict = None):
        """
        Mordred не завёлся локально, но нормально заработал в google.colab, поэтому просто сгенерил дескрипторы там.
        Дескрипторы уже нормализованы на (-1, 1). Код:

                from rdkit import Chem
                from mordred import Calculator, descriptors

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

    @classmethod
    def encode_sequences_from_unique(cls, sequences_list, max_length=27, normalize=True):

        unique_desc = load_unique_descriptors(descriptors=cls)

        # Create empty array where monomer descriptors will be kept
        x_senses = np.empty((0, max_length * cls.num_descriptors), float)

        for i, seq in enumerate(sequences_list, 1):
            descriptors_set = np.empty((0, cls.num_descriptors), float)
            print(f"{i}.", end=' ')
            for smiles in seq:
                try:
                    descriptors = unique_desc[smiles]
                    descriptors = np.array(descriptors)
                except Exception as ex:
                    print(ex)
                    print(smiles, end='\n\n')
                    break
                descriptors = descriptors.reshape((-1, cls.num_descriptors))
                descriptors_set = np.append(descriptors_set, descriptors, axis=0)
            print("\033[92m" + "done" + "\033[0m")
            if normalize is True:
                sc = MinMaxScaler(feature_range=(-1, 1))
                scaled_array = sc.fit_transform(descriptors_set)
            else:
                scaled_array = descriptors_set

            scaled_df = pd.DataFrame(scaled_array, columns=cls.descriptor_names)
            while scaled_df.shape[0] < max_length:
                scaled_df.loc[scaled_df.shape[0]] = [0] * cls.num_descriptors

            array = np.array(scaled_df)
            array = array.flatten()
            x_senses = np.vstack((x_senses, array))
        return x_senses


class Chemopy:
    descriptor_names = "RDFC6,RDFM19,RDFU9,RDFU3,RDFU6,RDFU4,Harary3D,RDFC5,MoRSEM6,MoRSEM4,MoRSEM2,MoRSEE30,MoRSEM8,MoRSEU10,MoRSEU12,MoRSEU14,MoRSEU16,MoRSEU18,FPSA3,FPSA1,MoRSEV19,MoRSEV13,MoRSEV11,MoRSEV17,MoRSEV15,RDFM14,RDFE8,RDFM10,RDFM12,RDFE2,MoRSEC25,RDFE6,RDFE4,WNSA1,WNSA3,L2p,RDFP14,RDFP16,RDFP10,RDFP12,RDFP19,Dm,MoRSEM22,MoRSEM18,RDFP26,MoRSEM10,MoRSEM12,MoRSEM14,MoRSEM16,RDFC27,RDFC25,RDFC23,RDFC21,MoRSEM25,RDFC29,MoRSEU30,L1v,RDFP5,RDFP7,RDFP1,RDFP2,RDFP9,MoRSEP5,L1m,P1m,MoRSEP6,P1u,RDFE14,RDFE16,PPSA1,PPSA3,MoRSEP11,E2v,E2p,MoRSEP17,DPSA1,MoRSEN5,MoRSEN3,MoRSEN1,E2e,E2m,RDFM28,L3m,MoRSEP8,RDFC16,RDFV30,MoRSEN16,RDFC14,RDFC15,P3m,P3e,P3p,P3u,RDFC18,FNSA1,FNSA3,RDFC12,RDFC10,P2p,RDFM6,RDFU21,RDFU23,MoRSEN22,RDFU27,RDFU29,MoRSEC20,MoRSEN29,RDFV18,RDFV16,RDFV14,RDFV12,RDFE19,RDFM9,MoRSEN24,RDFV9,MoRSEE9,RDFU25,RDFV8,MoRSEE3,Ve,RDFE10,Vm,MoRSEV18,Vv,RDFU13,RDFC30,MoRSEP15,WPSA2,MoRSEU8,MoRSEU6,MoRSEU4,MoRSEC30,MoRSEU1,MoRSEP19,RDFM21,RDFM23,RDFM25,RDFM27,RDFV7,RDFV5,RDFV3,RDFM4,MoRSEP20,MoRSEP22,MoRSEP24,MoRSEP26,MoRSEC8,RDFM17,AGDD,MoRSEC2,MoRSEC4,MoRSEC6,MoRSEE29,WPSA1,MoRSEE21,MoRSEE23,MoRSEE25,MoRSEE27,MoRSEC27,RDFE12,Ae,MoRSEP7,MoRSEC22,MoRSEV28,MoRSEV26,MoRSEV24,MoRSEV22,MoRSEV20,P2e,MoRSEC10,MoRSEC16,MoRSEC14,P2m,MoRSEC19,P2u,RDFE23,RDFE25,RDFE27,RDFE29,RDFP20,RDFP22,MoRSEP1,RPCS,RDFP28,grav,MoRSEN20,RDFV26,RDFV24,RDFV22,RDFV20,L2e,RDFV28,MoRSEM27,RDFU16,MoRSEP2,MoRSEE5,MoRSEV3,E3v,Ke,RDFV6,MoRSEV7,Kp,Ku,MoRSEP4,E3m,RDFU19,MoRSEM20,MoRSEV4,MoRSEM24,MoRSEN14,MoRSEM28,MoRSEN18,MoRSEV9,MoRSEU29,L2u,MoRSEN12,MoRSEU21,MoRSEU23,MoRSEU25,MoRSEU27,RDFU17,Tv,SEig,Tp,Tm,PNSA3,PNSA1,MSA,Ap,MoRSEE10,MoRSEE12,MoRSEP9,MoRSEE16,MoRSEE18,Du,MoRSEN8,E1p,E1v,E1m,MoRSEP28,E1e,W3D,RDFC9,RDFU15,RDFC7,E3e,RDFC3,Au,MoRSEP29,RDFM30,MoRSEN6".split(",")
    num_descriptors = len(descriptor_names)

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

    @classmethod
    def encode_sequences_from_unique(cls, sequences_list, max_length=27, normalize=True):

        unique_desc = load_unique_descriptors(descriptors=cls)

        # Create empty array where monomer descriptors will be kept
        x_senses = np.empty((0, max_length * cls.num_descriptors), float)

        for i, seq in enumerate(sequences_list, 1):
            descriptors_set = np.empty((0, cls.num_descriptors), float)
            print(f"{i}.", end=' ')
            for smiles in seq:
                try:
                    descriptors = unique_desc[smiles]
                    descriptors = np.array(descriptors)
                except Exception as ex:
                    print(ex)
                    print(smiles, end='\n\n')
                    break
                descriptors = descriptors.reshape((-1, cls.num_descriptors))
                descriptors_set = np.append(descriptors_set, descriptors, axis=0)
            print("\033[92m" + "done" + "\033[0m")
            if normalize is True:
                sc = MinMaxScaler(feature_range=(-1, 1))
                scaled_array = sc.fit_transform(descriptors_set)
            else:
                scaled_array = descriptors_set

            scaled_df = pd.DataFrame(scaled_array, columns=cls.descriptor_names)
            while scaled_df.shape[0] < max_length:
                scaled_df.loc[scaled_df.shape[0]] = [0] * cls.num_descriptors

            array = np.array(scaled_df)
            array = array.flatten()
            x_senses = np.vstack((x_senses, array))
        return x_senses


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
