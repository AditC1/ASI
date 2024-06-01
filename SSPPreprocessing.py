import os
import glob
import pickle
import pandas as pd
import click

from rdkit import Chem
from rdkit.Chem import AllChem

@click.command()
@click.option('--drug_dir', type=str, help='Directory containing drug files.')
@click.option('--input_dir', type=str, help='Directory containing input files.')
@click.option('--output_file', type=str, help='Output file path for drug similarity data.')

def drug_similarity_calculation(drug_dir, input_dir, output_file):
    drug_files = glob.glob(drug_dir + '*')
    input_files = glob.glob(input_dir + '*')
    similarity_data = {}
    for drug_file in drug_files:
        drug_id = os.path.basename(drug_file).split('.')[0]
        similarity_data[drug_id] = {}
        drug_mol = Chem.MolFromMolFile(drug_file)
        drug_mol = AllChem.AddHs(drug_mol)
        for input_file in input_files:
            input_id = os.path.basename(input_file).split('.')[0]
            input_mol = Chem.MolFromMolFile(input_file)
            input_mol = AllChem.AddHs(input_mol)
            fps = AllChem.GetMorganFingerprint(drug_mol, 2)
            fps2 = AllChem.GetMorganFingerprint(input_mol, 2)
            score = sum(fps) / sum(fps2)
            similarity_data[drug_id][input_id] = score

    df = pd.DataFrame.from_dict(similarity_data)
    df.to_csv(output_file)

@click.command()
@click.option('--drug_dir', type=str, help='Directory containing drug files.')
@click.option('--input_file', type=str, help='Input file path containing drug data.')
@click.option('--output_file', type=str, help='Output file path for structure similarity data.')
def structure_similarity_calculation(drug_dir, input_file, output_file):
    drug_files = glob.glob(drug_dir + '*')
    input_data = {}
    with open(input_file, 'r') as fp:
        for line in fp:
            drug1, smiles1, drug2, smiles2 = line.strip().split('\t')[:4]
            if drug1 not in input_data:
                input_data[drug1] = smiles1
            if drug2 not in input_data:
                input_data[drug2] = smiles2

    similarity_data = {}
    for input_drug, smiles in input_data.items():
        try:
            drug_mol = Chem.MolFromSmiles(smiles)
            drug_mol = AllChem.AddHs(drug_mol)
        except:
            continue
        similarity_data[input_drug] = {}
        for drug_file in drug_files:
            drug_id = os.path.basename(drug_file).split('.')[0]
            drug_mol = Chem.MolFromMolFile(drug_file)
            drug_mol = AllChem.AddHs(drug_mol)
            fps = AllChem.GetMorganFingerprint(drug_mol, 2)
            fps2 = AllChem.GetMorganFingerprint(drug_mol, 2)
            score = sum(fps) / sum(fps2)
            similarity_data[input_drug][drug_id] = score

    df = pd.DataFrame.from_dict(similarity_data)
    df.T.to_csv(output_file)

@click.command()
@click.option('--similarity_profile_file', type=str, help='Input file path for similarity profile.')
@click.option('--output_file', type=str, help='Output file path for PCA transformation data.')
@click.option('--pca_model', type=str, help='Pickle file containing PCA model.')
def pca_calculation(similarity_profile_file, output_file, pca_model):
    with open(pca_model, 'rb') as fid:
        pca = pickle.load(fid)
        df = pd.read_csv(similarity_profile_file, index_col=0)

        X = df.values
        X = pca.transform(X)

        new_df = pd.DataFrame(X, columns=['PC_%d' % (i + 1) for i in range(50)], index=df.index)
        new_df.to_csv(output_file)

@click.command()
@click.option('--input_file', type=str, help='Input file path containing drug data.')
@click.option('--pca_profile_file', type=str, help='Input file path for PCA profile.')
@click.option('--output_file', type=str, help='Output file path for input profile.')
def generate_input_profile(input_file, pca_profile_file, output_file):
    df = pd.read_csv(pca_profile_file, index_col=0)
    drug_info = dict(df.T)
    
    interaction_list = []
    with open(input_file, 'r') as fp:
        for line in fp:
            drug1, _, drug2, _ = line.strip().split('\t')[:4]
            if drug1 in df.index and drug2 in df.index:
                interaction_list.append([drug1, drug2])
                interaction_list.append([drug2, drug1])
    
    columns = ['PC_%d' % (i + 1) for i in range(50)]
    DDI_input = {}
    for drug_pair in interaction_list:
        drug1, drug2 = drug_pair
        key = f'{drug1}_{drug2}'
        
        DDI_input[key] = {}
        
        for col in columns:
            new_key1 = f'1_{col}'
            new_key2 = f'2_{col}'
            DDI_input[key][new_key1] = drug_info[drug1][col]
            DDI_input[key][new_key2] = drug_info[drug2][col]
    
    new_columns = [f'{i}_PC_{j}' for i in [1, 2] for j in range(1, 51)]
    df = pd.DataFrame.from_dict(DDI_input)
    df = df.T
    df = df[new_columns]
    df.to_csv(output_file)

if __name__ == '__main__':
    drug_similarity_calculation()
    structure_similarity_calculation()
    pca_calculation()
    generate_input_profile()
