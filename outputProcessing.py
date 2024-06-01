import os
import pandas as pd
import numpy as np

def extractInteractionInfo(info_file):    
    interaction_info = {}
    interaction_ddi_type_info = {}
    with open(info_file, 'r') as fp:
        next(fp)  # Skip header
        for line in fp:
            interaction_type, sentence, subject, new_interaction_type = line.strip().split(',')
            interaction_info[interaction_type] = [subject, sentence, new_interaction_type]
            interaction_ddi_type_info[new_interaction_type] = interaction_type
    return interaction_info, interaction_ddi_type_info


def parseDrugInformation(drug_info_file):
    drug_information = {}
    with open(drug_info_file, 'r') as fp:
        for line in fp:
            drugbank_id, drugbank_name, _, _, _, target, _, action, pharmacological_action = line.strip().split('\t')
            if action != 'None' and pharmacological_action == 'yes':
                if drugbank_id not in drug_information:
                    drug_information[drugbank_id] = [target]
                else:
                    drug_information[drugbank_id].append(target)
    return drug_information


def readDDIInformation(ddi_file):
    left_ddi_info = {}
    right_ddi_info = {}
    with open(ddi_file, 'r') as fp:
        next(fp)  # Skip header
        for line in fp:
            left_drug, right_drug, interaction_type = line.strip().split('\t')
            left_ddi_info.setdefault(interaction_type, []).append(left_drug)
            right_ddi_info.setdefault(interaction_type, []).append(right_drug)
    
    for interaction_type in left_ddi_info:
        left_ddi_info[interaction_type] = list(set(left_ddi_info[interaction_type]))
    
    for interaction_type in right_ddi_info:
        right_ddi_info[interaction_type] = list(set(right_ddi_info[interaction_type]))
        
    return left_ddi_info, right_ddi_info


def parseSimilarityFile(similarity_file):
    similarity_info = {}
    similarity_df = pd.read_csv(similarity_file, index_col=0)
    similarity_info = similarity_df.to_dict()

    return similarity_df, similarity_info


def annotateDrugs(DDI_output_file, drug_info_file, similarity_file, ddi_file, output_file, info_file, threshold):
    drug_info = parseDrugInformation(drug_info_file)    
    left_ddi_info, right_ddi_info = readDDIInformation(ddi_file)    
    similarity_df, similarity_info = parseSimilarityFile(similarity_file)
    DDI_prediction_df = pd.read_table(DDI_output_file)
    sentence_interaction_info, interaction_ddi_type_info = extractInteractionInfo(info_file)
    
    with open(output_file, 'w') as fp:
        fp.write('%s\t%s\t%s\t%s\t%s\t%s\n' % ('Drug pair', 'Interaction type', 'Sentence', 'Score', 'Similar approved drugs (left)', 'Similar approved drugs (right)'))
        
        for _, row in DDI_prediction_df.iterrows():
            drug_pair = row['Drug pair']
            left_drug, right_drug = drug_pair.split('_')
            new_DDI_type = row['DDI type']
            sentence = row['Sentence']
            score = row['Score']
            
            left_corresponding_drugs = left_ddi_info[new_DDI_type]
            right_corresponding_drugs = right_ddi_info[new_DDI_type]

            left_drug_similarity_df = similarity_df.loc[left_drug, left_corresponding_drugs]
            left_selected_drugs = list(left_drug_similarity_df[left_drug_similarity_df >= threshold].index)
            
            right_drug_similarity_df = similarity_df.loc[right_drug, right_corresponding_drugs]
            right_selected_drugs = list(right_drug_similarity_df[right_drug_similarity_df >= threshold].index)

            left_drug_annotation_list = []
            for each_drug in left_selected_drugs:
                if each_drug in drug_info:
                    targets = drug_info[each_drug]
                    drug_target_information = '%s(%s)' % (each_drug, '|'.join(targets))
                    left_drug_annotation_list.append(drug_target_information)

            right_drug_annotation_list = []
            for each_drug in right_selected_drugs:
                if each_drug in drug_info:
                    targets = drug_info[each_drug]
                    drug_target_information = '%s(%s)' % (each_drug, '|'.join(targets))
                    right_drug_annotation_list.append(drug_target_information)

            left_drug_annotation_string = ';'.join(left_drug_annotation_list)
            right_drug_annotation_string = ';'.join(right_drug_annotation_list)        
            
            fp.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (drug_pair, new_DDI_type, sentence, score, left_drug_annotation_string, right_drug_annotation_string))


def summarizeResults(result_file, output_file, info_file):    
    sentence_interaction_info, interaction_ddi_type_info = extractInteractionInfo(info_file)
    
    with open(result_file, 'r') as fp, open(output_file, 'w') as out_fp:
        out_fp.write('%s\t%s\t%s\t%s\n' % ('Drug pair', 'DDI type', 'Sentence', 'Score'))
        
        for line in fp:
            drug_pair, DDI_class, predicted_score = line.strip().split('\t')[:3]
            new_interaction_type = sentence_interaction_info[DDI_class][2]
            
            if sentence_interaction_info[DDI_class][0] == '2':
                drug1, drug2 = drug_pair.split('_')
            else:
                drug1, drug2 = drug_pair.split('_')

            template_sentence = sentence_interaction_info[DDI_class][1]
            prediction_outcome = template_sentence.replace('#Drug1', drug1)
            prediction_outcome = prediction_outcome.replace('#Drug2', drug2)
            
            out_fp.write('%s\t%s\t%s\t%s\n' % (drug_pair, new_interaction_type, prediction_outcome, predicted_score))




