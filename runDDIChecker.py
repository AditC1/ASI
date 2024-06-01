import click
import os
import shutil
import logging
import time

@click.command()
@click.option('-o', '--output_dir', required=True, help="Output directory")
@click.option('-i', '--input_file', required=True, help="Input file")
@click.option('-p', '--PCA_profile_file', help="PCA profile file")
def main(output_dir, input_file, pca_profile_file):
    start = time.time()
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    drug_dir = './data/DrugBank5.0_Approved_drugs/'
    trained_model = 'model.h5'
    
    multiclass_trained_model = './data/Multiclass_weight.ckpt'
    binaryclass_trained_model = './data/Binary_weight.ckpt'
    DDI_sentence_information_file = './data/Interaction_information.csv'
    
    known_DDI_file = './data/DrugBank_known_ddi.txt'
    drug_information_file = './data/Approved_drug_Information.txt'
    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    ddi_input_file = os.path.join(output_dir, 'tanimoto_PCA50_DDI_Input.csv')
    output_file = os.path.join(output_dir, 'DDI_result.txt')
    ddi_output_file = os.path.join(output_dir, 'Final_DDI_result.txt')
    annotation_output_file = os.path.join(output_dir, 'Final_annotated_DDI_result.txt')
    known_drug_similarity_file = './data/drug_similarity.csv'
    
    if pca_profile_file is None:
        similarity_profile = os.path.join(output_dir, 'similarity_profile.csv')
        pca_similarity_profile = os.path.join(output_dir, 'PCA_transformed_similarity_profile.csv')
        pca_profile_file = os.path.join(output_dir, 'PCA_transformed_similarity_profile.csv')
        print('calculate structure similarity profile')
        preprocessing.calculate_structure_similarity(drug_dir, input_file, similarity_profile)
        preprocessing.calculate_pca(similarity_profile, pca_similarity_profile, pca_model)
        print('combine structural similarity profile')
        preprocessing.generate_input_profile(input_file, pca_similarity_profile, ddi_input_file)
    else:
        pca_similarity_profile = pca_profile_file
        print('generate input profile')
        preprocessing.generate_input_profile(input_file, pca_similarity_profile, ddi_input_file)
    
    threshold = 0.5
    model.predict_DDI(output_dir, output_file, ddi_input_file, trained_weight_model, threshold)    
    result_processing.summarize_prediction_outcome(output_file, ddi_output_file, DDI_sentence_information_file)
    result_processing.annotate_similar_drugs(ddi_output_file, drug_information_file, similarity_profile, known_DDI_file, annotation_output_file, DDI_sentence_information_file, 0.75)
    
    logging.info(time.strftime("Elapsed time %H:%M:%S", time.gmtime(time.time() - start)))

if __name__ == '__main__':
    main()
