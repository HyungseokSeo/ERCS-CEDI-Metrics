matrices = generate_all_confusion_matrices()
# Individual figures for detailed analysis
for arch_name, cm in matrices.items():
    fig = create_confusion_matrix_figure(cm, arch_name, 
                                        save_path=f'supp_{arch_name}.png')

# Combined figure for paper
combined_fig = create_combined_figure(matrices, 
                                     save_path='supplementary_confusion_matrices.png')
export_matrices_to_csv(matrices, output_dir='./supplementary_data/') 
