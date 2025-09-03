matrices = generate_all_confusion_matrices()
# Individual figures for detailed analysis
for arch_name, cm in matrices.items():
    fig = create_confusion_matrix_figure(cm, arch_name, 
                                        save_path=f'supp_{arch_name}.png')

# Combined figure for paper
combined_fig = create_combined_figure(matrices, 
                                     save_path='supplementary_confusion_matrices.png')
export_matrices_to_csv(matrices, output_dir='./supplementary_data/')
### Section S1: Complete Confusion Matrices

# Figure S1 presents the full 7×7 confusion matrices for all evaluated architectures. 
# Each matrix shows the percentage of predictions for each true emotion class.

### Key Observations:
# 1. **Systematic confusion patterns**: Fear-Surprise confusion is consistent across all architectures (12-18%)
# 2. **Architecture correlation**: Better overall accuracy correlates with reduced off-diagonal values
# 3. **Emotion-specific challenges**: Disgust shows highest confusion rates with Angry (20-28%)

### Data Files:
# - `EfficientNet-B0_confusion_matrix.csv`
# - `VGG16_confusion_matrix.csv`
# - `ResNet50_confusion_matrix.csv`
# - `MobileNetV2_confusion_matrix.csv`
#- `AlexNet_confusion_matrix.csv`

# These matrices provide detailed insights into each architecture's emotion 
# discrimination capabilities, supporting the CEDI analysis in the main paper.
