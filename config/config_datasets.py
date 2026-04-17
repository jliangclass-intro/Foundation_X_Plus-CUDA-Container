import os
import yaml

DATASET_DIR = "/scratch/sejong/class-dataset/" ## For SOL
# DATASET_DIR = "/scratch/nuislam/Data/" ## For PHX

DATASETS_CONFIG = {
    # NIH ChestX-ray14 - Classification - Localization
    "cls_nih_root": DATASET_DIR + "cxr-256/nih_xray14/",
    "cls_nih_trainList": DATASET_DIR + "cxr-256/nih_xray14/data_files_splits/xray14/official/train_official.txt",
    "cls_nih_valList": DATASET_DIR + "cxr-256/nih_xray14/data_files_splits/xray14/official/val_official.txt",
    "cls_nih_testList": DATASET_DIR + "cxr-256/nih_xray14/data_files_splits/xray14/official/test_official.txt",

    "loc_nih_testTag": "nihLOC_test",
    "loc_nih_root": DATASET_DIR + "NIH_Localization/bbox_img",
    "loc_nih_testList": DATASET_DIR + "NIH_Localization/nih_bbox_coco_1024_fromTensorcsv.json",
    "loc_nih_test2List": DATASET_DIR + "NIH_Localization/nih_bbox_coco_1024_basedOnChestX-Det.json",

    # CheXpert - Classification
    "cls_chexpert_root": DATASET_DIR + "cxr-256/CheXpert/",
    "cls_chexpert_trainList": DATASET_DIR + "cxr-256/CheXpert/data_files_splits/ark6_classificationTask/CheXpert-v1.0_train.csv",
    "cls_chexpert_valList": DATASET_DIR + "cxr-256/CheXpert/data_files_splits/ark6_classificationTask/CheXpert-v1.0_valid.csv",
    "cls_chexpert_testList": DATASET_DIR + "cxr-256/CheXpert/data_files_splits/ark6_classificationTask/CheXpert_test_official.csv",

    # VINDrCXR - Classification
    "cls_vindrcxr_root": DATASET_DIR + "cxr-256/vindr_cxr/",
    "cls_vindrcxr_trainList": DATASET_DIR + "cxr-256/vindr_cxr/data_files_splits/ark6_classificationTask/VinDrCXR_train_pe_global_one.txt",
    "cls_vindrcxr_valList": DATASET_DIR + "cxr-256/vindr_cxr/data_files_splits/ark6_classificationTask/VinDrCXR_test_pe_global_one.txt",
    "cls_vindrcxr_testList": DATASET_DIR + "cxr-256/vindr_cxr/data_files_splits/ark6_classificationTask/VinDrCXR_test_pe_global_one.txt",

    "loc_vindrcxr_trainTag": "vindrcxr_train",
    "loc_vindrcxr_testTag": "vindrcxr_test",
    "loc_vindrcxr_trainRoot": DATASET_DIR + "VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/train_jpeg",
    "loc_vindrcxr_testRoot": DATASET_DIR + "VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/test_jpeg",
    "loc_vindrcxr_trainList": DATASET_DIR + "Vindr-CXR/annotations/VinDrCXR_Kaggle_14Diseases_TRAIN_modified_wbf.json",
    "loc_vindrcxr_testList": DATASET_DIR + "Vindr-CXR/annotations/VinDrCXR_Kaggle_14Diseases_TEST_modified_wbf.json",

    # ShenzenCXR - Classification
    "cls_shenzencxr_root": DATASET_DIR + "shenzhen/CXR_png/",
    "cls_shenzencxr_trainList": DATASET_DIR + "shenzhen/data_files_splits/ark6_classificationTask/ShenzenCXR_train_data.txt",
    "cls_shenzencxr_valList": DATASET_DIR + "shenzhen/data_files_splits/ark6_classificationTask/ShenzenCXR_valid_data.txt",
    "cls_shenzencxr_testList": DATASET_DIR + "shenzhen/data_files_splits/ark6_classificationTask/ShenzenCXR_test_data.txt",

    # Mimic-CXR - Classification
    "cls_mimiccxr_root": DATASET_DIR + "cxr-256/MIMIC_jpeg/",
    "cls_mimiccxr_trainList": DATASET_DIR + "cxr-256/MIMIC_jpeg/data_files_splits/ark6_classificationTask/mimic-cxr-2.0.0-train.csv",
    "cls_mimiccxr_valList": DATASET_DIR + "cxr-256/MIMIC_jpeg/data_files_splits/ark6_classificationTask/mimic-cxr-2.0.0-validate.csv",
    "cls_mimiccxr_testList": DATASET_DIR + "cxr-256/MIMIC_jpeg/data_files_splits/ark6_classificationTask/mimic-cxr-2.0.0-test.csv",

    # TBX11K - Classification - Localization
    "cls_tbx11k_root": DATASET_DIR + "cxr-full/TBX11K/",
    "cls_tbx11k_trainList": "lists/TBX11K_train.txt",
    "cls_tbx11k_valList": None,
    "cls_tbx11k_testList": "lists/TBX11K_val.txt",

    "loc_tbx11k_trainTag": "tbx11k_catagnostic_train",
    "loc_tbx11k_testTag": "tbx11k_catagnostic_test",
    "loc_tbx11k_root": DATASET_DIR + "cxr-full/TBX11K/imgs",
    "loc_tbx11k_trainList": DATASET_DIR + "cxr-full/TBX11K/annotations/json/all_train.json",
    "loc_tbx11k_testList": DATASET_DIR + "cxr-full/TBX11K/annotations/json/all_val.json",

    # NODE21 - Classification - Localization
    "cls_node21_root": DATASET_DIR + "NODE21/",
    "cls_node21_trainList": DATASET_DIR + "NODE21/data_files_splits/node21_dataset/train.txt",
    "cls_node21_valList": None,
    "cls_node21_testList": DATASET_DIR + "NODE21/data_files_splits/node21_dataset/test.txt",

    "loc_node21_trainTag": "node21_noduleDataset_train",
    "loc_node21_testTag": "node21_noduleDataset_test",
    "loc_node21_root": DATASET_DIR + "NODE21/png_images",
    "loc_node21_trainList": DATASET_DIR + "NODE21/data_files_splits/node21_dataset/Node21_Nodule_Bbox_NAD_train_3.json",
    "loc_node21_testList": DATASET_DIR + "NODE21/data_files_splits/node21_dataset/Node21_Nodule_Bbox_NAD_test_3.json",

    # ChestX-Det - Classification - Localization - Segmentation
    "cls_chestxdet_root": DATASET_DIR + "cxr-256/ChestXDet/",
    "cls_chestxdet_trainList": DATASET_DIR + "ChestX-Det/data_files_splits/chestxdetdataset/ChestX_det_train_NAD_v2.json",
    "cls_chestxdet_valList": None,
    "cls_chestxdet_testList": DATASET_DIR + "ChestX-Det/data_files_splits/chestxdetdataset/ChestX_det_test_NAD_v2.json",

    "loc_chestxdet_trainTag": "chestxdet_train",
    "loc_chestxdet_testTag": "chestxdet_test",
    "loc_chestxdet_trainRoot": DATASET_DIR + "cxr-256/ChestXDet/train",
    "loc_chestxdet_testRoot": DATASET_DIR + "cxr-256/ChestXDet/test",
    "loc_chestxdet_trainList": DATASET_DIR + "ChestX-Det/data_files_splits/chestxdetdataset/ChestX_det_train_NAD_v2.json",
    "loc_chestxdet_testList": DATASET_DIR + "ChestX-Det/data_files_splits/chestxdetdataset/ChestX_det_test_NAD_v2.json",

    "seg_chestxdet_trainRoot": DATASET_DIR + "cxr-256/ChestXDet/train/",
    "seg_chestxdet_trainList": DATASET_DIR + "cxr-256/ChestXDet/train_binary_mask/",
    "seg_chestxdet_testRoot": DATASET_DIR + "cxr-256/ChestXDet/test/",
    "seg_chestxdet_testList": DATASET_DIR + "cxr-256/ChestXDet/test_binary_mask/",

    # RSNAPneumonia - Classification - Localization
    "cls_rsnapneumonia_root": DATASET_DIR + "rsna-pneumonia/stage_2_train_images_png/",
    "cls_rsnapneumonia_trainList": DATASET_DIR + "rsna-pneumonia/data_files_splits/rsna_pneumonia/RSNAPneumonia_train.txt",
    "cls_rsnapneumonia_valList": DATASET_DIR + "rsna-pneumonia/data_files_splits/rsna_pneumonia/RSNAPneumonia_val.txt",
    "cls_rsnapneumonia_testList": DATASET_DIR + "rsna-pneumonia/data_files_splits/rsna_pneumonia/RSNAPneumonia_test.txt",

    "loc_rsnapneumonia_trainTag": "rsnaPneumoniaDetection_Train",
    "loc_rsnapneumonia_testTag": "rsnaPneumoniaDetection_Valid",
    "loc_rsnapneumonia_root": DATASET_DIR + "rsna-pneumonia-detection-challenge/stage_2_train_images_png",
    "loc_rsnapneumonia_trainList": DATASET_DIR + "rsna-pneumonia/data_files_splits/rsna_pneumonia/rsnaPneumoniaDetection_Train.json",
    "loc_rsnapneumonia_valList": DATASET_DIR + "rsna-pneumonia/data_files_splits/rsna_pneumonia/rsnaPneumoniaDetection_Valid.json",
    "loc_rsnapneumonia_testList": DATASET_DIR + "rsna-pneumonia/data_files_splits/rsna_pneumonia/rsnaPneumoniaDetection_Test.json",

    # SIIM-ACR - Classification - Localization - Segmentation
    "cls_siimacr_root": DATASET_DIR + "cxr-full/SIIM-ACR-Pneumothorax/",
    "cls_siimacr_trainList": DATASET_DIR + "cxr-full/SIIM-ACR-Pneumothorax/data_files_splits/siimacr_ptx/SIIMPTX_cls_train.txt",
    "cls_siimacr_valList": DATASET_DIR + "cxr-full/SIIM-ACR-Pneumothorax/data_files_splits/siimacr_ptx/SIIMPTX_cls_val.txt",
    "cls_siimacr_testList": DATASET_DIR + "cxr-full/SIIM-ACR-Pneumothorax/data_files_splits/siimacr_ptx/SIIMPTX_cls_test.txt",

    "loc_siimacr_trainTag": "siimacr_train",
    "loc_siimacr_testTag": "siimacr_val",
    "loc_siimacr_trainRoot": DATASET_DIR + "siim_pneumothorax_segmentation/train_jpeg",
    "loc_siimacr_valRoot": DATASET_DIR + "siim_pneumothorax_segmentation/val_jpeg",
    "loc_siimacr_testRoot": DATASET_DIR + "siim_pneumothorax_segmentation/test_jpeg",
    "loc_siimacr_trainList": DATASET_DIR + "cxr-full/SIIM-ACR-Pneumothorax/data_files_splits/siimacr_ptx/siim_pneumothorax_train_coco.json",
    "loc_siimacr_valList": DATASET_DIR + "cxr-full/SIIM-ACR-Pneumothorax/data_files_splits/siimacr_ptx/siim_pneumothorax_val_coco.json",
    "loc_siimacr_testList": DATASET_DIR + "cxr-full/SIIM-ACR-Pneumothorax/data_files_splits/siimacr_ptx/siim_pneumothorax_test_coco.json",

    "seg_siimacr_trainRoot": DATASET_DIR + "siim_pneumothorax_segmentation/train_jpeg",
    "seg_siimacr_trainList": DATASET_DIR + "pxs/train.txt",
    "seg_siimacr_testRoot": DATASET_DIR + "siim_pneumothorax_segmentation/test_jpeg",
    "seg_siimacr_testList": DATASET_DIR + "pxs/test.txt",

    # CANDID-PTX - Classification - Localization - Segmentation
    "cls_candidptx_root": DATASET_DIR + "CANDID-PTX/",
    "cls_candidptx_trainList": DATASET_DIR + "CANDID-PTX/data_files_splits/candid_ptx/CANDIDPTX_cls_train.txt",
    "cls_candidptx_valList": DATASET_DIR + "CANDID-PTX/data_files_splits/candid_ptx/CANDIDPTX_cls_val.txt",
    "cls_candidptx_testList": DATASET_DIR + "CANDID-PTX/data_files_splits/candid_ptx/CANDIDPTX_cls_test.txt",

    "loc_candidptx_trainTag": "candidptx_pneumothorax_train_full",
    "loc_candidptx_testTag": "candidptx_pneumothorax_val",
    "loc_candidptx_root": DATASET_DIR + "CANDID-PTX/png",
    "loc_candidptx_trainList": DATASET_DIR + "CANDID-PTX/data_files_splits/candid_ptx/CANDID_PTX_train_1.json",
    "loc_candidptx_valList": DATASET_DIR + "CANDID-PTX/data_files_splits/candid_ptx/CANDID_PTX_valid_1.json",
    "loc_candidptx_testList": DATASET_DIR + "CANDID-PTX/data_files_splits/candid_ptx/CANDID_PTX_test_1.json",

    "seg_candidptx_root": DATASET_DIR + "CANDID-PTX/dataset",
    "seg_candidptx_trainList": DATASET_DIR + "CANDID-PTX/data_files_splits/candid_ptx/train.txt",
    "seg_candidptx_testList": DATASET_DIR + "CANDID-PTX/data_files_splits/candid_ptx/test.txt",


    # RSNA PE - Classification - Localization
    "cls_rsnape_root": DATASET_DIR + "/RawInputImages_Exams_ODDS_EVENS_indiNP_256each_224by224",
    "cls_rsnape_trainList": DATASET_DIR + "RSNA_PE/data_files_splits/rsna_pe_cls/RSNA_Exam_Odds_Evens_Train_256List.txt",
    "cls_rsnape_valList": DATASET_DIR + "RSNA_PE/data_files_splits/rsna_pe_cls/RSNA_Exam_Odds_Evens_Val_256List.txt",
    "cls_rsnape_testList": None,

    "loc_rspect_trainTag": "rspect_train",
    "loc_rspect_testTag": "rspect_test",
    "loc_rspect_root": DATASET_DIR + "RSNA_PE_RSPECT/localization_png_augmented_files_224by224",
    "loc_rspect_trainList": DATASET_DIR + "RSNA_PE_RSPECT/RSPECT_384_Train_Loc_png_v2_50perBigger_224by224.json",
    "loc_rspect_valList": None,
    "loc_rspect_testList": DATASET_DIR + "/RSNA_PE_RSPECT/RSPECT_61_Test_Loc_png_v2_50perBigger_224by224.json",

    # RadFusion PE - Classification
    "cls_radfusion_trainRoot": DATASET_DIR + "CT_RadFusion/RawInputImages_Exams_ODDS_EVENS_indiNP_256each_224by224",
    "cls_radfusion_testRoot": DATASET_DIR + "CT_RadFusion/RawInputImages_Exams_ODDS_EVENS_indiNP_256each_TEST_224by224",
    "cls_radfusion_trainList": DATASET_DIR + "CT_RadFusion/RadFusion_Exam_Odds_Evens_Train+Val_256List_v1.txt",
    "cls_radfusion_valList": DATASET_DIR + "CT_RadFusion/RadFusion_Exam_Odds_Evens_Test_256List.txt",
    "cls_radfusion_testList": None,

    # INSPECT - Classification
    "cls_inspect_Root": DATASET_DIR + "stanford/INSPECT-Multaimodal-PE/full/CTPA/",
    "cls_inspect_testRoot": DATASET_DIR + "INSPECT-Multimodal-PE/CT_INSPECT_26gb/INSPECT_Official_TrainList.csv",
    "cls_inspect_trainList": None,
    "cls_inspect_valList": DATASET_DIR + "INSPECT-Multimodal-PE/CT_INSPECT_26gb/INSPECT_Official_TestList.csv",
    "cls_inspect_testList": None,

    # FUMPE - Classification - Localization - Segmentation
    "cls_fumpe_root": DATASET_DIR + "FUMPE/",
    "cls_fumpe_trainList": "",
    "cls_fumpe_valList": "",
    "cls_fumpe_testList": None,

    "loc_fumpe_trainTag": "fumpe_train",
    "loc_fumpe_testTag": "fumpe_test",
    "loc_fumpe_root": DATASET_DIR + "FUMPE/png_images_224by224",
    "loc_fumpe_trainList": DATASET_DIR + "FUMPE/Train_BBox_List_v2_50perBigger_224by224.json",
    "loc_fumpe_valList": None,
    "loc_fumpe_testList": DATASET_DIR + "FUMPE/Test_BBox_List_v2_50perBigger_224by224.json",

    "seg_fumpe_rootDir": DATASET_DIR + "FUMPE/",
    "seg_fumpe_root": DATASET_DIR + "FUMPE/np_images_224by224/",
    "seg_fumpe_mask": "np_masks_224by224/",

    # PE CAD 91 - Classification - Localization - Segmentation
    "cls_pecad91_root": DATASET_DIR + "PE_CAD_91_Challenge/",
    "cls_pecad91_trainList": "",
    "cls_pecad91_valList": "",
    "cls_pecad91_testList": None,

    "loc_pecad91_trainTag": "pecad91_train",
    "loc_pecad91_testTag": "pecad91_test",
    "loc_pecad91_root": DATASET_DIR + "PE_CAD_91_Challenge/png_images_v3_224by224",
    "loc_pecad91_trainList": DATASET_DIR + "PE_CAD_91_Challenge/Train_BBox_List_v2_50perBigger_v2_224by224.json",
    "loc_pecad91_valList": None,
    "loc_pecad91_testList": DATASET_DIR + "PE_CAD_91_Challenge/Test_BBox_List_v2_50perBigger_v2_224by224.json",

    "seg_pecad91_rootDir": DATASET_DIR + "PE_CAD_91_Challenge/",
    "seg_pecad91_root": DATASET_DIR + "PE_CAD_91_Challenge/np_images_v3_224by224/",
    "seg_pecad91_mask": DATASET_DIR + "PE_CAD_91_Challenge/np_masks_v2_224by224/",

    # SARS COVID CT SCAN - Classification
    "cls_SARScovid_root": DATASET_DIR + "SARS-COV-2_CT_Scan",
    "cls_SARScovid_trainList": DATASET_DIR + "SARS-COV-2_CT_Scan/SARS_train_nonOfficial_split.txt",
    "cls_SARScovid_valList": "",
    "cls_SARScovid_testList": DATASET_DIR + "SARS-COV-2_CT_Scan/SARS_test_nonOfficial_split.txt",

    # SARS Chest-CTscan-Images-Dataset - Classification
    "cls_ChestCTscanImages_root": DATASET_DIR + "Chest-CTscan-Images-Dataset/Data",
    "cls_ChestCTscanImages_trainList": DATASET_DIR + "Chest-CTscan-Images-Dataset/Data/Chest_CTscan_Train.txt",
    "cls_ChestCTscanImages_valList": DATASET_DIR + "Chest-CTscan-Images-Dataset/Data/Chest_CTscan_Valid.txt",
    "cls_ChestCTscanImages_testList": DATASET_DIR + "Chest-CTscan-Images-Dataset/Data/Chest_CTscan_Test.txt",

    # COVID-CT-MD - Classification
    "cls_CovidCTMd_root": DATASET_DIR + "COVID-CT-MD/Data_Processed_224x224", # "/scratch/jliang12/data/COVID-CT-MD/Data_Processed_224x224", # "/data/jliang12/shared/dataset/COVID-CT-MD/Data_Processed_224x224"
    "cls_CovidCTMd_trainList": DATASET_DIR + "COVID-CT-MD/Data_Processed_224x224/Annotations/COVID-CT-MD_Train.txt",
    "cls_CovidCTMd_valList": None,
    "cls_CovidCTMd_testList": DATASET_DIR + "COVID-CT-MD/Data_Processed_224x224/Annotations/COVID-CT-MD_Test.txt",

    # LIDC-IDRI - Classification - Localization - Segmentation
    "cls_lidcidri_root":DATASET_DIR + "LIDC-IDRI-Segmentation-CT/Data_Processed_Correct_224x224",  # /scratch/jliang12/data/LIDC-IDRI-Segmentation-CT/Data_Processed_Correct_224x224  /data/jliang12/shared/dataset/LIDC-IDRI-Segmentation-CT/Data_Processed_Correct_224x224
    "cls_lidcidri_trainList": DATASET_DIR + "LIDC-IDRI-Segmentation-CT/Annotations/LIDC-IDRI_train_slices_k10.txt",
    "cls_lidcidri_valList": None,
    "cls_lidcidri_testList": DATASET_DIR + "LIDC-IDRI-Segmentation-CT/Annotations/LIDC-IDRI_test_slices_k10.txt",

    "loc_lidcidri_trainTag": "lidcidri_train",
    "loc_lidcidri_testTag": "lidcidri_test",
    "loc_lidcidri_root": DATASET_DIR + "LIDC-IDRI-Segmentation-CT/Data_Processed_Correct_224x224",
    "loc_lidcidri_trainList": DATASET_DIR + "LIDC-IDRI-Segmentation-CT/Annotations/LIDC-IDRI_coco_train_k10.json",
    "loc_lidcidri_valList": None,
    "loc_lidcidri_testList": DATASET_DIR + "LIDC-IDRI-Segmentation-CT/Annotations/LIDC-IDRI_coco_test_k10.json",

    "seg_lidcidri_rootDir": DATASET_DIR + "LIDC-IDRI-Segmentation-CT/Data_Processed_Correct_224x224",
    "seg_lidcidri_trainList": DATASET_DIR + "LIDC-IDRI-Segmentation-CT/Annotations/LIDC-IDRI_train_slices_k10.txt",
    "seg_lidcidri_testList": DATASET_DIR + "LIDC-IDRI-Segmentation-CT/Annotations/LIDC-IDRI_test_slices_k10.txt",

    # NSCLC - Classification - Localization - Segmentation
    "cls_nsclc_root": DATASET_DIR + "NSCLC_Radiomics/data_processed", ## "/data/jliang12/shared/dataset/NSCLC_Radiomics/data_processed", 
    "cls_nsclc_trainList": DATASET_DIR + "NSCLC_Radiomics/annotations/NSCLC_CT_train_list_k20.txt",
    "cls_nsclc_valList": None,
    "cls_nsclc_testList": DATASET_DIR + "NSCLC_Radiomics/annotations/NSCLC_CT_test_list_k20.txt",

    "loc_nsclc_trainTag": "nsclc_train",
    "loc_nsclc_testTag": "nsclc_test",
    "loc_nsclc_root": DATASET_DIR + "NSCLC_Radiomics/data_processed",
    "loc_nsclc_trainList": DATASET_DIR + "NSCLC_Radiomics/annotations/NSCLC_coco_train_bbox_k20.json",
    "loc_nsclc_valList": None,
    "loc_nsclc_testList": DATASET_DIR + "NSCLC_Radiomics/annotations/NSCLC_coco_test_bbox_k20.json",

    "seg_nsclc_rootDir": DATASET_DIR + "NSCLC_Radiomics/data_processed",
    "seg_nsclc_trainList": DATASET_DIR + "NSCLC_Radiomics/annotations/NSCLC_CT_train_list_k20.txt",
    "seg_nsclc_testList": DATASET_DIR + "NSCLC_Radiomics/annotations/NSCLC_CT_test_list_k20.txt",

    # # x
    # "cls_x_root": "",
    # "cls_x_trainList": "",
    # "cls_x_valList": "",
    # "cls_x_testList": "",
}


DATASETS_HEADS = {
    # -------- Classification heads --------
    "TBX11K_CLS": 0,
    "NODE21_CLS": 1,
    "CANDIDPTX_CLS": 2,
    "RNSNAPNEUMONIA_CLS": 3,
    "CHESTXDET_CLS": 4,
    "SIIMACR_CLS": 5,
    "CHEXPERT_CLS": 6,
    "NIHCHESTXRAY14_CLS": 7,
    "VINDRCXR_CLS": 8,
    "NIHSHENZEN_CLS": 9,
    "MIMIC2_CLS": 10,

    # -------- Localization heads --------
    "TBX11K_LOC": 0,
    "NODE21_LOC": 1,
    "CANDIDPTX_LOC": 2,
    "RSNAPNUMONIA_LOC": 3,
    "CHESTXDET_LOC": 4,
    "SIIMACR_LOC": 5,

    # -------- Segmentation heads --------
    "CANDIDPTX_SEG": 2,
    "CHESTXDET_SEG": 4,
    "SIIMACR_SEG": 5,
}


def _load_dataset_config_from_yml():
    """Load dataset paths from YAML if FOUNDATION_X_DATASET_LOCATIONS_YML env var is set."""
    yml_path = os.environ.get("FOUNDATION_X_DATASET_LOCATIONS_YML")
    if not yml_path:
        return  # Use defaults
    
    try:
        with open(yml_path, 'r') as f:
            yml_config = yaml.safe_load(f)
        
        if not yml_config or 'classification' not in yml_config and 'directories' not in yml_config:
            return  # Invalid yml, use defaults
        
        # Helper to build full paths
        classroom_root = yml_config.get('roots', {}).get('classroom', '')
        
        # Map dataset names to config keys
        dataset_mapping = {
            'nih_chestxray14': ('nih', 'cls'),
            'chexpert': ('chexpert', 'cls'),
            'mimic_cxr': ('mimiccxr', 'cls'),
            'vindr_cxr': ('vindrcxr', 'cls'),
            'shenzhen': ('shenzencxr', 'cls'),
            'tbx11k': ('tbx11k', 'cls_loc'),
            'node21': ('node21', 'cls_loc'),
            'chestx_det': ('chestxdet', 'cls_loc_seg'),
            'rsna_pneumonia': ('rsnapneumonia', 'cls_loc'),
            'siim_acr_ptx': ('siimacr', 'cls_loc_seg'),
            'candid_ptx': ('candidptx', 'cls_loc_seg'),
        }
        
        # Process classification datasets
        classification = yml_config.get('classification', {})
        for yml_key, (config_prefix, task_type) in dataset_mapping.items():
            if yml_key not in classification:
                continue
            
            dataset = classification[yml_key]
            if 'cls' in task_type:
                # Update classification paths
                root = dataset.get('images', '')
                splits = dataset.get('splits', {})
                
                config_root_key = f'cls_{config_prefix}_root'
                config_train_key = f'cls_{config_prefix}_trainList'
                config_val_key = f'cls_{config_prefix}_valList'
                config_test_key = f'cls_{config_prefix}_testList'
                
                if config_root_key in DATASETS_CONFIG and root:
                    DATASETS_CONFIG[config_root_key] = root
                
                if config_train_key in DATASETS_CONFIG and 'train' in splits:
                    train_path = splits['train']
                    if not train_path.startswith('/'):
                        train_path = os.path.join(root, train_path) if root else train_path
                    DATASETS_CONFIG[config_train_key] = train_path
                
                if config_val_key in DATASETS_CONFIG and 'val' in splits:
                    val_path = splits['val']
                    if not val_path.startswith('/'):
                        val_path = os.path.join(root, val_path) if root else val_path
                    DATASETS_CONFIG[config_val_key] = val_path
                
                if config_test_key in DATASETS_CONFIG and 'test' in splits:
                    test_path = splits['test']
                    if not test_path.startswith('/'):
                        test_path = os.path.join(root, test_path) if root else test_path
                    DATASETS_CONFIG[config_test_key] = test_path
        
        # Process localization datasets
        for yml_key, (config_prefix, task_type) in dataset_mapping.items():
            if yml_key not in classification:
                continue
            
            dataset = classification[yml_key]
            if 'loc' not in task_type:
                continue
            
            splits = dataset.get('splits', {})
            root = dataset.get('images', '')
            
            config_root_key = f'loc_{config_prefix}_root'
            config_train_key = f'loc_{config_prefix}_trainList'
            config_test_key = f'loc_{config_prefix}_testList'
            
            # Allow localization to override root independently from classification,
            # e.g. classification.node21.loc_node21_root in YAML.
            loc_root = dataset.get(config_root_key, root)
            if config_root_key in DATASETS_CONFIG and loc_root:
                DATASETS_CONFIG[config_root_key] = loc_root
            
            # Handle JSON splits for localization
            json_train_key = f'train_json'
            json_test_key = f'test_json'
            
            config_val_key = f'loc_{config_prefix}_valList'
            json_val_key = 'val_json'

            if json_train_key in splits and config_train_key in DATASETS_CONFIG:
                DATASETS_CONFIG[config_train_key] = splits[json_train_key]
            
            if json_val_key in splits and config_val_key in DATASETS_CONFIG:
                DATASETS_CONFIG[config_val_key] = splits[json_val_key]

            if json_test_key in splits and config_test_key in DATASETS_CONFIG:
                DATASETS_CONFIG[config_test_key] = splits[json_test_key]
        
        # Process segmentation datasets
        segmentation = yml_config.get('segmentation', {})
        for yml_key, (config_prefix, task_type) in dataset_mapping.items():
            if yml_key not in segmentation:
                continue
            
            if 'seg' not in task_type:
                continue
            
            seg_config = segmentation[yml_key]
            
            config_root_key = f'seg_{config_prefix}_root'
            config_train_root_key = f'seg_{config_prefix}_trainRoot'
            config_train_list_key = f'seg_{config_prefix}_trainList'
            config_test_root_key = f'seg_{config_prefix}_testRoot'
            config_test_list_key = f'seg_{config_prefix}_testList'
            
            if 'images_train' in seg_config and config_train_root_key in DATASETS_CONFIG:
                DATASETS_CONFIG[config_train_root_key] = seg_config['images_train']
            
            if 'masks_train' in seg_config and config_train_list_key in DATASETS_CONFIG:
                DATASETS_CONFIG[config_train_list_key] = seg_config['masks_train']
            
            if 'images_test' in seg_config and config_test_root_key in DATASETS_CONFIG:
                DATASETS_CONFIG[config_test_root_key] = seg_config['images_test']
            
            if 'masks_test' in seg_config and config_test_list_key in DATASETS_CONFIG:
                DATASETS_CONFIG[config_test_list_key] = seg_config['masks_test']

            if 'train_list' in seg_config and config_train_list_key in DATASETS_CONFIG:
                DATASETS_CONFIG[config_train_list_key] = seg_config['train_list']

            if 'test_list' in seg_config and config_test_list_key in DATASETS_CONFIG:
                DATASETS_CONFIG[config_test_list_key] = seg_config['test_list']
    
    except Exception as e:
        print(f"Warning: Failed to load dataset config from {yml_path}: {e}")


# Load from YAML if environment variable is set
_load_dataset_config_from_yml()