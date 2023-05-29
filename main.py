from balancing import DataBalancingStage
from cleaning import DataCleaningStage
from evaluate import EvaluationStage
from feature_engineering import FeatureEngineeringStage
from gathering import DataGatheringStage
from preprocessing import PreprocessingStage

if __name__ == '__main__':
    data_gathering_stage = DataGatheringStage()
    cleaning_stage = DataCleaningStage(outlier_removal_method=0)
    balancing_stage = DataBalancingStage("adasyn") # random_undersample, smote or adasyn
    feature_engineering_stage =FeatureEngineeringStage()
    preprocesing_stage = PreprocessingStage()
    model_evaluation_stage = EvaluationStage()

    pipeline = [
        #data_gathering_stage,
        #cleaning_stage,
        #balancing_stage,
        #feature_engineering_stage,
        #preprocesing_stage,
        model_evaluation_stage
    ]

    for current_stage in pipeline:
        current_stage.execute_stage()
