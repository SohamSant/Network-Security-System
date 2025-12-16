from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    """
    Artifact (output) from the Data Ingestion component.
    Contains paths to the generated train and test datasets.
    """
    trained_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    """
    Artifact (output) from the Data Validation component.
    Contains validation status and report path.
    """
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str


@dataclass
class DataTransformationArtifact:
    """
    Artifact (output) from the Data Transformation component.
    Contains paths to transformed data and preprocessing objects.
    """
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str


@dataclass
class ClassificationMetricArtifact:
    """
    Artifact containing classification metrics for model evaluation.
    """
    f1_score: float
    precision_score: float
    recall_score: float


@dataclass
class ModelTrainerArtifact:
    """
    Artifact (output) from the Model Trainer component.
    Contains paths to trained model and training metrics.
    """
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact


@dataclass
class ModelEvaluationArtifact:
    """
    Artifact (output) from the Model Evaluation component.
    Determines if the new model should replace the existing one.
    """
    is_model_accepted: bool
    improved_accuracy: float
    best_model_path: str
    trained_model_path: str
    train_model_metric_artifact: ClassificationMetricArtifact
    best_model_metric_artifact: ClassificationMetricArtifact


@dataclass
class ModelPusherArtifact:
    """
    Artifact (output) from the Model Pusher component.
    Contains path where the model is pushed/deployed.
    """
    saved_model_path: str
    model_file_path: str
