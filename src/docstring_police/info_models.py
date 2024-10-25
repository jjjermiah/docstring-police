import ast
import json
from typing import List, Optional, Dict
from dataclasses import dataclass, asdict, field

@dataclass
class FunctionInfo:
    name: str
    docstring: Optional[str]
    args: Optional[str]
    return_val: Optional[str] = field(default=None)

@dataclass
class ClassInfo:
    class_name: str
    docstring: Optional[str]
    methods: List[FunctionInfo] = field(default_factory=list)
    

    def to_json(self) -> str:
        """Convert the ClassInfo object to a JSON string."""
        return json.dumps(asdict(self), indent=4)

def extract_function_info(func_node: ast.FunctionDef) -> FunctionInfo:
    """Extracts information about a function/method including name, docstring, and type annotations."""
    function_name = func_node.name
    docstring = ast.get_docstring(func_node, clean=True)
    args = ast.unparse(func_node.args)
    return_val = ast.unparse(func_node.returns) if func_node.returns else None

    return FunctionInfo(name=function_name, docstring=docstring, args=args, return_val = return_val)

def extract_class_info(class_node: ast.ClassDef) -> ClassInfo:
    # Extract class name
    class_name = class_node.name
    
    # Extract class-level docstring
    docstring = ast.get_docstring(class_node, clean=True)
    
    # Extract method information
    methods = []
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef):  # It's a method
            method_info = extract_function_info(node)
            methods.append(method_info)
    
    # Create and return the ClassInfo object
    return ClassInfo(class_name=class_name, docstring=docstring, methods=methods)

# Example usage:
if __name__ == "__main__":
    from docstring_police.extractor import CodeBlockExtractor
    from pathlib import Path
    class_string = '''
class MLTrainingPipeline(AdvancedPipeline, metaclass=ABCMeta):
    """
    A machine learning training pipeline for processing data and training models.
    
    This class represents a customizable machine learning pipeline designed to 
    streamline the process of data preparation, model training, and evaluation. It supports 
    flexible configurations and allows for integration with various machine learning frameworks.
    
    Attributes
    ----------
    name : str
        The name of the pipeline.
    model_params : Dict[str, Union[str, int, float]]
        Parameters for the machine learning model.
    data : Optional[List[float]]
        The data to be processed by the pipeline.
    
    Methods
    -------
    setup() -> None
        Configures the pipeline for machine learning training.
    run(data: Optional[List[float]] = None) -> float
        Runs the training process and returns the model accuracy.
    shutdown() -> None
        Finalizes the pipeline after training.
    _train_model(data: List[float]) -> float
        Trains a model using the provided data and returns accuracy.
    
    Parameters
    ----------
    name : str
        The name of the pipeline. (Default value: "bob")
    model_params : Dict[str, Union[str, int, float]]
        Parameters for configuring the machine learning model.
    
    Raises
    ------
    ValueError
        If no data is provided for training.
    
    See Also
    --------
    AdvancedPipeline : The base class that defines the interface for this pipeline.
    DataProcessor : Can be used for preprocessing steps before data is passed to this pipeline.
    
    Notes
    -----
    This class is designed to be extended for more specific machine learning pipelines.
    The `run` method integrates data processing, model training, and evaluation into a 
    single step to simplify typical machine learning workflows. The pipeline can be adjusted 
    for different datasets and model configurations by modifying the `model_params` attribute.
    
    References
    ----------
    .. [1] Pedregosa et al., "Scikit-learn: Machine Learning in Python", JMLR 12, pp. 2825-2830, 2011.
       Available: https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html
    .. [2] "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow", Géron, A., 2nd Edition.
    
    Examples
    --------
    >>> model_params = {"learning_rate": 0.01, "num_layers": 3, "activation": "relu"}
    >>> pipeline = MLTrainingPipeline("ExamplePipeline", model_params)
    >>> pipeline.setup()
    Setting up the ExamplePipeline pipeline with parameters: {'learning_rate': 0.01, 'num_layers': 3, 'activation': 'relu'}
    >>> data = [0.5, 0.7, 1.2, 1.5, 2.0]
    >>> accuracy = pipeline.run(data)
    >>> print(f"Model accuracy: {accuracy}")
    Model accuracy: 1.18
    >>> pipeline.shutdown()
    Shutting down the ExamplePipeline pipeline.
    """
    
    def __init__(self, model_params: Dict[str, Union[str, int, float]],  name: str = "bob",):
        super().__init__(name)
        self.model_params = model_params
        self.data = None

    def setup(self) -> None:
        """Configures the pipeline for machine learning training."""
        print(f"Setting up the {self.name} pipeline with parameters: {self.model_params}")
        
    def run(self, data: Optional[List[float]] = None) -> float:
        """
        Runs the training process and returns the model accuracy.
        
        Parameters
        ----------
        data : Optional[List[float]]
            The data to be processed. If not provided, the pipeline will use internal data.
        
        Returns
        -------
        float
            The accuracy of the trained model.
        
        Raises
        ------
        ValueError
            If no data is provided for training.
        """
        if data is not None:
            self.data = data
        if self.data is None:
            raise ValueError("No data provided for training.")
        
        return self._train_model(start=1, data=self.data)

    def shutdown(self) -> None:
        """Finalizes the pipeline after training."""
        print(f"Shutting down the {self.name} pipeline.")

    def _train_model(self, start: Union[int,str], data: List[float] = [0.5, 0.7, 1.2, 1.5, 2.0], name: str = "bob") -> float:
        """Trains a model using the provided data and returns accuracy."""
        # Placeholder logic for training
        accuracy = sum(data) / len(data)  # Simplified accuracy calculation
        return accuracy
    '''

    tree = ast.parse(class_string)
    class_node = tree.body[0] 
    assert isinstance(class_node, ast.ClassDef)
    class_info = extract_class_info(class_node)
    print(class_info.to_json())
    