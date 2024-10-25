# test_11.py

from typing import List, Tuple, Dict, Optional
from abc import ABC, ABCMeta, abstractmethod
from typing import List, Dict, Union, Optional

def easy_addition(a: int, b: int) -> int:
    """
    Add two integers together.
    
    Parameters
    ----------
    a : int
        The first integer.
    b : int
        The second integer.
    
    Returns
    -------
    int
        The sum of `a` and `b`.
    
    Examples
    --------
    >>> easy_addition(2, 3)
    5
    """
    return a + b


def medium_process_data(data: List[float], threshold: float = 0.5) -> List[Tuple[int, float]]:
    """
    Process a list of float values and filter them based on a threshold.
    
    This function takes a list of float values, normalizes them, and filters out values 
    that are below a certain threshold. The filtered results are returned as a list of tuples, 
    where each tuple contains the index of the value and the normalized value.
    
    Parameters
    ----------
    data : List[float]
        A list of float values to be processed.
    threshold : float, optional
        The threshold below which values will be discarded (default is 0.5).
    
    Returns
    -------
    List[Tuple[int, float]]
        A list of tuples where each tuple contains the index of the value and the normalized value.
    
    Raises
    ------
    ValueError
        If `data` is empty or contains non-numeric values.
    
    Notes
    -----
    This function assumes that all values in `data` are positive. Normalization is 
    performed by dividing each value by the sum of all values in the list.
    
    Examples
    --------
    >>> medium_process_data([0.2, 0.5, 0.8, 0.4], threshold=0.4)
    [(1, 0.5), (2, 0.8)]
    """
    if not data:
        raise ValueError("The input data cannot be empty.")
        
    total = sum(data)
    normalized_data = [(index, value / total) for index, value in enumerate(data)]
    return [(index, value) for index, value in normalized_data if value >= threshold]


def hard_analyze_complex_structure(
    structure: Dict[str, List[Optional[int]]], factor: float = 1.5, iterations: int = 3
) -> Dict[str, float]:
    """
    Analyze a complex data structure and compute weighted averages.
    
    This function processes a dictionary where each key maps to a list of integers or `None` values. 
    It will compute a weighted average for each key, ignoring `None` values, and apply a scaling 
    factor. The result is a dictionary mapping the original keys to the computed averages.
    
    Parameters
    ----------
    structure : Dict[str, List[Optional[int]]]
        A dictionary containing lists of integers or `None` values.
    factor : float, optional
        A scaling factor to apply to each computed average (default is 1.5).
    iterations : int, optional
        Number of times to run the smoothing operation for more refined results (default is 3).
    
    Returns
    -------
    Dict[str, float]
        A dictionary where each key corresponds to the original input keys and the values are 
        the weighted averages after applying the scaling factor.
    
    Raises
    ------
    ValueError
        If `structure` is empty or if any list in the dictionary is empty.
    
    See Also
    --------
    medium_process_data : This function can be used to preprocess the data before analysis.
    
    Notes
    -----
    The smoothing operation applies a simple averaging over the computed weighted average 
    and the scaling factor for a specified number of iterations. This allows for gradual 
    adjustment of the final values.
    
    Examples
    --------
    >>> hard_analyze_complex_structure({'a': [1, 2, None], 'b': [3, None, 6]}, factor=2.0)
    {'a': 3.0, 'b': 12.0}
    """
    def compute_weighted_average(values: List[Optional[int]]) -> float:
        clean_values = [v for v in values if v is not None]
        if not clean_values:
            return 0.0
        weight = len(clean_values) / sum(range(1, len(clean_values) + 1))
        return weight * sum(clean_values) / len(clean_values)

    if not structure:
        raise ValueError("The input structure cannot be empty.")
    
    result = {}
    for key, values in structure.items():
        if not values:
            raise ValueError(f"The list for key '{key}' cannot be empty.")
            
        average = compute_weighted_average(values)
        # Apply iterative smoothing and scaling
        for _ in range(iterations):
            average = (average + factor) / 2
        result[key] = average * factor
    
    return result





class SimpleCounter:
    """
    A simple counter class that can increment, decrement, and reset its value.
    
    Attributes
    ----------
    count : int
        The current value of the counter.
    
    Methods
    -------
    increment(by: int = 1) -> None
        Increments the counter by a specified value.
    decrement(by: int = 1) -> None
        Decrements the counter by a specified value.
    reset() -> None
        Resets the counter to zero.
    """
    
    def __init__(self, initial: int = 0):
        """
        Initializes the counter with an optional starting value.
        
        Parameters
        ----------
        initial : int, optional
            The initial value of the counter (default is 0).
        """
        self.count = initial

    def increment(self, by: int = 1) -> None:
        """Increments the counter by a specified value."""
        self.count += by

    def decrement(self, by: int = 1) -> None:
        """Decrements the counter by a specified value."""
        self.count -= by

    def reset(self) -> None:
        """Resets the counter to zero."""
        self.count = 0


class DataProcessor(ABC):
    """
    Abstract base class for data processors.
    
    This class defines the interface for any data processing class that needs to 
    load, process, and save data. Subclasses should implement these methods.
    
    Methods
    -------
    load_data(source: str) -> None
        Loads data from the specified source.
    process_data() -> None
        Processes the loaded data.
    save_data(destination: str) -> None
        Saves the processed data to the specified destination.
    """
    
    @abstractmethod
    def load_data(self, source: str) -> None:
        pass

    @abstractmethod
    def process_data(self) -> None:
        pass

    @abstractmethod
    def save_data(self, destination: str) -> None:
        pass


class CSVDataProcessor(DataProcessor):
    """
    A concrete implementation of DataProcessor for handling CSV files.
    
    This class can load data from CSV files, process it, and save it back to CSV files.
    
    Attributes
    ----------
    data : List[Dict[str, Union[str, int, float]]]
        The data loaded from the CSV file.
    
    Methods
    -------
    load_data(source: str) -> None
        Loads data from a CSV file.
    process_data() -> None
        Cleans and processes the loaded CSV data.
    save_data(destination: str) -> None
        Saves the processed data back to a CSV file.
    _clean_data() -> None
        Helper method to clean the loaded data.
    """
    
    def __init__(self):
        """Initializes the CSVDataProcessor with an empty dataset."""
        self.data = []

    def load_data(self, source: str) -> None:
        """
        Loads data from a CSV file.
        
        Parameters
        ----------
        source : str
            The path to the CSV file to load data from.
        """
        import csv
        
        with open(source, mode='r') as file:
            reader = csv.DictReader(file)
            self.data = [row for row in reader]

    def process_data(self) -> None:
        """Cleans and processes the loaded CSV data."""
        self._clean_data()

    def save_data(self, destination: str) -> None:
        """
        Saves the processed data back to a CSV file.
        
        Parameters
        ----------
        destination : str
            The path to save the processed CSV data.
        """
        import csv
        
        with open(destination, mode='w', newline='') as file:
            if self.data:
                writer = csv.DictWriter(file, fieldnames=self.data[0].keys())
                writer.writeheader()
                writer.writerows(self.data)

    def _clean_data(self) -> None:
        """Helper method to clean the loaded data by removing empty fields."""
        self.data = [{k: v for k, v in row.items() if v is not None and v != ""} for row in self.data]


class AdvancedPipeline(ABC):
    """
    An abstract base class representing an advanced data processing pipeline.
    
    Attributes
    ----------
    name : str
        The name of the pipeline.
    
    Methods
    -------
    setup() -> None
        Sets up the pipeline with necessary configurations.
    run(data: Optional[List[float]] = None) -> Union[str, float]
        Runs the pipeline process.
    shutdown() -> None
        Shuts down the pipeline after execution.
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def setup(self) -> None:
        pass

    @abstractmethod
    def run(self, data: Optional[List[float]] = None) -> Union[str, float]:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass


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