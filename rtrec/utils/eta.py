from abc import ABC, abstractmethod
from typing import Any

class EtaEstimator(ABC):
    def __init__(self, initial_eta: float):
        """
        Initialize the EtaEstimator with the initial learning rate.
        
        :param initial_eta: Initial value for the learning rate.
        """
        self.eta0 = initial_eta
    
    @abstractmethod
    def update(self):
        """
        Update the estimator state. This method should be called once per update.
        """
        pass

    @abstractmethod
    def get_eta(self) -> float:
        """
        Calculate the learning rate based on the current state of the estimator.
        
        :return: The current learning rate eta.
        """
        pass

    @property
    def value(self) -> float:
        """
        Accessor property to get the current learning rate.
        
        :return: The current learning rate eta.
        """
        return self.get_eta()

class FixedEtaEstimator(EtaEstimator):
    def __init__(self, initial_eta: float = 0.1):
        """
        Initialize the FixedEtaEstimator with a fixed learning rate.
        
        :param initial_eta: Fixed value for the learning rate.
        """
        super().__init__(initial_eta)

    def update(self):
        """
        No state update needed for a fixed learning rate.
        """
        pass

    def get_eta(self) -> float:
        """
        Get the fixed learning rate.
        
        :return: The fixed learning rate eta.
        """
        return self.eta0

class InvscalingEtaEstimator(EtaEstimator):
    def __init__(self, initial_eta: float = 0.1, power_t: float = 0.1):
        """
        Initialize the InvscalingEtaEstimator with initial_eta and power_t.
        
        :param initial_eta: Initial learning rate value.
        :param power_t: Power for the inverse scaling.
        """
        super().__init__(initial_eta)
        self.power_t = power_t
        self.t = 0  # Initialize iteration count

    def update(self):
        """
        Increment the iteration count.
        """
        self.t += 1

    def get_eta(self) -> float:
        """
        Calculate the learning rate based on the inverse scaling formula.
        
        :return: The current learning rate eta.
        """
        if self.t == 0:
            return self.eta0 # Avoid division by zero
        return self.eta0 / pow(self.t, self.power_t)


def get_eta_estimator(name: str, **kwargs: Any) -> EtaEstimator:
    """
    Create an instance of an EtaEstimator based on the provided name and keyword arguments.

    :param name: The type of estimator to create ("fixed" or "invscaling").
    :param kwargs: Additional parameters required for the estimator initialization.
    :return: An instance of the specified EtaEstimator subclass.
    :raises ValueError: If an unknown estimator name is provided.
    """
    name = name.lower()

    if name == "fixed":
        initial_eta = kwargs.get("eta0", 0.1)
        return FixedEtaEstimator(initial_eta)
    elif name == "invscaling":
        initial_eta = kwargs.get("eta0", 0.1)
        power_t = kwargs.get("power_t", 0.1)
        return InvscalingEtaEstimator(initial_eta, power_t)
    else:
        raise ValueError(f"Unknown estimator type '{name}'. Available options: 'fixed', 'invscaling'.")
