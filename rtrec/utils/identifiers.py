from typing import Any, Optional

import numpy as np

class IdentifierError(Exception):
    """Custom exception for Identifier class key errors."""
    def __init__(self, id_name: str, obj_id: int):
        super().__init__(f"Identifier not found for {id_name}: {obj_id}")

class Identifier:
    def __init__(self, name: str="ID", force_identify: bool=False, **kwargs: Any) -> None:
        """
        Initialize the Identifier class.
        Args:
            name (str): Name of the identifier.
            force_identify (bool): Flag to force identification.
        """
        self.name = name
        self.force_identify = force_identify  # Flag to force identification
        self.obj_to_id: dict[Any, int] = {}  # Store object-to-ID mapping
        self.id_to_obj: list[Any] = []  # Store ID-to-object mapping
        if force_identify:
            self.pass_through = False
        else:
            self.pass_through : Optional[bool] = None # If True, return the object as-is if it is an integer

    def identify(self, obj: Any) -> int:
        # If the object is an integer, return it as-is
        if not self.force_identify:
            if isinstance(obj, int):
                if self.pass_through is False:
                    raise ValueError(f"Mixed types detected for {self.name}: {obj}")
                self.pass_through = True
                return obj
            elif isinstance(obj, np.integer):
                if self.pass_through is False:
                    raise ValueError(f"Mixed types detected for {self.name}: {obj}")
                self.pass_through = True
                return int(obj) # Convert numpy integer to Python integer

        if self.pass_through is True:
            raise ValueError(f"Mixed types detected for {self.name}: {obj}")

        # If the object is already in the set, find its ID
        obj_id = self.obj_to_id.get(obj, None)
        if obj_id is not None:
            return obj_id

        # Otherwise, assign a new ID
        new_id = len(self.id_to_obj)
        self.obj_to_id[obj] = new_id
        self.id_to_obj.append(obj)
        self.pass_through = False  # Disable pass-through after adding non-integer objects
        return new_id

    def get_id(self, obj: Any) -> Optional[int]:
        # If the object is an integer, return it as-is
        if not self.force_identify:
            if isinstance(obj, int):
                if not self.pass_through:
                    raise ValueError(f"Mixed types detected for {self.name}: {obj}")
                return obj
            elif isinstance(obj, np.integer):
                if not self.pass_through:
                    raise ValueError(f"Mixed types detected for {self.name}: {obj}")
                return int(obj)

        return self.obj_to_id.get(obj, None)

    def get(self, obj_id: int) -> Any:
        # If pass_through is enabled, return the object as-is if it is an integer
        if self.pass_through:
            return obj_id

        # Retrieve the object based on the ID if it exists
        if 0 <= obj_id < len(self.id_to_obj):
            return self.id_to_obj[obj_id]
        raise IdentifierError(self.name, obj_id)

    def get_or_default(self, obj_id: int, default: Optional[Any] = None) -> Any:
        # If pass_through is enabled, return the object as-is if it is an integer
        if self.pass_through:
            return obj_id

        # Retrieve the object based on the ID if it exists, otherwise return the default value
        if 0 <= obj_id < len(self.id_to_obj):
            return self.id_to_obj[obj_id]
        return default

    def __getitem__(self, obj_id: int) -> Any:
        """Allow indexing to access objects by their ID."""
        return self.get(obj_id)  # Use the existing get method for retrieval
