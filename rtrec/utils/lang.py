from typing import Callable, Dict, Any

def extract_func_args(func: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts arguments for a function from a dictionary.

    Args:
        func (Callable): The function whose arguments to extract.
        kwargs (Dict[str, Any]): A dictionary containing potential arguments.

    Returns:
        Dict[str, Any]: A dictionary of arguments matching the function's parameter names.
    """
    func_arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
    return {key: kwargs[key] for key in func_arg_names if key in kwargs}
