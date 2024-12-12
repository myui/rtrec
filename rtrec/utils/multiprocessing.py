from multiprocessing import shared_memory
import numpy as np

def create_shared_array(array: np.ndarray) -> shared_memory.SharedMemory:
    """
    Create a shared memory segment for a NumPy array and copy the array's data into it.

    Args:
        array (np.ndarray): The NumPy array to share.

    Returns:
        shared_memory.SharedMemory: The shared memory object containing the array's data.
    """
    # Create a shared memory segment
    shm = shared_memory.SharedMemory(create=True, size=array.nbytes)

    # Create a NumPy array backed by the shared memory buffer and copy data into it
    shared_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    np.copyto(shared_array, array)

    return shm

