#
from typing import List
from llmsat.llmsat import AlgorithmResult, AlgorithmStatus, get_logger
from llmsat.utils.aws import get_algorithm_result_of_status

@dataclass
class CoderConfig:
    algorithm_response_path: str
    code_path: str

class CoderCls:
    def __init__(self, config: CoderConfig):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

    def generate_code(self, algorithm: str) -> str:
        pass

    def get_algorithms(self) -> List[str]:
        algorithms = get_algorithm_result_of_status(AlgorithmStatus.Evaluating)
        return algorithms

def main():
    pass

if __name__ == "__main__":
    main()