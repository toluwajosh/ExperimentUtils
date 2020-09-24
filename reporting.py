"""
utils for generating reports
"""

import json
from typing import Dict


class ResultData(object):
    __categories = []
    __summaries = {}

    def __init__(self, method_name: str, save_path: str = "./results") -> None:
        self.__method_name = method_name
        self.__save_path = save_path

    def add_category(self, category: Dict) -> None:
        """Add a result category

        Args:
            category (Dict): a dictionary of category_name: score
        """
        self.__categories.append(category)

    def add_summary(self, summary: Dict) -> None:
        """Add a summary and value

        Args:
            summary (Dict): a dictionary of summary: value
        """
        self.__summaries.update(summary)

    def to_dict(self):
        return {
            "method": self.__method_name,
            "results": self.__categories,
            "summary": self.__summaries,
        }

    def to_json(self):
        json_file = f"{self.__save_path}/{self.__method_name}.json"
        with open(json_file, "w") as file:
            json.dump(self.to_dict(), file)

    def __str__(self):
        out_string = f"*** {self.__method_name} Results ***\n"
        for item in self.__categories:
            out_string += f"{str(item)}\n"
        return out_string

    @property
    def method_name(self):
        print("getting method name")
        return self.__method_name

    @property
    def save_path(self):
        print("getting save path")
        return self.__save_path


if __name__ == "__main__":
    # example:
    result_data = ResultData("LogReg")
    result_data.add_category({"name": "category 1", "score": 0.00})
    result_data.add_category({"name": "category 2", "score": 0.01})
    result_data.add_category({"name": "category 3", "score": 0.02})
    result_data.add_summary({"average": 90})
    result_data.to_json()
    print(result_data)
