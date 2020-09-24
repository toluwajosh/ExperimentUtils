"""
utils for generating reports
"""

import json
from typing import Dict


class ResultData(object):
    def __init__(self, method_name: str, save_path: str = "./results") -> None:
        self.__method_name = method_name
        self.__save_path = save_path

        # Default containers
        self.__metrics = {}
        # self.__categories = []
        self.__summaries = {}

    def add_result(self, metric: str, category: Dict) -> None:
        """Add a result category

        Args:
            metric (str): the name of the result metric
            category (Dict): a dictionary of category_name: score
        """
        cur_metric_results = self.__metrics.get(metric, [])
        cur_metric_results.append(category)
        self.__metrics.update({metric: cur_metric_results})

    def add_summary(self, metric: str, summary: Dict) -> None:
        """Add a summary and value

        Args:
            summary (Dict): a dictionary of summary: value
        """
        if metric in self.__metrics:
            cur_summary = self.__summaries.get(metric, {})
            cur_summary.update(summary)
            self.__summaries.update({metric: cur_summary})
        else:
            raise ValueError(f"No metric by {metric}")

    def to_dict(self):
        assert len(self.__summaries) == len(
            self.__metrics
        ), "Size of summaries and metrics must be equal"
        return {
            "method": self.__method_name,
            "metrics": [
                {"metric": metric, "results": category, "summary": summary}
                for (metric, category, summary) in zip(
                    self.__metrics.keys(),
                    self.__metrics.values(),
                    self.__summaries.values(),
                )
            ],
        }

    def to_json(self):
        json_file = f"{self.__save_path}/{self.__method_name}.json"
        with open(json_file, "w") as file:
            json.dump(self.to_dict(), file)

    def __str__(self):
        out_string = f"*** {self.__method_name} Results ***\n"
        for item in self.__metrics:
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
    result_data = ResultData("new_example_result_2", "examples/results")
    result_data.add_result("scorer", {"category": "category 1", "score": 0.01})
    result_data.add_result("scorer", {"category": "category 2", "score": 0.03})
    result_data.add_summary("scorer", {"average": 90})
    result_data.add_result("scorer2", {"category": "category 3", "score": 0.06})
    result_data.add_result("scorer2", {"category": "category 3", "score": 0.06})
    result_data.add_summary("scorer2", {"average": 80})
    result_data.to_json()
    # print(result_data.to_dict())
    # print(result_data)
