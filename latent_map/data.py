import numpy as np
from typing import List


class Data_Factory:
    """
    Collections of different objective functions
    """

    def convex_1(self, dim: int = 3, num: int = 1000) -> np:
        """
        simple d-dim convex function
        """
        if dim == 1:
            self.config = np.random.normal(loc=0, scale=1, size=[num, dim])
        elif dim > 1:
            self.config = np.random.multivariate_normal(
                mean=np.zeros(dim), cov=np.eye(dim), size=num
            )
        self.target_value = np.sum(self.config ** 2, axis=1)[:, np.newaxis]
        self.data = np.hstack([self.config, self.target_value])
        return self.data

    @staticmethod
    def nearest(np_data, point: List) -> np:
        """
        Tool to find nearest data point
        """
        length = np.size(point)
        diff = np.abs(np_data[:, :length] - np.array(point))
        return np_data[np.argmin(np.sum(diff, axis=1))]

    def obj_func(self, test_p: list) -> float:
        """For query druing optimization"""
        input_dim = np.shape(test_p)[0]
        if input_dim != np.size(test_p) and input_dim > 1:
            value = [
                Data_Factory.nearest(self.data, test_p[i, :])[-1]
                for i in range(input_dim)
            ]
            return value
        else:
            data_point = Data_Factory.nearest(self.data, test_p)
            return data_point[-1]

    def plot_1Dsample(self):
        plt.scatter(self.data[:, 0], self.data[:, -1])
        plt.title("1-D Demo of Original Data")
        plt.xlabel("Config")
        plt.ylabel("Target Value")
