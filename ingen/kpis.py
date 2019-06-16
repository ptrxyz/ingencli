import numpy as np
import dask.bag as daskb


class KPIs():
    def __init__(self, real_data, generated_data, binning):
        """real_data is the data source of the original data,
           generated_data is the data source of the newly generated data.
           binning is the binning the generated data is based on"""

        self.__rd = real_data
        self.__gd = generated_data
        self.__binning = binning

        hist1 = real_data.get_histogram(binning)
        hist2 = generated_data.get_histogram(binning)

        self.__h1 = hist1           # Model Source, realH
        self.__h2 = hist2           # ndH, generated

        self.__h1f = hist1.values.flatten()
        self.__h2f = hist2.values.flatten()

        self.__diff = (self.__h2f - self.__h1f)

    def error(self):
        # bins with too many generated packages
        return sum(self.__diff[self.__diff > 0]) / sum(self.__h2f)

    def quality(self):
        def get_quality(r, delta):
            if (r == 0 and delta == 0):
                return 1
            if (r == 0 or abs(delta) > r):
                return 0
            return 1 - abs(delta) / r

        def quality(index):
            if not any(index):
                return 2.0
            else:
                bin_vols = self.__h2.binning.volumes.flatten()
                pairs = zip(self.__h1f[index],
                            self.__diff[index],
                            bin_vols[index])
                return sum([get_quality(r, delta) * vol
                            for r, delta, vol in pairs]) / \
                    bin_vols[index].sum()

        abi = [True] * self.__h1f.size
        nebi = self.__h1f > 0
        ebi = self.__h1f == 0

        return (quality(abi), quality(nebi), quality(ebi))

    def distance(self):
        def _half(data_point, data_set):
            return np.linalg.norm(data_set - data_point, axis=1).mean()

        def apply_to_chunk(chunk, other_set):
            res = 0
            for data_point in chunk:
                res += _half(data_point, other_set)
            return res

        def split_set_and_apply(set1, set2):
            bag = daskb.from_sequence(
                np.array_split(set1, 16))
            return sum(bag.map(apply_to_chunk, other_set=set2).compute())

        def set2set(set1, set2):
            s1 = split_set_and_apply(set1, set2)
            s2 = split_set_and_apply(set2, set1)
            return (s1 + s2) / (set1.shape[0] + set2.shape[0])

        return set2set(self.__rd(normalized=True), self.__gd(normalized=True))
