import unittest

from matplotlib import pyplot as plt

from cv_geoguessr.grid.grid_partitioning import parse_boundary_csv, Boundary, Partitioning

LONDON_FILE_NAME = 'data/london.csv'


class TestStringMethods(unittest.TestCase):
    def test_boundary_csv_parsing(self):
        london_boundary = {
            'max_lat': 51.56709377706768,
            'max_lng': -0.007381439208984376,
            'min_lat': 51.46090592316693,
            'min_lng': -0.22212982177734378,
            'verts': [(-0.17337799072265625, 51.48218467930003),
                      (-0.1653099060058594, 51.47341775085057),
                      (-0.1600742340087891, 51.46122678211958),
                      (-0.13887405395507815, 51.46090592316693),
                      (-0.12239456176757814, 51.47213464436686),
                      (-0.10059356689453126, 51.49330128672925),
                      (-0.0854015350341797, 51.493835670725524),
                      (-0.07355690002441408, 51.50900956106959),
                      (-0.007381439208984376, 51.51119974058721),
                      (-0.010471343994140625, 51.52003951689637),
                      (-0.011029243469238283, 51.52690184594944),
                      (-0.013604164123535156, 51.52938484639296),
                      (-0.02454757690429688, 51.53507120815272),
                      (-0.02669334411621094, 51.54115723084824),
                      (-0.03313064575195313, 51.54516074932728),
                      (-0.03793716430664063, 51.54911054234792),
                      (-0.05596160888671876, 51.55823645513434),
                      (-0.07295608520507814, 51.56522641296103),
                      (-0.07844924926757812, 51.56709377706768),
                      (-0.09922027587890625, 51.56143809589396),
                      (-0.11681556701660158, 51.55615526777012),
                      (-0.12368202209472658, 51.55316672962128),
                      (-0.14651298522949222, 51.53603221320984),
                      (-0.14513969421386722, 51.525353239477596),
                      (-0.15586853027343753, 51.5231103367263),
                      (-0.1762962341308594, 51.53619237874674),
                      (-0.19260406494140625, 51.53693981046692),
                      (-0.22212982177734378, 51.53026587851656),
                      (-0.22084236145019534, 51.51525930707282),
                      (-0.21663665771484378, 51.50452203516731),
                      (-0.2003288269042969, 51.492713457097935),
                      (-0.18565177917480472, 51.48341405282735),
                      (-0.18161773681640625, 51.47993965082005)]
        }

        read_boundary = parse_boundary_csv(LONDON_FILE_NAME)
        print(read_boundary)

        self.assertEqual(london_boundary, read_boundary)

    def test_plot_boundary(self):
        boundary = Boundary(LONDON_FILE_NAME)
        boundary.plot()
        plt.show()

    def test_plot_partitioning(self):
        partitions = Partitioning(LONDON_FILE_NAME, 0.02)
        partitions.plot()
        plt.show()

    def test_one_hot(self):
        partitions = Partitioning(LONDON_FILE_NAME, 0.02)

        one_hot = partitions.one_hot((-0.1331, 51.505))
        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.assertEqual(expected, one_hot)

    def test_voronoi(self):
        plt.figure(figsize=(12,8))
        partitions = Partitioning(LONDON_FILE_NAME, 0.02, voronoi=False)
        partitions.plot()
        # plt.ylim((51.45, 51.57))
        # plt.xlim((-0.25, 0))
        plt.show()


    def test_vis(self):
        import numpy as np
        partitions = Partitioning(LONDON_FILE_NAME, 0.02, voronoi=True)
        partitions.plot_prediction(np.random.uniform(size=60))

        # print(len(partitions))
        # plt.ylim((51.45, 51.57))
        # plt.xlim((-0.25, 0))
        plt.show()



if __name__ == '__main__':
    unittest.main()
