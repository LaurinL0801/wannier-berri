from .w90file import W90_file
import numpy as np


class MOM(W90_file):
    def __init__(self, seedname="wannier90", **kwargs):
        self.npz_tags = ["data"]
        self.seedname = seedname
        super().__init__(seedname=seedname, ext="mom", **kwargs)

    @property
    def NK(self):
        return self.data.shape[0]

    @property
    def NB(self):
        return self.data.shape[1]

    @property
    def NW(self):
        return self.data.shape[2]

    def from_w90_file(self, seedname):
        data = np.loadtxt(seedname + ".p")
        NK = int(data[:, 2].max())
        NB = int(data[:, 1].max())
        pmat = np.array(
            [data[:, 3 + 2 * i] + 1j * data[:, 4 + 2 * i] for i in range(3)]
        ).T
        data = pmat.reshape(NK, NB, NB, 3).transpose((0, 2, 1, 3))
        self.data = data

    def to_w90_file(self, seedname):
        file = open(seedname + ".p", "w")
        for ik in range(self.NK):
            for ib1 in range(self.NB):
                for ib2 in range(self.NB):
                    file.write(f" {ib1 + 1:4d} {ib2 + 1:4d} {ik + 1:4d}")

                    for i in range(3):
                        file.write(f" {self.data[ik, ib1, ib2, i] :17.12f}")
                    file.write(f"\n")

    def apply_window(self, selected_bands):
        if selected_bands is not None:
            self.data = self.data[:, selected_bands]
