# import gillespy
import gillespy2 as gillespy
import numpy as np

from stochnet_v2.CRN_models.base import BaseCRNModel


class LotkaVolterra(BaseCRNModel):
    """
    Class for Lotka-Volterra model.
    """

    params = {
        'k1': 15.0,
        'k2': 0.15,
        'k3': 5.0,
    }

    def __init__(
            self,
            endtime,
            timestep
    ):
        """
        Initialize model.

        Parameters
        ----------
        endtime : simulation endtime
        timestep : simulation time-step
        """
        super().__init__(
            endtime=endtime,
            timestep=timestep,
            model_name="Lotka-Volterra",
        )

        X = gillespy.Species(name='X', initial_value=100)
        Y = gillespy.Species(name='Y', initial_value=100)

        self.add_species([X, Y])

        k1 = gillespy.Parameter(name='k1', expression=self.params['k1'])
        k2 = gillespy.Parameter(name='k2', expression=self.params['k2'])
        k3 = gillespy.Parameter(name='k3', expression=self.params['k3'])

        self.add_parameter([k1, k2, k3])

        r1 = gillespy.Reaction(
            name='r1',
            reactants={X: 1},
            products={X: 2},
            rate=k1,
        )

        r2 = gillespy.Reaction(
            name='r2',
            reactants={X: 1, Y: 1},
            products={Y: 2},
            rate=k2,
        )

        r3 = gillespy.Reaction(
            name='r3',
            reactants={Y: 1},
            products={},
            rate=k3,
        )

        self.add_reaction([r1, r2, r3])

    @staticmethod
    def get_species_names():
        """Returns list of all species names."""
        return ['X', 'Y']

    @staticmethod
    def get_initial_state():
        """Returns list of species initial values."""
        return [100, 100]

    @classmethod
    def get_initial_settings(cls, n_settings, sigm=0.7):
        """
        Generate a set of random initial states.
        Parameters
        ----------
        n_settings : number of initial states to generate
        sigm : float parameter to set the upper bound for sampling species initial value:
            - lower bound is set as `0.1 * val`,
            - upper as `val + int(val * sigm)`,
            where val is the species initial value returned by `get_initial_state` method.

        Returns
        -------
        settings : array of initial settings (states) of size (n_settings, n_species)

        """
        n_species = cls.get_n_species()
        initial_state = cls.get_initial_state()
        settings = np.zeros((n_settings, n_species))

        for i in range(n_species):
            val = initial_state[i]
            if val == 0:
                low = 0
                high = 1
            else:
                low = int(val * 0.5)
                high = val + int(val * sigm)
            settings[:, i] = np.random.randint(low, high, n_settings)
        return settings

    @classmethod
    def get_histogram_bounds(cls, species_names_list=None):
        """
        Returns bounds for species histograms.

        Parameters
        ----------
        species_names_list: list of species to produce bounds (optional)

        Returns
        -------
        histogram_bounds: list of [min, max] values for species selected either by
            species_names_list or get_species_for_histogram method

        """
        n_species_for_histogram = len(cls.get_species_for_histogram())
        histogram_bounds = [[0, 300] * n_species_for_histogram]
        return histogram_bounds

    @staticmethod
    def get_species_for_histogram():
        """Returns list of species to create histograms for evaluation"""
        return ['X', 'Y']