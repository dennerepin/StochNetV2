# import gillespy
import gillespy2 as gillespy
import numpy as np

from stochnet_v2.CRN_models.base import BaseCRNModel


class X47(BaseCRNModel):
    """
    Class for (47) model defined in https://arxiv.org/pdf/1801.09200.pdf.
    For training: timestep=100.0, endtime=1000.
    """

    params = {
        'a11': 100.0,
        'b1': 1.0,
        'b2': 1.0,
        'b3': 10.0,
        'b4': 1.0,
        'gamma12': 1.0,
        'gamma21': 1.5,
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
            model_name="X47",
        )

        G1 = gillespy.Species(name='G1', initial_value=1)
        G2 = gillespy.Species(name='G2', initial_value=1)
        P1 = gillespy.Species(name='P1', initial_value=100)
        P2 = gillespy.Species(name='P2', initial_value=100)

        self.add_species([G1, G2, P1, P2])

        a11 = gillespy.Parameter(name='a11', expression=self.params['a11'])
        b1 = gillespy.Parameter(name='b1', expression=self.params['b1'])
        b2 = gillespy.Parameter(name='b2', expression=self.params['b2'])
        b3 = gillespy.Parameter(name='b3', expression=self.params['b3'])
        b4 = gillespy.Parameter(name='b4', expression=self.params['b4'])

        epsilon = gillespy.Parameter(name='epsilon', expression=0.001)

        gamma12 = gillespy.Parameter(name='gamma12', expression=f'epsilon * {self.params["gamma12"]}')
        gamma21 = gillespy.Parameter(name='gamma21', expression=f'epsilon * {self.params["gamma21"]}')

        self.add_parameter([a11, b1, b2, b3, b4, epsilon, gamma12, gamma21])

        r1 = gillespy.Reaction(
            name='r1',
            reactants={G1: 1, P2: 2},
            products={G1: 1, P1: 1, P2: 1},
            propensity_function='a11 * G1 * P2 * (P2 - 1)',
        )

        r2 = gillespy.Reaction(
            name='r2',
            reactants={},
            products={P2: 1},
            rate=b1,
        )

        r2b = gillespy.Reaction(
            name='r2b',
            reactants={P2: 1},
            products={},
            rate=b2,
        )

        r3 = gillespy.Reaction(
            name='r3',
            reactants={P2: 1},
            products={P1: 1},
            rate=b3,
        )

        r4 = gillespy.Reaction(
            name='r4',
            reactants={P1: 1, P2: 2},
            products={P2: 3},
            propensity_function='b4 * P1 * P2 * (P2 - 1)',  # 'b4 * P1 * P2 * P2'
        )

        r5 = gillespy.Reaction(
            name='r5',
            reactants={G1: 1},
            products={G2: 1},
            rate=gamma12,
        )

        r5b = gillespy.Reaction(
            name='r5b',
            reactants={G2: 1},
            products={G1: 1},
            rate=gamma21,
        )

        self.add_reaction([r1, r2, r2b, r3, r4, r5, r5b])  # , r2, r2b, r3, r4, r5, r5b

    @staticmethod
    def get_species_names():
        """Returns list of all species names."""
        return ['G1', 'G2', 'P1', 'P2']

    @staticmethod
    def get_initial_state():
        """Returns list of species initial values."""
        return [1, 1, 100, 100]

    @classmethod
    def get_initial_settings(cls, n_settings, sigm=0.7, conservation_constant=2):
        """
        Generate a set of random initial states.
        Parameters
        ----------
        n_settings : number of initial states to generate
        sigm : float parameter to set the upper bound for sampling species initial value:
            - lower bound is set as `0.1 * val`,
            - upper as `val + int(val * sigm)`,
            where val is the species initial value returned by `get_initial_state` method.
        conservation_constant : conservation constant for catalysts G1 and G2.

        Returns
        -------
        settings : array of initial settings (states) of size (n_settings, n_species)

        """
        n_species = cls.get_n_species()
        initial_state = cls.get_initial_state()
        settings = np.zeros((n_settings, n_species))

        for i in range(2, n_species):
            val = initial_state[i]
            if val == 0:
                low = 0
                high = 1
            else:
                low = int(val * 0.1)
                high = val + int(val * sigm)
            settings[:, i] = np.random.randint(low, high, n_settings)
        x = np.random.randint(0, conservation_constant + 1, n_settings)
        settings[:, 0] = x
        settings[:, 1] = conservation_constant - x

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
        histogram_bounds = [[0, 500] * n_species_for_histogram]
        return histogram_bounds

    @staticmethod
    def get_species_for_histogram():
        """Returns list of species to create histograms for evaluation"""
        return ['P1', 'P2']