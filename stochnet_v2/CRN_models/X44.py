# import gillespy
import gillespy2 as gillespy
import numpy as np

from stochnet_v2.CRN_models.base import BaseCRNModel


class X44(BaseCRNModel):
    """
    Class for (44) model defined in https://arxiv.org/pdf/1801.09200.pdf.
    For training: timestep=50.0, endtime=500.
    """

    params = {
        'a11': 100.0,
        'a21': 50.0,
        'a31': 1.0,
        'a32': 1.0,
        'b1': 1.0,
        'b2': 1.0,
        'b3': 1.0,
        'b4': 50.0,  # 50.0 or 200.0
        'b5': 1.0,
        'b6': 100.0,
        'gamma12': 1.0,
        'gamma21': 20.0,
        'gamma23': 2.0,
        'gamma32': 1.0,
    }

    def __init__(
            self,
            endtime,
            timestep,
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
            model_name="X44",
        )

        G1 = gillespy.Species(name='G1', initial_value=1)
        G2 = gillespy.Species(name='G2', initial_value=1)
        G3 = gillespy.Species(name='G3', initial_value=0)
        P1 = gillespy.Species(name='P1', initial_value=100)
        P2 = gillespy.Species(name='P2', initial_value=100)
        P3 = gillespy.Species(name='P3', initial_value=0)
        P4 = gillespy.Species(name='P4', initial_value=0)

        self.add_species([G1, G2, G3, P1, P2, P3, P4])

        a11 = gillespy.Parameter(name='a11', expression=self.params['a11'])
        a21 = gillespy.Parameter(name='a21', expression=self.params['a21'])
        a31 = gillespy.Parameter(name='a31', expression=self.params['a31'])
        a32 = gillespy.Parameter(name='a32', expression=self.params['a32'])
        b1 = gillespy.Parameter(name='b1', expression=self.params['b1'])
        b2 = gillespy.Parameter(name='b2', expression=self.params['b2'])
        b3 = gillespy.Parameter(name='b3', expression=self.params['b3'])
        b4 = gillespy.Parameter(name='b4', expression=self.params['b4'])
        b5 = gillespy.Parameter(name='b5', expression=self.params['b5'])
        b6 = gillespy.Parameter(name='b6', expression=self.params['b6'])

        epsilon = gillespy.Parameter(name='epsilon', expression=0.001)

        gamma12 = gillespy.Parameter(name='gamma12', expression=f'epsilon * {self.params["gamma12"]}')
        gamma21 = gillespy.Parameter(name='gamma21', expression=f'epsilon * {self.params["gamma21"]}')
        gamma23 = gillespy.Parameter(name='gamma23', expression=f'epsilon * {self.params["gamma23"]}')
        gamma32 = gillespy.Parameter(name='gamma32', expression=f'epsilon * {self.params["gamma32"]}')

        self.add_parameter([a11, a21, a31, a32, b1, b2, b3, b4, b5, b6, epsilon, gamma12, gamma21, gamma23, gamma32])

        r1 = gillespy.Reaction(
            name='r1',
            reactants={G1: 1},
            products={G1: 1, P1: 1},
            rate=a11,
        )

        r2 = gillespy.Reaction(
            name='r2',
            reactants={G2: 1},
            products={G2: 1, P2: 1},
            rate=a21,
        )

        r3 = gillespy.Reaction(
            name='r3',
            reactants={G3: 1, P1: 1},
            products={G3: 1},
            rate=a31,
        )

        r4 = gillespy.Reaction(
            name='r4',
            reactants={G3: 1, P2: 1},
            products={G3: 1},
            rate=a32,
        )

        r5 = gillespy.Reaction(
            name='r5',
            reactants={P1: 1},
            products={},
            rate=b1,
        )

        r6 = gillespy.Reaction(
            name='r6',
            reactants={P2: 1},
            products={},
            rate=b2,
        )

        r7 = gillespy.Reaction(
            name='r7',
            reactants={P1: 1, P2: 1},
            products={P3: 1},
            rate=b3,
        )

        r7b = gillespy.Reaction(
            name='r7b',
            reactants={P3: 1},
            products={P1: 1, P2: 1},
            rate=b6,
        )

        r8 = gillespy.Reaction(
            name='r8',
            reactants={P3: 1},
            products={P2: 1, P4: 1},
            rate=b4,
        )

        r8b = gillespy.Reaction(
            name='r8b',
            reactants={P2: 1, P4: 1},
            products={P3: 1},
            rate=b5,
        )

        r9 = gillespy.Reaction(
            name='r9',
            reactants={G1: 1},
            products={G2: 1},
            rate=gamma12,
        )

        r9b = gillespy.Reaction(
            name='r9b',
            reactants={G2: 1},
            products={G1: 1},
            rate=gamma21,
        )

        r10 = gillespy.Reaction(
            name='r10',
            reactants={G2: 1},
            products={G3: 1},
            rate=gamma23,
        )

        r10b = gillespy.Reaction(
            name='r10b',
            reactants={G3: 1},
            products={G2: 1},
            rate=gamma32,
        )

        self.add_reaction([r1, r2, r3, r4, r5, r6, r7, r7b, r8, r8b, r9, r9b, r10, r10b])

    @staticmethod
    def get_species_names():
        """Returns list of all species names."""
        return ['G1', 'G2', 'G3', 'P1', 'P2', 'P3', 'P4']

    @staticmethod
    def get_initial_state():
        """Returns list of species initial values."""
        return [1, 1, 0, 200, 200, 0, 0]

    @staticmethod
    def _conservation_settings(size, n_settings, conservation_constant):
        settings = np.zeros((n_settings, size))
        a = np.arange(n_settings)
        for _ in range(conservation_constant):
            idxs = np.random.randint(0, size, n_settings)
            settings[(a, idxs)] += 1
        return settings

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

        for i in range(3, n_species):
            val = initial_state[i]
            if val == 0:
                low = 0
                high = 1
            else:
                low = int(val * 0.1)
                high = val + int(val * sigm)
            settings[:, i] = np.random.randint(low, high, n_settings)
        settings[:, :3] = cls._conservation_settings(3, n_settings, conservation_constant)

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
        return ['P4']