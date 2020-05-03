# import gillespy
import gillespy2 as gillespy
import numpy as np

from stochnet_v2.CRN_models.base import BaseCRNModel


class Bees(BaseCRNModel):
    """
    Class for bees stinging model. All reaction rates are mass-action,
    except the rate for reaction `become_aggressive`: the impact of pheromone (P)
    is expressed as sigmoid function with saturation level at L/2.
    """

    params = {
        'stinging_rate': 0.1,            # 0.01 0.01
        'p_stinging_rate': 0.4,          # 0.03 0.04
        'become_aggressive_rate': 0.125,   # 0.01 0.015
        'p_degradation_rate': 0.15,      # 0.15 0.15
        # 'calm_down_rate': 0.2,
    }

    def __init__(
            self,
            endtime,
            timestep,
            n_bees=10,
            non_aggressive_frac=0.3,
    ):
        """
        Initialize model.

        Parameters
        ----------
        endtime : simulation endtime
        timestep : simulation time-step
        n_bees : number of bees in population
        non_aggressive_frac : float, fraction of non-aggressive bees
        """
        super().__init__(
            endtime=endtime,
            timestep=timestep,
            model_name="Bees",
        )

        P_init_val = 0
        n_aggr = int(n_bees * (1 - non_aggressive_frac))
        n_non_aggr = n_bees - n_aggr

        Bee = gillespy.Species(name='Bee', initial_value=n_non_aggr)
        BeeA = gillespy.Species(name='BeeA', initial_value=n_aggr)
        BeeD = gillespy.Species(name='BeeD', initial_value=0)
        P = gillespy.Species(name='P', initial_value=P_init_val)

        self.add_species([
            Bee,
            BeeA,
            BeeD,
            P,
        ])

        stinging_rate = gillespy.Parameter(
            name='stinging_rate', expression=self.params['stinging_rate'])
        p_stinging_rate = gillespy.Parameter(
            name='p_stinging_rate', expression=self.params['p_stinging_rate'])
        become_aggressive_rate = gillespy.Parameter(
            name='become_aggressive_rate', expression=self.params['become_aggressive_rate'])
        p_degradation_rate = gillespy.Parameter(
            name='p_degradation_rate', expression=self.params['p_degradation_rate'])
        # calm_down_rate = gillespy.Parameter(
        #     name='calm_down_rate', expression=self.params['calm_down_rate'])

        exp = gillespy.Parameter(name='exp', expression=2.71828)
        L = gillespy.Parameter(name='L', expression=1.0)  # 5.0 L/2 is the max
        s = gillespy.Parameter(name='s', expression=0.2)  # 1.0 steepness

        self.add_parameter([
            stinging_rate,
            p_stinging_rate,
            become_aggressive_rate,
            p_degradation_rate,
            # calm_down_rate,
            exp,
            L,
            s,
        ])

        stinging = gillespy.Reaction(
            name='stinging',
            reactants={BeeA: 1},
            products={BeeD: 1, P: 2},
            rate=stinging_rate,
        )

        p_stinging = gillespy.Reaction(
            name='p_stinging',
            reactants={BeeA: 1, P: 1},
            products={BeeD: 1, P: 3},
            rate=p_stinging_rate,
        )

        become_aggressive = gillespy.Reaction(
            name='become_aggressive',
            reactants={Bee: 1, P: 1},
            products={BeeA: 1, P: 1},
            propensity_function='become_aggressive_rate * Bee * (L / (1 + pow(exp,-s*P)) - L/2)',
        )

        # calm_down = gillespy.Reaction(
        #     name='calm_down',
        #     reactants={BeeA: 1},
        #     products={Bee: 1},
        #     rate=calm_down_rate,
        # )

        p_degradation = gillespy.Reaction(
            name='p_degradation',
            reactants={P: 1},
            products={},
            rate=p_degradation_rate,
        )

        self.add_reaction([
            stinging,
            p_stinging,
            become_aggressive,
            p_degradation,
            # calm_down,
        ])

    @staticmethod
    def get_species_names():
        """Returns list of all species names."""
        return [
            'Bee',
            'BeeA',
            'BeeD',
            'P',
        ]

    @staticmethod
    def get_initial_state():
        """Returns list of species initial values."""
        return [20, 0, 0, 10]

    @classmethod
    def get_initial_settings(cls, n_settings, sigm=0.5):
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
            if i in [2]:
                settings[:, i] = 0.
                continue
            val = initial_state[i]
            if val == 0:
                low = 0
                high = 1
            else:
                # low = int(val * sigm)
                low = 0
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
        histogram_bounds = [[0.5, 18.5]] * n_species_for_histogram
        return histogram_bounds

    @staticmethod
    def get_species_for_histogram():
        """Returns list of species to create histograms for evaluation"""
        return ['Bee', 'BeeA', 'BeeNA', 'BeeD']
