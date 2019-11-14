import gillespy
import numpy as np

from stochnet_v2.CRN_models.base import BaseCRNModel


class Bees(BaseCRNModel):

    def __init__(
            self,
            endtime,
            timestep,
            n_bees=10,
            non_aggressive_frac=0.3,
    ):

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

        stinging_rate = gillespy.Parameter(name='stinging_rate', expression=0.01)  # 0.01,  0.03
        p_stinging_rate = gillespy.Parameter(name='p_stinging_rate', expression=0.04)  # 0.03
        become_aggressive_rate = gillespy.Parameter(name='become_aggressive_rate', expression=0.015)  # 0.01
        p_degradation_rate = gillespy.Parameter(name='p_degradation_rate', expression=0.15)  # 0.15
        exp = gillespy.Parameter(name='exp', expression=2.71828)
        L = gillespy.Parameter(name='L', expression=5.0)  # L/2 is the max
        s = gillespy.Parameter(name='s', expression=1.0)  # steepness

        self.add_parameter([
            stinging_rate,
            p_stinging_rate,
            become_aggressive_rate,
            p_degradation_rate,
            exp,
            L,
            s,
        ])

        stinging = gillespy.Reaction(
            name='stinging',
            reactants={BeeA: 1},
            products={BeeD: 1, P: 1},
            rate=stinging_rate,
        )

        p_stinging = gillespy.Reaction(
            name='p_stinging',
            reactants={BeeA: 1, P: 1},
            products={BeeD: 1, P: 2},
            rate=p_stinging_rate,
        )

        become_aggressive = gillespy.Reaction(
            name='become_aggressive',
            reactants={Bee: 1, P: 1},
            products={BeeA: 1, P: 1},
            propensity_function='become_aggressive_rate * Bee * (L / (1 + pow(exp,-s*P)) - L/2)',
            # rate=become_aggressive_rate,
        )

        f_degradation = gillespy.Reaction(
            name='f_degradation',
            reactants={P: 1},
            products={},
            rate=p_degradation_rate,
        )

        self.add_reaction([
            stinging,
            p_stinging,
            become_aggressive,
            f_degradation,
        ])

    @staticmethod
    def get_species_names():
        return [
            'Bee',
            'BeeA',
            'BeeD',
            'P',
        ]

    @staticmethod
    def get_initial_state():
        return [80, 20, 0, 10]

    @classmethod
    def get_n_species(cls):
        return len(cls.get_species_names())

    @classmethod
    def get_initial_settings(cls, n_settings, sigm=0.5):
        n_species = cls.get_n_species()
        initial_state = cls.get_initial_state()
        settings = np.zeros((n_settings, n_species))

        for i in range(n_species):
            if i in [3]:
                settings[:, i] = 0.
                continue
            val = initial_state[i]
            if val == 0:
                low = 0
                high = 1
            else:
                low = int(val * 0.1)
                high = val + int(val * sigm)
            settings[:, i] = np.random.randint(low, high, n_settings)
        return settings

    @classmethod
    def get_histogram_bounds(cls, species_names_list=None):
        n_species_for_histogram = len(cls.get_species_for_histogram())
        histogram_bounds = [[0.5, 18.5]] * n_species_for_histogram
        return histogram_bounds

    @staticmethod
    def get_species_for_histogram():
        return ['Bee', 'BeeA', 'BeeNA', 'BeeD']


class BeesMA(BaseCRNModel):

    def __init__(
            self,
            endtime,
            timestep,
            n_bees=100,
            non_aggressive_frac=0.2,
    ):

        super().__init__(
            endtime=endtime,
            timestep=timestep,
            model_name="Bees",
        )

        P_init_val = 100
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

        stinging_rate = gillespy.Parameter(name='stinging_rate', expression=0.01)  # 0.01,  0.03
        p_stinging_rate = gillespy.Parameter(name='p_stinging_rate', expression=0.04)  # 0.03
        become_aggressive_rate = gillespy.Parameter(name='become_aggressive_rate', expression=0.015)  # 0.01
        p_degradation_rate = gillespy.Parameter(name='p_degradation_rate', expression=0.15)  # 0.15

        self.add_parameter([
            stinging_rate,
            p_stinging_rate,
            become_aggressive_rate,
            p_degradation_rate,
        ])

        stinging = gillespy.Reaction(
            name='stinging',
            reactants={BeeA: 1},
            products={BeeD: 1, P: 1},
            rate=stinging_rate,
        )

        p_stinging = gillespy.Reaction(
            name='p_stinging',
            reactants={BeeA: 1, P: 1},
            products={BeeD: 1, P: 2},
            rate=p_stinging_rate,
        )

        become_aggressive = gillespy.Reaction(
            name='become_aggressive',
            reactants={Bee: 1, P: 1},
            products={BeeA: 1, P: 1},
            rate=become_aggressive_rate,
        )

        f_degradation = gillespy.Reaction(
            name='f_degradation',
            reactants={P: 1},
            products={},
            rate=p_degradation_rate,
        )

        self.add_reaction([
            stinging,
            p_stinging,
            become_aggressive,
            f_degradation,
        ])

    @staticmethod
    def get_species_names():
        return [
            'Bee',
            'BeeA',
            'BeeD',
            'P',
        ]

    @staticmethod
    def get_initial_state():
        return [80, 20, 0, 10]

    @classmethod
    def get_n_species(cls):
        return len(cls.get_species_names())

    @classmethod
    def get_initial_settings(cls, n_settings, sigm=0.5):
        n_species = cls.get_n_species()
        initial_state = cls.get_initial_state()
        settings = np.zeros((n_settings, n_species))

        for i in range(n_species):
            if i in [3]:
                settings[:, i] = 0.
                continue
            val = initial_state[i]
            if val == 0:
                low = 0
                high = 1
            else:
                low = int(val * 0.1)
                high = val + int(val * sigm)
            settings[:, i] = np.random.randint(low, high, n_settings)
        return settings

    @classmethod
    def get_histogram_bounds(cls, species_names_list=None):
        n_species_for_histogram = len(cls.get_species_for_histogram())
        histogram_bounds = [[0.5, 18.5]] * n_species_for_histogram
        return histogram_bounds

    @staticmethod
    def get_species_for_histogram():
        return ['Bee', 'BeeA', 'BeeNA', 'BeeD']
