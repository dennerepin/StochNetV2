import gillespy
import numpy as np

from stochnet_v2.CRN_models.base import BaseCRNModel


class Bees(BaseCRNModel):

    def __init__(self, endtime, timestep):

        super().__init__(
            endtime=endtime,
            timestep=timestep,
            model_name="Gene",
        )

        N = 100
        F_init_val = 0

        # stinging_frac = 0.85
        # A_init_val = int(N * stinging_frac)
        A_init_val = 50
        NA_init_val = 50

        Bee = gillespy.Species(name='Bee', initial_value=N)
        BeeA = gillespy.Species(name='BeeA', initial_value=0)
        BeeNA = gillespy.Species(name='BeeNA', initial_value=0)
        BeeD = gillespy.Species(name='BeeD', initial_value=0)
        A = gillespy.Species(name='A', initial_value=A_init_val)
        NA = gillespy.Species(name='NA', initial_value=NA_init_val)
        F = gillespy.Species(name='F', initial_value=F_init_val)

        self.add_species([
            Bee,
            BeeA,
            BeeNA,
            BeeD,
            A,
            NA,
            F,
        ])

        become_aggressive_rate = gillespy.Parameter(name='become_aggressive_rate', expression=0.0005)
        become_non_aggressive_rate = gillespy.Parameter(name='become_non_aggressive_rate', expression=0.0002)
        calm_down_rate = gillespy.Parameter(name='calm_down_rate', expression=0.01)
        stinging_rate = gillespy.Parameter(name='stinging_rate', expression=0.0001)
        f_stinging_rate = gillespy.Parameter(name='f_stinging_rate', expression=0.005)
        f_degradation_rate = gillespy.Parameter(name='f_degradation_rate', expression=0.01)

        self.add_parameter([
            become_aggressive_rate,
            become_non_aggressive_rate,
            calm_down_rate,
            stinging_rate,
            f_stinging_rate,
            f_degradation_rate,
        ])

        become_aggressive = gillespy.Reaction(
            name='become_aggressive',
            reactants={Bee: 1, A: 1},
            products={BeeA: 1},
            rate=become_aggressive_rate,
        )

        become_non_aggressive = gillespy.Reaction(
            name='become_non_aggressive',
            reactants={Bee: 1, NA: 1},
            products={BeeNA: 1},
            rate=become_non_aggressive_rate,
        )

        calm_down = gillespy.Reaction(
            name='calm_down',
            reactants={BeeA: 1},
            products={Bee: 1, A: 1},
            rate=calm_down_rate,
        )

        stinging = gillespy.Reaction(
            name='stinging',
            reactants={BeeA: 1},
            products={BeeD: 1, F: 1},
            rate=stinging_rate,
        )

        f_stinging = gillespy.Reaction(
            name='f_stinging',
            reactants={BeeA: 1, F: 1},
            products={BeeD: 1, F: 2},
            rate=f_stinging_rate,
        )

        f_degradation = gillespy.Reaction(
            name='f_degradation',
            reactants={F: 1},
            products={},
            rate=f_degradation_rate,
        )

        self.add_reaction([
            become_aggressive,
            become_non_aggressive,
            calm_down,
            stinging,
            f_stinging,
            f_degradation,
        ])

    @staticmethod
    def get_species_names():
        return [
            'Bee',
            'BeeA',
            'BeeNA',
            'BeeD',
            'A',
            'NA',
            'F',
        ]

    @staticmethod
    def get_initial_state():
        return [100, 0, 0, 0, 50, 50, 10]

    @classmethod
    def get_n_species(cls):
        return len(cls.get_species_names())

    @classmethod
    def get_initial_settings(cls, n_settings, sigm=0.5):
        n_species = cls.get_n_species()
        initial_state = cls.get_initial_state()
        settings = np.zeros((n_settings, n_species))

        for i in range(n_species):
            if i in [1, 2]:
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


class Bees_(BaseCRNModel):

    def __init__(self, endtime, timestep):

        super().__init__(
            endtime=endtime,
            timestep=timestep,
            model_name="Gene",
        )

        N = 100
        F_init_val = 0

        # stinging_frac = 0.85
        # A_init_val = int(N * stinging_frac)
        A_init_val = 50

        Bee = gillespy.Species(name='Bee', initial_value=N)
        BeeA = gillespy.Species(name='BeeA', initial_value=0)
        BeeD = gillespy.Species(name='BeeD', initial_value=0)
        A = gillespy.Species(name='A', initial_value=A_init_val)
        F = gillespy.Species(name='F', initial_value=F_init_val)

        self.add_species([
            Bee,
            BeeA,
            BeeD,
            A,
            F,
        ])

        become_aggressive_rate = gillespy.Parameter(name='become_aggressive_rate', expression=0.001)
        # become_aggressive_f_rate = gillespy.Parameter(name='become_aggressive_f_rate', expression=0.01)
        stinging_rate = gillespy.Parameter(name='stinging_rate', expression=0.0001)
        f_stinging_rate = gillespy.Parameter(name='f_stinging_rate', expression=0.005)
        calm_down_rate = gillespy.Parameter(name='calm_down_rate', expression=0.01)
        f_degradation_rate = gillespy.Parameter(name='f_degradation_rate', expression=0.01)

        self.add_parameter([
            become_aggressive_rate,
            # become_aggressive_f_rate,
            calm_down_rate,
            stinging_rate,
            f_stinging_rate,
            f_degradation_rate,
        ])

        become_aggressive = gillespy.Reaction(
            name='become_aggressive',
            reactants={Bee: 1, A: 1},
            products={BeeA: 1},
            rate=become_aggressive_rate,
        )

        # become_aggressive_f = gillespy.Reaction(
        #     name='become_aggressive_f',
        #     reactants={Bee: 1, A: 1, F: 1},
        #     products={BeeA: 1, F: 1},
        #     propensity_function='become_aggressive_f_rate * (Bee * A * F) / (Bee + BeeA + BeeD + A + F)',
        # )

        calm_down = gillespy.Reaction(
            name='calm_down',
            reactants={BeeA: 1},
            products={Bee: 1, A: 1},
            rate=calm_down_rate,
        )

        stinging = gillespy.Reaction(
            name='stinging',
            reactants={BeeA: 1},
            products={BeeD: 1, F: 1},
            rate=stinging_rate,
        )

        f_stinging = gillespy.Reaction(
            name='f_stinging',
            reactants={BeeA: 1, F: 1},
            products={BeeD: 1, F: 2},
            rate=f_stinging_rate,
        )

        f_degradation = gillespy.Reaction(
            name='f_degradation',
            reactants={F: 1},
            products={},
            rate=f_degradation_rate,
        )

        self.add_reaction([
            become_aggressive,
            # become_aggressive_f,
            calm_down,
            stinging,
            f_stinging,
            f_degradation,
        ])

    @staticmethod
    def get_species_names():
        return [
            'Bee',
            'BeeA',
            'BeeD',
            'A',
            'F',
        ]

    @staticmethod
    def get_initial_state():
        return [100, 0, 0, 50, 10]

    @classmethod
    def get_n_species(cls):
        return len(cls.get_species_names())

    @classmethod
    def get_initial_settings(cls, n_settings, sigm=0.5):
        n_species = cls.get_n_species()
        initial_state = cls.get_initial_state()
        settings = np.zeros((n_settings, n_species))

        for i in range(n_species):
            if i in [1, 2]:
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
