# import gillespy
import gillespy2 as gillespy
import numpy as np
import math

from stochnet_v2.CRN_models.base import BaseCRNModel


class LST(BaseCRNModel):
    """Class for LST model."""

    def __init__(self, endtime, timestep):
        """
        Initialize the model.

        Parameters
        ----------
        endtime : endtime of simulations
        timestep : time-step of simulations
        """
        super().__init__(
            endtime=endtime,
            timestep=timestep,
            model_name="LST",
        )

        self.N = 10 ** 9
        on_rnap_rate = 10 ** 4 / self.N

        on_rnap = gillespy.Parameter(name='on_rnap', expression=on_rnap_rate)  # 10 ** 4 / self.N
        off_rnap = gillespy.Parameter(name='off_rnap', expression=0.016)  # 0.016 !0.0005!
        on_rnap_if_a = gillespy.Parameter(name='on_rnap_if_a', expression=49 * on_rnap_rate)
        on_r = gillespy.Parameter(name='on_r', expression=8.8 * 10 ** 7 / self.N)  # 8.8 * 10 ** 7 / self.N
        off_r = gillespy.Parameter(name='off_r', expression=0.016)  # 0.016 !0.001!
        on_a = gillespy.Parameter(name='on_a', expression=8.8 * 10 ** 7 / self.N)
        off_a = gillespy.Parameter(name='off_a', expression=0.0264)  # 0.0264
        p_on = gillespy.Parameter(name='p_on', expression=0.5)  # 0.5
        p_off = gillespy.Parameter(name='p_off', expression=0.001)  # 0.001

        self.add_parameter([on_rnap, off_rnap, on_rnap_if_a, on_r, off_r, on_a, off_a, p_on, p_off])

        # Species
        DFU = gillespy.Species(name='DFU', initial_value=1)
        DFB = gillespy.Species(name='DFB', initial_value=0)
        DFR = gillespy.Species(name='DFR', initial_value=0)
        DAU = gillespy.Species(name='DAU', initial_value=0)
        DAB = gillespy.Species(name='DAB', initial_value=0)
        DAR = gillespy.Species(name='DAR', initial_value=0)
        RNAP = gillespy.Species(name='RNAP', initial_value=1500)
        P = gillespy.Species(name='P', initial_value=240)
        A = gillespy.Species(name='A', initial_value=275)
        R = gillespy.Species(name='R', initial_value=10)

        self.add_species([DFU, DFB, DFR, DAU, DAB, DAR, RNAP, P, A, R])

        # ACTIVATION
        act_bind_u = gillespy.Reaction(
            name='act_bind_u',
            reactants={DFU: 1, A: 1},
            products={DAU: 1},
            rate=on_a,
        )
        act_unbind_u = gillespy.Reaction(
            name='act_unbind_u',
            reactants={DAU: 1},
            products={DFU: 1, A: 1},
            rate=off_a,
        )

        act_bind_b = gillespy.Reaction(
            name='act_bind_b',
            reactants={DFB: 1, A: 1},
            products={DAB: 1},
            rate=on_a,
        )
        act_unbind_b = gillespy.Reaction(
            name='act_unbind_b',
            reactants={DAB: 1},
            products={DFB: 1, A: 1},
            rate=off_a,
        )

        act_bind_r = gillespy.Reaction(
            name='act_bind_r',
            reactants={DFR: 1, A: 1},
            products={DAR: 1},
            rate=on_a,
        )
        act_unbind_r = gillespy.Reaction(
            name='act_unbind_r',
            reactants={DAR: 1},
            products={DFR: 1, A: 1},
            rate=off_a,
        )

        # INHIBITION
        r_bind_f = gillespy.Reaction(
            name='r_bind_f',
            reactants={DFU: 1, R: 1},
            products={DFR: 1},
            rate=on_r,
        )
        r_unbind_f = gillespy.Reaction(
            name='r_unbind_f',
            reactants={DFR: 1},
            products={DFU: 1, R: 1},
            rate=off_r,
        )

        r_bind_a = gillespy.Reaction(
            name='r_bind_a',
            reactants={DAU: 1, R: 1},
            products={DAR: 1},
            rate=on_r,
        )
        r_unbind_a = gillespy.Reaction(
            name='r_unbind_a',
            reactants={DAR: 1},
            products={DAU: 1, R: 1},
            rate=off_r,
        )

        # POLYMERASE
        rnap_bind_f = gillespy.Reaction(
            name='rnap_bind_f',
            reactants={DFU: 1, RNAP: 1},
            products={DFB: 1},
            rate=on_rnap,
        )
        rnap_unbind_f = gillespy.Reaction(
            name='rnap_bind_f',
            reactants={DFB: 1},
            products={DFU: 1, RNAP: 1},
            rate=off_rnap,
        )

        rnap_bind_a = gillespy.Reaction(
            name='rnap_bind_a',
            reactants={DAU: 1, RNAP: 1},
            products={DAB: 1},
            rate=on_rnap_if_a,
        )
        rnap_unbind_a = gillespy.Reaction(
            name='rnap_bind_f',
            reactants={DAB: 1},
            products={DAU: 1, RNAP: 1},
            rate=off_rnap,
        )

        # PROTEIN
        p_express_f = gillespy.Reaction(
            name='p_express',
            reactants={DFB: 1},
            products={DFB: 1, P: 1},
            rate=p_on,
        )
        p_express_a = gillespy.Reaction(
            name='p_express',
            reactants={DAB: 1},
            products={DAB: 1, P: 1},
            rate=p_on,
        )
        p_dissolve = gillespy.Reaction(
            name='p_dissolve',
            reactants={P: 1},
            products={},
            rate=p_off,
        )

        self.add_reaction([
            act_bind_u,
            act_bind_b,
            act_unbind_u,
            act_unbind_b,
            act_bind_r,
            act_unbind_r,
            r_bind_f,
            r_bind_a,
            r_unbind_f,
            r_unbind_a,
            rnap_bind_f,
            rnap_bind_a,
            rnap_unbind_f,
            rnap_unbind_a,
            p_express_f,
            p_express_a,
            p_dissolve,
        ])

        nb_of_steps = int(math.ceil((endtime / timestep))) + 1
        self.timespan(np.linspace(0, endtime, nb_of_steps))

    @staticmethod
    def get_species_names():
        """
        Returns list of species names.

        Returns
        -------
        list of all species names. The order of names should be coherent
        with the list returned by get_initial_state method.

        """
        return ['DFU', 'DFB', 'DFR', 'DAU', 'DAB', 'DAR', 'RNAP', 'P', 'A', 'R']

    @staticmethod
    def get_initial_state():
        """
        Returns default initial state.

        Returns
        -------
        list of species initial values. The order of values should be coherent
        with the list returned by get_species_names method.

        """
        return [1, 0, 0, 0, 0, 0, 1500, 240, 275, 10]

    @classmethod
    def get_initial_settings(cls, n_settings, sigm=1.0):
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
            if i == 0:
                settings[:, i] = 1.
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
        """
        Returns bounds for species histograms.

        Parameters
        ----------
        species_names_list: not used

        Returns
        -------
        histogram_bounds: list of [min, max] values for species

        """
        n_species_for_histogram = len(cls.get_species_for_histogram())
        histogram_bounds = [[0.5, 1800.5]] * n_species_for_histogram
        return histogram_bounds

    @staticmethod
    def get_species_for_histogram():
        """Returns list of species to create histograms for evaluation"""
        return ['P']
