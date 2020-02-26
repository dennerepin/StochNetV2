# import gillespy
import gillespy2 as gillespy
import numpy as np
import os
from stochnet_v2.CRN_models.base import BaseSBMLModel


class EGFR(BaseSBMLModel):
    """
    Class for epidermal growth-factor receptor (EGFR) reaction model of
    cellular signal transduction.
    """

    _hist_top_bound = 200

    def __init__(
            self,
            endtime,
            timestep,
            filename='../SBML_models/BIOMD0000000048_url.xml'
    ):
        """
        Initialize model.

        Parameters
        ----------
        endtime : simulation endtime
        timestep : simulation time-step
        filename : path to file containing SBML definition of the model.
        """
        filename = os.path.join(os.path.dirname(__file__), filename)
        filename = os.path.abspath(filename)
        super().__init__(endtime, timestep, filename=filename, model_name='EGFR')

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
            val = initial_state[i]
            if val == 0:
                low = 0
                high = 50
            else:
                low = int(val * 0.1)
                high = val + int(val * sigm)
            settings[:, i] = np.random.randint(low, high, n_settings)
        return settings

    @staticmethod
    def get_species_names():
        """Returns list of all species names."""
        return ['EGF', 'R', 'Ra', 'R2', 'RP', 'PLCg', 'RPLCg', 'RPLCgP', 'PLCgP', 'Grb', 'RG', 'SOS',
                'RGS', 'GS', 'Shc', 'RSh', 'RShP', 'ShP', 'RShG', 'ShG', 'RShGS', 'ShGS', 'PLCgl']

    @staticmethod
    def get_initial_state():
        """Returns list of species initial values."""
        return [680, 100, 0, 0, 0, 105, 0, 0, 0, 85, 0, 34, 0, 0, 150, 0, 0, 0, 0, 0, 0, 0, 0]

    @staticmethod
    def get_species_for_histogram():
        """Returns list of species to create histograms for evaluation"""
        return ['EGF', 'R', 'PLCg']

    def process_params(self, params):
        """Set model parameters extracted from SBML definition."""
        for name, val in params.items():
            self.add_parameter(
                gillespy.Parameter(
                    name=name,
                    expression=val))

    def process_reactions(self, reactions):
        """
        Set model reactions extracted from SBML definition.
        Reversible reactions are and their kinetic laws are split into forward and
        backward. SBML keyword `compartment` is removed from the expressions of kinetic laws.

        Parameters
        ----------
        reactions : list of rections (libsbml.Reaction)

        Returns
        -------
        None

        """
        for reaction in reactions:
            r_name = reaction.id
            r_reactants_dict = self.get_reactants_dict(reaction)
            r_products_dict = self.get_products_dict(reaction)
            r_kinetic_law = self._kinetic_law_conversion(reaction, forward=True)
            print("Reaction: {}".format(r_name))
            print("reactants: {}".format(r_reactants_dict))
            print("products: {}".format(r_products_dict))
            print("Original: {}".format(reaction.getKineticLaw().formula))

            self.add_reaction(
                gillespy.Reaction(
                    name=r_name,
                    reactants=r_reactants_dict,
                    products=r_products_dict,
                    propensity_function=r_kinetic_law))

            if reaction.reversible is True:
                print("Forward:  {}".format(r_kinetic_law))
                r_kinetic_law = self._kinetic_law_conversion(reaction, forward=False)
                print("Backward: {}".format(r_kinetic_law))

                self.add_reaction(
                    gillespy.Reaction(
                        name=r_name + '_inverse',
                        reactants=r_products_dict,
                        products=r_reactants_dict,
                        propensity_function=r_kinetic_law))
            else:
                print("Converted: {}".format(r_kinetic_law))
            print()

    def process_species(self, species):
        """
        Set model species extracted from SBML definition.
        `EmptySet` species is ignored.

        Parameters
        ----------
        species : list of species (libsbml.Species)

        Returns
        -------
        None

        """
        for spec in species:
            name = spec.id
            if name == 'EmptySet':
                continue
            initial_value = self._concentration_conversion(spec.initial_concentration)
            self.species.append(name)
            self.initial_state.append(initial_value)
            self.add_species(gillespy.Species(name=name, initial_value=initial_value))

    @staticmethod
    def _concentration_conversion(n):
        return int(n)

    def _kinetic_law_conversion(self, reaction, forward=True):
        """
        Removes `compartment` keyword, optionally splits kinetic law into forward and backward.

        Parameters
        ----------
        reaction : libsbml.Reaction
        forward : returns either forward or backward part

        Returns
        -------
        law : preprocessed string expression of kinetic law

        """
        law = reaction.getKineticLaw().formula
        law = law.replace(" * compartment", "")
        reactants_dict = self.get_reactants_dict(reaction)

        if reaction.reversible:
            law = law.replace(' - ', ') - (')
            if forward:
                law = law.split(" - ")[0]
            else:
                reactants_dict = self.get_products_dict(reaction)
                law = law.split(" - ")[1].replace(' + ', ' - ')

        if len(reactants_dict) == 0:
            pass
        elif len(reactants_dict) == 1:
            if list(reactants_dict.values())[0] == 2:
                pass
            else:
                pass
        elif len(reactants_dict) == 2:
            pass

        return law
