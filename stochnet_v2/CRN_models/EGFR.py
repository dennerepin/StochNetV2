import gillespy
import numpy as np
import os
from stochnet_v2.CRN_models.base import SBMLModel


class EGFR(SBMLModel):

    _hist_top_bound = 200

    def __init__(
            self,
            endtime,
            timestep,
            filename='../SBML_models/BIOMD0000000048_url.xml'
    ):
        filename = os.path.join(os.path.dirname(__file__), filename)
        filename = os.path.abspath(filename)
        super().__init__(endtime, timestep, filename=filename, model_name='EGFR')

    @classmethod
    def get_initial_settings(cls, n_settings, sigm=0.5):
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
        return ['EGF', 'R', 'Ra', 'R2', 'RP', 'PLCg', 'RPLCg', 'RPLCgP', 'PLCgP', 'Grb', 'RG', 'SOS',
                'RGS', 'GS', 'Shc', 'RSh', 'RShP', 'ShP', 'RShG', 'ShG', 'RShGS', 'ShGS', 'PLCgl']

    @staticmethod
    def get_initial_state():
        return [680, 100, 0, 0, 0, 105, 0, 0, 0, 85, 0, 34, 0, 0, 150, 0, 0, 0, 0, 0, 0, 0, 0]

    @staticmethod
    def get_species_for_histogram():
        return ['EGF', 'R', 'PLCg']

    def process_params(self, params):
        for name, val in params.items():
            self.add_parameter(
                gillespy.Parameter(
                    name=name,
                    expression=val))

    def process_reactions(self, reactions):
        for reaction in reactions:
            r_name = reaction.id
            r_reactants_dict = self.get_reactants_dict(reaction)
            r_products_dict = self.get_products_dict(reaction)
            r_kinetic_law = self.kinetic_law_conversion(reaction, forward=True)
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
                r_kinetic_law = self.kinetic_law_conversion(reaction, forward=False)
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
        for spec in species:
            name = spec.id
            if name == 'EmptySet':
                continue
            initial_value=self.concentration_conversion(spec.initial_concentration)
            self.species.append(name)
            self.initial_state.append(initial_value)
            self.add_species(gillespy.Species(name=name, initial_value=initial_value))

    @staticmethod
    def concentration_conversion(n):
        return int(n)

    def kinetic_law_conversion(self, reaction, forward=True):

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
