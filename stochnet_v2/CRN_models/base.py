import gillespy
import numpy as np
import math
import libsbml

from abc import abstractmethod


class BaseCRNModel(gillespy.Model):

    def __init__(
            self,
            endtime,
            timestep,
            model_name,
    ):
        super().__init__(name=model_name)

        self.species = []
        self.initial_state = []

        nb_of_steps = int(math.ceil((endtime / timestep))) + 1
        self.timespan(np.linspace(0, endtime, nb_of_steps))

    def set_species_initial_value(self, species_initial_value):
        for s_name in self.species:
            idx = self.species.index(s_name)
            self.listOfSpecies[s_name].initial_value = species_initial_value[idx]

    def get_species_values(self):
        return [s.initial_value for s in self.listOfSpecies.values()]

    @staticmethod
    @abstractmethod
    def get_species_names():
        """List of species names"""
        pass

    @staticmethod
    def get_initial_state():
        """Default Initial State"""
        pass

    @staticmethod
    @abstractmethod
    def get_species_for_histogram():
        """Subset of species of interest"""
        pass

    @classmethod
    @abstractmethod
    def get_initial_settings(cls, n_settings, sigm):
        pass

    @classmethod
    def get_n_species(cls):
        return len(cls.get_species_names())

    @classmethod
    def get_histogram_bounds(cls, species_names_list=None):
        species = cls.get_species_names()
        species_for_histogram = species_names_list or cls.get_species_for_histogram()
        initial_state = cls.get_initial_state()
        histogram_bounds = []
        for s in species_for_histogram:
            idx = species.index(s)
            val = initial_state[idx]
            histogram_bounds.append([0.5, val * 2] if val != 0 else [0.5, cls._hist_top_bound])
        return histogram_bounds


class SBMLModel(BaseCRNModel):

    def __init__(
            self,
            endtime,
            timestep,
            filename,
            model_name,
    ):
        super().__init__(endtime, timestep, model_name)

        reader = libsbml.SBMLReader()
        document = reader.readSBML(filename)
        print("Errors in file: {}".format(document.getNumErrors()))

        model = document.getModel()
        params = {e.id: e.value for e in model.getListOfAllElements() if e.element_name == 'parameter'}
        species = model.getListOfSpecies()
        reactions = model.getListOfReactions()

        self.process_params(params)
        self.process_species(species)
        self.process_reactions(reactions)

    @abstractmethod
    def process_params(self, params):
        pass

    @abstractmethod
    def process_species(self, species):
        pass

    @abstractmethod
    def process_reactions(self, reactions):
        pass

    @staticmethod
    def get_reactants_dict(reaction):
        reactants = reaction.reactants
        reactants_dict = {}
        for reactant in reactants:
            species = reactant.species
            stoichiometry = reactant.stoichiometry
            reactants_dict[species] = stoichiometry
        return reactants_dict

    @staticmethod
    def get_products_dict(reaction):
        products = reaction.products
        products_dict = {}
        for product in products:
            species = product.species
            stoichiometry = product.stoichiometry
            products_dict[species] = stoichiometry
        return products_dict
