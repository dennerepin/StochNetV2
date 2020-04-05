# import gillespy
import gillespy2 as gillespy
import numpy as np
import math
import libsbml

from abc import abstractmethod


class BaseCRNModel(gillespy.Model):

    params = {}

    def __init__(
            self,
            endtime,
            timestep,
            model_name,
    ):
        """
        Base class for CRN.

        Parameters
        ----------
        endtime : endtime of simulations
        timestep : time-step of simulations
        model_name : name of the model
        """
        super().__init__(name=model_name)

        self.species = []
        self.initial_state = []

        nb_of_steps = int(math.ceil((endtime / timestep))) + 1
        self.timespan(np.linspace(0, endtime, nb_of_steps))

    def set_species_initial_value(self, species_initial_value):
        """
        Set initial values to species.

        Parameters
        ----------
        species_initial_value : list or 1d array of values, size should be equal
            to the number of species, the order should be coherent with theo order
            of species returned by get_initial_state method.

        Returns
        -------
        None

        """
        species_names = self.get_species_names()
        for s_name in species_names:
            idx = species_names.index(s_name)
            self.listOfSpecies[s_name].initial_value = species_initial_value[idx]

    def get_species_values(self):
        return [s.initial_value for s in self.listOfSpecies.values()]

    @staticmethod
    @abstractmethod
    def get_species_names():
        """
        Returns list of species names.
        Should be implemented by all descendant classes.

        Returns
        -------
        list of all species names. The order of names should be coherent
        with the list returned by get_initial_state method.

        """
        pass

    @staticmethod
    @abstractmethod
    def get_initial_state():
        """
        Returns default initial state.
        Should be implemented by all descendant classes.

        Returns
        -------
        list of species initial values. The order of values should be coherent
        with the list returned by get_species_names method.

        """
        pass

    @staticmethod
    @abstractmethod
    def get_species_for_histogram():
        """
        Returns subset of species of interest.
        Should be implemented by all descendant classes.

        Returns
        -------
        list of species names.

        """
        pass

    @classmethod
    @abstractmethod
    def get_initial_settings(cls, n_settings, sigm):
        """
        Generate a set of (random) initial states.
        Should be implemented by all descendant classes.

        Parameters
        ----------
        n_settings : number of settings to produce.
        sigm : float parameter for variation of produced settings.

        Returns
        -------
        array of n_settings initial states

        """
        pass

    @classmethod
    def get_n_species(cls):
        """Total number of species."""
        return len(cls.get_species_names())

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
        species = cls.get_species_names()
        species_for_histogram = species_names_list or cls.get_species_for_histogram()
        initial_state = cls.get_initial_state()
        histogram_bounds = []
        for s in species_for_histogram:
            idx = species.index(s)
            val = initial_state[idx]
            histogram_bounds.append([0.5, val * 2] if val != 0 else [0.5, cls._hist_top_bound])
        return histogram_bounds

    @classmethod
    def get_randomized_parameters(cls, param_names, n_settings, sigm=0.4):
        randomized = {}
        for name in param_names:
            if name not in cls.params:
                raise KeyError(f"Could not find param {name} in {cls.__name__} class `params` dict.")
            val = float(cls.params[name])

            randomized[name] = np.random.uniform(val * (1. - sigm), val * (1. + sigm), n_settings)
        return randomized

    def set_parameters(self, params_dict):
        for name, val in params_dict.items():
            if name not in self.listOfParameters:
                raise KeyError(
                    f"Could not find {name} parameter in {self.__class__.__name__} model listOfParameters.")
            if isinstance(val, np.ndarray):
                if len(val) != 1:
                    raise ValueError(
                        f"Expected a single parameter value, got an array of shape {val.shape}")
                val = val[0]
            self.set_parameter(name, str(val))


class BaseSBMLModel(BaseCRNModel):

    def __init__(
            self,
            endtime,
            timestep,
            filename,
            model_name,
    ):
        """
        Base model for CRNs defined in SBML format.
        Though gillespy lib supports direct import of SBML models,
        not all models are imported correctly. This class (and its descendants)
        can be used to preprocess SBML definition to build model.

        Parameters
        ----------
        endtime : endtime of simulations
        timestep : time-step of simulations
        filename : path to file containing SBML model
        model_name : name of the model
        """
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
        """
        Set model parameters extracted from SBML definition.
        Parameters may be optionally preprocessed.

        Parameters
        ----------
        params : dict {name: value} of parameters

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def process_species(self, species):
        """
        Set model species extracted from SBML definition.

        Parameters
        ----------
        species : list of species (libsbml.Species) extracted from SBML definition

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def process_reactions(self, reactions):
        """
        Set model reactions extracted from SBML definition.
        Reactions may be optionally preprocessed: e.g. reversible reactions
        should be split into two and the rates should be modified correspondingly.
        Some SBML keywords like `compartment` included in rate expression can't be
        understood, so they should be either manually added to model parameters or
        removed from expression.

        Parameters
        ----------
        reactions : list of reactions (libsbml.Reaction) extracted from SBML definition

        Returns
        -------

        """
        pass

    @staticmethod
    def get_reactants_dict(reaction):
        """
        Returns stoichiometry of reactants.

        Parameters
        ----------
        reaction : libsbml.Reaction instance

        Returns
        -------
        reactants_dict : dict of {species_name: amount_consumed}

        """
        reactants = reaction.reactants
        reactants_dict = {}
        for reactant in reactants:
            species = reactant.species
            stoichiometry = reactant.stoichiometry
            reactants_dict[species] = stoichiometry
        return reactants_dict

    @staticmethod
    def get_products_dict(reaction):
        """
        Returns stoichiometry of reaction products.

        Parameters
        ----------
        reaction : libsbml.Reaction instance

        Returns
        -------
        products_dict : dict of {species_name: amount_produced}

        """
        products = reaction.products
        products_dict = {}
        for product in products:
            species = product.species
            stoichiometry = product.stoichiometry
            products_dict[species] = stoichiometry
        return products_dict
