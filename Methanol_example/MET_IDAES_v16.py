##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2020, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes-pse".
##############################################################################
"""
Task: IDAES Support for ARPE-E Differentiate
Scenario: Methanol Synthesis From Syngas
Author: D. Wang, J. Bao, Y. Chen, T. Ma, B. Paul and M. Zamarripa
"""

from timeit import default_timer as timer
import os
import sys
import numpy as np
sys.path.append(os.path.abspath("./METH_properties"))

# Import Pyomo libraries
from pyomo.environ import (Constraint,
                           Objective,
                           Var,
                           Expression,
                           ConcreteModel,
                           TransformationFactory,
                           value,
                           maximize,
                           minimize,
                           units as pyunits)
from pyomo.environ import TerminationCondition

# Import IDAES core libraries
from idaes.core import FlowsheetBlock
from idaes.core.util import get_solver, scaling as iscale
from idaes.core.util.model_statistics import (degrees_of_freedom, fixed_variables_set)
from idaes.core.util.initialization import propagate_state
import idaes.logger as idaeslog
from pyomo.network import Arc, SequentialDecomposition

# Import required models
from idaes.generic_models.properties.core.generic.generic_property import \
    GenericParameterBlock
from idaes.generic_models.properties.core.generic.generic_reaction import \
    GenericReactionParameterBlock

# modules to update "vapor only" version of properties dictionary
from idaes.core import VaporPhase
from idaes.generic_models.properties.core.eos.ideal import Ideal

import methanol_ideal_VLE as thermo_props_VLE
import methanol_reactions as reaction_props

from idaes.generic_models.unit_models import (
    Mixer,
    Heater,
    Compressor,
    Turbine,
    StoichiometricReactor,
    Separator as Splitter,
    Product,
    Feed,
    Flash)
from idaes.generic_models.unit_models.mixer import MomentumMixingType
from idaes.generic_models.unit_models.pressure_changer import \
    ThermodynamicAssumption
import idaes.core.util.unit_costing as costing
from idaes.core.util.unit_costing import initialize as init_costing

tot_flow = 3 # total fuel inlet [mol/s]

#%%
def build_model(m, list_unit, list_inlet, list_outlet, scale_thermo=False):
    
    # # change upper state bound on temperature to allow convergence
    # thermo_props.config_dict["state_bounds"]["temperature"] = (198.15, 298.15, 512.15, pyunits.K)
    
    m.fs.thermo_params_VLE = GenericParameterBlock(
        default=thermo_props_VLE.config_dict)

    if scale_thermo is True:
        m.fs.thermo_params_VLE.set_default_scaling("flow_mol", 1)
        m.fs.thermo_params_VLE.set_default_scaling("temperature", 1e-2)
        m.fs.thermo_params_VLE.set_default_scaling("pressure", 1e-2)       
        m.fs.thermo_params_VLE.set_default_scaling("enth_mol", 1e-3)
        m.fs.thermo_params_VLE.set_default_scaling("entr_mol", 1e-1)
        for comp in thermo_props_VLE.config_dict["components"]:
            m.fs.thermo_params_VLE.set_default_scaling("mole_frac_comp", 1e2, index=comp)
            m.fs.thermo_params_VLE.set_default_scaling("enth_mol_comp", 1e2, index=comp)
            m.fs.thermo_params_VLE.set_default_scaling("entr_mol_phase_comp", 1e2, index=comp)
            for attr in dir(getattr(m.fs.thermo_params_VLE, comp)):
                if 'coef' in attr:
                    iscale.set_scaling_factor(getattr(getattr(m.fs.thermo_params_VLE, comp), attr), 1)
            m.fs.thermo_params_VLE.set_default_scaling("flow_mol_phase_comp", 1, index=comp)

    # restrict a version of the properties only use vapor to help calculations when liquid is not present

    import copy
    vapor_config_dict = copy.deepcopy(thermo_props_VLE.config_dict)
    vapor_config_dict["phases"] = {'Vap': {"type": VaporPhase, "equation_of_state": Ideal}}
    vapor_config_dict.pop("phases_in_equilibrium")
    vapor_config_dict.pop("phase_equilibrium_state")
    vapor_config_dict.pop("bubble_dew_method")
    m.fs.thermo_params_vapor = GenericParameterBlock(
        default=vapor_config_dict)

    if scale_thermo is True:
        m.fs.thermo_params_vapor.set_default_scaling("flow_mol", 1)
        m.fs.thermo_params_vapor.set_default_scaling("temperature", 1e-2)
        m.fs.thermo_params_vapor.set_default_scaling("pressure", 1e-2)       
        m.fs.thermo_params_vapor.set_default_scaling("enth_mol", 1e-3)
        m.fs.thermo_params_vapor.set_default_scaling("entr_mol", 1e-1)
        for comp in vapor_config_dict["components"]:
            m.fs.thermo_params_vapor.set_default_scaling("mole_frac_comp", 1e2, index=comp)
            m.fs.thermo_params_vapor.set_default_scaling("enth_mol_comp", 1e2, index=comp)
            m.fs.thermo_params_vapor.set_default_scaling("entr_mol_phase_comp", 1e2, index=comp)
            for attr in dir(getattr(m.fs.thermo_params_vapor, comp)):
                if 'coef' in attr:
                    iscale.set_scaling_factor(getattr(getattr(m.fs.thermo_params_vapor, comp), attr), 1)
            m.fs.thermo_params_vapor.set_default_scaling("flow_mol_phase_comp", 1, index=comp)

    m.fs.reaction_params = GenericReactionParameterBlock(
        default={"property_package": m.fs.thermo_params_vapor,
                 **reaction_props.config_dict})

    # exhaust and product
    if 'exhaust_1' in list_unit:
        m.fs.exhaust_1 = Product(default={'property_package': m.fs.thermo_params_vapor})

    if 'exhaust_2' in list_unit:
        m.fs.exhaust_2 = Product(default={'property_package': m.fs.thermo_params_vapor})

    if 'product_0' in list_unit:
        m.fs.product_0 = Product(default={'property_package': m.fs.thermo_params_VLE})

    # mixing feed streams
    if 'mixer_0' in list_unit: # must exist
        m.fs.mixer_0 = Mixer(
            default={"property_package": m.fs.thermo_params_vapor,
                    "momentum_mixing_type": MomentumMixingType.minimize,
                    "has_phase_equilibrium": True,
                    "inlet_list": ['inlet_1', 'inlet_2']})
    
    if 'mixer_1' in list_unit:
        m.fs.mixer_1 = Mixer(
            default={"property_package": m.fs.thermo_params_vapor,
                    "momentum_mixing_type": MomentumMixingType.minimize,
                    "has_phase_equilibrium": True,
                    "inlet_list": ['inlet_1', 'inlet_2']})

    if 'mixer_2' in list_unit:
        m.fs.mixer_2 = Mixer(
            default={"property_package": m.fs.thermo_params_vapor,
                    "momentum_mixing_type": MomentumMixingType.minimize,
                    "has_phase_equilibrium": True,
                    "inlet_list": ['inlet_1', 'inlet_2']})

    # pre-compression
    if 'compressor_1' in list_unit:
        m.fs.compressor_1 = Compressor(
            default={"dynamic": False,
                    "property_package": m.fs.thermo_params_vapor,
                    "compressor": True,
                    "thermodynamic_assumption": ThermodynamicAssumption.isothermal
                    })

    if 'compressor_2' in list_unit:
        m.fs.compressor_2 = Compressor(
            default={"dynamic": False,
                    "property_package": m.fs.thermo_params_vapor,
                    "compressor": True,
                    "thermodynamic_assumption": ThermodynamicAssumption.isothermal
                    })

    # pre-heating
    if 'heater_1' in list_unit:
        m.fs.heater_1 = Heater(
            default={"property_package": m.fs.thermo_params_vapor,
                    "has_pressure_change": False,
                    "has_phase_equilibrium": False})

    if 'heater_2' in list_unit:
        m.fs.heater_2 = Heater(
            default={"property_package": m.fs.thermo_params_vapor,
                    "has_pressure_change": False,
                    "has_phase_equilibrium": False})

    # StReactor
    if 'StReactor_1' in list_unit:
        m.fs.StReactor_1 = StoichiometricReactor(
            default={"has_heat_transfer": True,
                    "has_heat_of_reaction": True,
                    "has_pressure_change": False,
                    "property_package": m.fs.thermo_params_vapor,
                    "reaction_package": m.fs.reaction_params})

    if 'StReactor_2' in list_unit:
        m.fs.StReactor_2 = StoichiometricReactor(
            default={"has_heat_transfer": True,
                    "has_heat_of_reaction": True,
                    "has_pressure_change": False,
                    "property_package": m.fs.thermo_params_vapor,
                    "reaction_package": m.fs.reaction_params})

    # post-expansion
    if 'expander_1' in list_unit:
        m.fs.expander_1 = Turbine(
            default={"dynamic": False,
                    "property_package": m.fs.thermo_params_vapor})

    if 'expander_2' in list_unit:
        m.fs.expander_2 = Turbine(
            default={"dynamic": False,
                    "property_package": m.fs.thermo_params_vapor})

    # post-cooling
    if 'cooler_1' in list_unit:
        m.fs.cooler_1 = Heater(
            default={"property_package": m.fs.thermo_params_vapor,
                    "has_pressure_change": False,
                    "has_phase_equilibrium": False})

    if 'cooler_2' in list_unit:
        m.fs.cooler_2 = Heater(
            default={"property_package": m.fs.thermo_params_vapor,
                    "has_pressure_change": False,
                    "has_phase_equilibrium": False})

    # product recovery
    if 'flash_0' in list_unit:
        m.fs.flash_0 = Flash(
            default={"property_package": m.fs.thermo_params_VLE,
                    "has_heat_transfer": True,
                    "has_pressure_change": True})

    if 'flash_1' in list_unit:
        m.fs.flash_1 = Flash(
            default={"property_package": m.fs.thermo_params_VLE,
                    "has_heat_transfer": True,
                    "has_pressure_change": True})

    # splitter
    if 'splitter_1' in list_unit:
        m.fs.splitter_1 = Splitter(default={
            "property_package": m.fs.thermo_params_vapor,
            "ideal_separation": False,
             "outlet_list": ["outlet_1", "outlet_2"]})

    if 'splitter_2' in list_unit:
        m.fs.splitter_2 = Splitter(default={
            "property_package": m.fs.thermo_params_vapor,
            "ideal_separation": False,
             "outlet_list": ["outlet_1", "outlet_2"]})

    # build arcs
    for i in range(len(list_inlet)):
        expression = 'm.fs.Arc'+str(i)+' = Arc(source = m.fs.'\
            +list_outlet[i]+', destination = m.fs.'+list_inlet[i]+')'
        exec(expression)

    TransformationFactory("network.expand_arcs").apply_to(m)
    print("Degrees of Freedom (build) = %d" % degrees_of_freedom(m))

def set_inputs(m, list_unit):

    #  feed streams, post WGS
    if 'mixer_0' in list_unit:
        m.fs.mixer_0.inlet_1.flow_mol[0].fix(tot_flow*2.0/3.0)  # mol/s, relative to 177 kmol/h (49.2 mol/s)
        m.fs.mixer_0.inlet_1.mole_frac_comp[0, "H2"].fix(1)
        m.fs.mixer_0.inlet_1.mole_frac_comp[0, "CO"].fix(1e-6)
        m.fs.mixer_0.inlet_1.mole_frac_comp[0, "CH3OH"].fix(1e-6)
        m.fs.mixer_0.inlet_1.mole_frac_comp[0, "CH4"].fix(1e-6)
#        m.fs.mixer_0.inlet_1.mole_frac_comp[0, "H2O"].fix(1e-6)
        m.fs.mixer_0.inlet_1.enth_mol[0].fix(-142.4)  # J/mol
        m.fs.mixer_0.inlet_1.pressure.fix(30e5)  # Pa

        m.fs.mixer_0.inlet_2.flow_mol[0].fix(tot_flow/3.0)  # mol/s, relative to 88 kmol/h (24.4 mol/s)
        m.fs.mixer_0.inlet_2.mole_frac_comp[0, "H2"].fix(1e-6)
        m.fs.mixer_0.inlet_2.mole_frac_comp[0, "CO"].fix(1)
        m.fs.mixer_0.inlet_2.mole_frac_comp[0, "CH3OH"].fix(1e-6)
        m.fs.mixer_0.inlet_2.mole_frac_comp[0, "CH4"].fix(1e-6)
#        m.fs.mixer_0.inlet_1.mole_frac_comp[0, "H2O"].fix(1e-6)
        m.fs.mixer_0.inlet_2.enth_mol[0].fix(-110676.4)  # J/mol
        m.fs.mixer_0.inlet_2.pressure.fix(30e5)  # Pa

    # units specifications
    if 'compressor_1' in list_unit:
        m.fs.compressor_1.outlet.pressure.fix(51e5)  # Pa

    if 'compressor_2' in list_unit:
        m.fs.compressor_2.outlet.pressure.fix(51e5)  # Pa

    if 'heater_1' in list_unit:
        m.fs.heater_1.outlet_temp = Constraint(
            expr=m.fs.heater_1.control_volume.properties_out[0].temperature == 488.15)

    if 'heater_2' in list_unit:
        m.fs.heater_2.outlet_temp = Constraint(
            expr=m.fs.heater_2.control_volume.properties_out[0].temperature == 488.15)

    if 'StReactor_1' in list_unit:
        m.fs.StReactor_1.conversion = Var(initialize=0.75, bounds=(0, 1))
        m.fs.StReactor_1.conv_constraint = Constraint(
            expr=(m.fs.StReactor_1.conversion * m.fs.StReactor_1.inlet.flow_mol[0] *
                m.fs.StReactor_1.inlet.mole_frac_comp[0, "CO"] ==
                m.fs.StReactor_1.inlet.flow_mol[0] *
                m.fs.StReactor_1.inlet.mole_frac_comp[0, "CO"]
                - m.fs.StReactor_1.outlet.flow_mol[0] *
                m.fs.StReactor_1.outlet.mole_frac_comp[0, "CO"]))
        m.fs.StReactor_1.conversion.fix(0.75)
        m.fs.StReactor_1.outlet_temp = Constraint(
            expr=m.fs.StReactor_1.control_volume.properties_out[0].temperature == 507.15)
        m.fs.StReactor_1.heat_duty.setub(0)  # rxn is exothermic, so duty is cooling only

    if 'StReactor_2' in list_unit:
        m.fs.StReactor_2.conversion = Var(initialize=0.75, bounds=(0, 1))
        m.fs.StReactor_2.conv_constraint = Constraint(
            expr=(m.fs.StReactor_2.conversion * m.fs.StReactor_2.inlet.flow_mol[0] *
                m.fs.StReactor_2.inlet.mole_frac_comp[0, "CO"] ==
                m.fs.StReactor_2.inlet.flow_mol[0] *
                m.fs.StReactor_2.inlet.mole_frac_comp[0, "CO"]
                - m.fs.StReactor_2.outlet.flow_mol[0] *
                m.fs.StReactor_2.outlet.mole_frac_comp[0, "CO"]))
        m.fs.StReactor_2.conversion.fix(0.75)
        m.fs.StReactor_2.outlet_temp = Constraint(
            expr=m.fs.StReactor_2.control_volume.properties_out[0].temperature == 507.15)
        m.fs.StReactor_2.heat_duty.setub(0)  # rxn is exothermic, so duty is cooling only

    if 'expander_1' in list_unit:
        m.fs.expander_1.deltaP.fix(-2e6)
        m.fs.expander_1.efficiency_isentropic.fix(0.9)

    if 'expander_2' in list_unit:
        m.fs.expander_2.deltaP.fix(-2e6)
        m.fs.expander_2.efficiency_isentropic.fix(0.9)

    if 'cooler_1' in list_unit:
        m.fs.cooler_1.outlet_temp = Constraint(
            expr=m.fs.cooler_1.control_volume.properties_out[0].temperature == 407.15)

    if 'cooler_2' in list_unit:
        m.fs.cooler_2.outlet_temp = Constraint(
            expr=m.fs.cooler_2.control_volume.properties_out[0].temperature == 407.15)

    if 'flash_0' in list_unit:
        m.fs.flash_0.recovery = Var(initialize=0.01, bounds=(0, 1))
        m.fs.flash_0.rec_constraint = Constraint(
            expr=(m.fs.flash_0.recovery == m.fs.flash_0.liq_outlet.flow_mol[0] *
                m.fs.flash_0.liq_outlet.mole_frac_comp[0, "CH3OH"] /
                (m.fs.flash_0.inlet.flow_mol[0] *
                m.fs.flash_0.inlet.mole_frac_comp[0, "CH3OH"])))
        m.fs.flash_0.deltaP.fix(0)  # Pa
        m.fs.flash_0.outlet_temp = Constraint(
            expr=m.fs.flash_0.control_volume.properties_out[0].temperature == 407.15)

    if 'flash_1' in list_unit:
        m.fs.flash_1.recovery = Var(initialize=0.01, bounds=(0, 1))
        m.fs.flash_1.rec_constraint = Constraint(
            expr=(m.fs.flash_1.recovery == m.fs.flash_1.liq_outlet.flow_mol[0] *
                m.fs.flash_1.liq_outlet.mole_frac_comp[0, "CH3OH"] /
                (m.fs.flash_1.inlet.flow_mol[0] *
                m.fs.flash_1.inlet.mole_frac_comp[0, "CH3OH"])))
        m.fs.flash_1.deltaP.fix(0)  # Pa
        m.fs.flash_1.outlet_temp = Constraint(
            expr=m.fs.flash_1.control_volume.properties_out[0].temperature == 407.15)

    if 'splitter_1' in list_unit:
        m.fs.splitter_1.split_fraction[0, "outlet_2"].fix(0.9999)
        
    if 'splitter_2' in list_unit:
        m.fs.splitter_2.split_fraction[0, "outlet_2"].fix(0.9999)
        
    if 'flash_0' in list_unit:
        m.fs.meth_flow = Expression(expr=(m.fs.flash_0.liq_outlet.flow_mol[0] * \
                                     m.fs.flash_0.liq_outlet.mole_frac_comp[0, "CH3OH"]))
        m.fs.efficiency = Expression(expr=(m.fs.flash_0.liq_outlet.flow_mol[0] * \
                                     m.fs.flash_0.liq_outlet.mole_frac_comp[0, "CH3OH"]/(tot_flow/3.0)))

    print("Degrees of Freedom (set inputs) = %d" % degrees_of_freedom(m))

def scale_variables(m, list_unit):

    for var in m.fs.component_data_objects(Var, descend_into=True):
        if 'flow_mol' in var.name:
            iscale.set_scaling_factor(var, 1)
        if 'temperature' in var.name:
            iscale.set_scaling_factor(var, 1e-2)
        if 'pressure' in var.name:
            iscale.set_scaling_factor(var, 1e-2)
        if 'enth_mol' in var.name:
            iscale.set_scaling_factor(var, 1e-3)
        if 'mole_frac' in var.name:
            iscale.set_scaling_factor(var, 1e2)
        if 'entr_mol' in var.name:
            iscale.set_scaling_factor(var, 1e-1)
        if 'rate_reaction_extent' in var.name:
            iscale.set_scaling_factor(var, 1e-3)
        if 'heat' in var.name:
            iscale.set_scaling_factor(var, 1e-3)
        if 'work' in var.name:
            iscale.set_scaling_factor(var, 1e-3)

    # manual scaling
    for unit in list_unit:
        block = getattr(m.fs, unit)
        if 'mixer' in unit:
            iscale.set_scaling_factor(block.inlet_1_state[0.0].mole_frac_comp, 1e2)
            iscale.set_scaling_factor(block.inlet_2_state[0.0].mole_frac_comp, 1e2)
            iscale.set_scaling_factor(block.mixed_state[0.0].mole_frac_comp, 1e2)
            iscale.set_scaling_factor(block.inlet_1_state[0.0].enth_mol_phase, 1e-3)
            iscale.set_scaling_factor(block.inlet_2_state[0.0].enth_mol_phase, 1e-3)
            iscale.set_scaling_factor(block.mixed_state[0.0].enth_mol_phase, 1e-3)
        if 'splitter' in unit:
            iscale.set_scaling_factor(block.mixed_state[0.0].mole_frac_comp, 1e2)
            iscale.set_scaling_factor(block.outlet_1_state[0.0].mole_frac_comp, 1e2)
            iscale.set_scaling_factor(block.outlet_2_state[0.0].mole_frac_comp, 1e2)
            iscale.set_scaling_factor(block.mixed_state[0.0].enth_mol_phase, 1e-3)
            iscale.set_scaling_factor(block.outlet_1_state[0.0].enth_mol_phase, 1e-3)
            iscale.set_scaling_factor(block.outlet_2_state[0.0].enth_mol_phase, 1e-3)
        print(block)
        if hasattr(block, "control_volume"):
            iscale.set_scaling_factor(
                block.control_volume.properties_in[0.0].mole_frac_comp, 1e2)
            iscale.set_scaling_factor(
                block.control_volume.properties_out[0.0].mole_frac_comp, 1e2)
            iscale.set_scaling_factor(
                block.control_volume.properties_in[0.0].enth_mol_phase, 1e2)
            iscale.set_scaling_factor(
                block.control_volume.properties_out[0.0].enth_mol_phase, 1e2)
            if hasattr(block.control_volume, "rate_reaction_extent"):
                iscale.set_scaling_factor(block.control_volume.rate_reaction_extent, 1e3)
            if hasattr(block.control_volume, "heat"):
                iscale.set_scaling_factor(block.control_volume.heat, 1e-3)
            if hasattr(block.control_volume, "work"):
                iscale.set_scaling_factor(block.control_volume.work, 1e-3)
        if hasattr(block, "properties_isentropic"):
            iscale.set_scaling_factor(
                block.properties_isentropic[0.0].mole_frac_comp, 1e2)
            iscale.set_scaling_factor(
                block.properties_isentropic[0.0].enth_mol_phase, 1e-3)
        if hasattr(block, "properties"):
            iscale.set_scaling_factor(
                block.properties[0.0].mole_frac_comp, 1e2)

    iscale.calculate_scaling_factors(m)

def scale_constraints(m, list_unit):
    # set scaling for unit constraints
    for name in list_unit:
        unit = getattr(m.fs, name)
        # mixer constraints
        if hasattr(unit, 'material_mixing_equations'):
            for (t, j), c in unit.material_mixing_equations.items():
                iscale.constraint_scaling_transform(c, 1, overwrite=False)
        if hasattr(unit, 'enthalpy_mixing_equations'):
            for t, c in unit.enthalpy_mixing_equations.items():
                iscale.constraint_scaling_transform(c, 1e-3, overwrite=False)
        if hasattr(unit, 'minimum_pressure_constraint'):
            for (t, i), c in unit.minimum_pressure_constraint.items():
                iscale.constraint_scaling_transform(c, 1e-5, overwrite=False)
        if hasattr(unit, 'mixture_pressure'):
            for t, c in unit.mixture_pressure.items():
                iscale.constraint_scaling_transform(c, 1e-5, overwrite=False)
        if hasattr(unit, 'pressure_equality_constraints'):
            for (t, i), c in unit.pressure_equality_constraints.items():
                iscale.constraint_scaling_transform(c, 1e-5, overwrite=False)

        # splitter constraints
        if hasattr(unit, 'material_splitting_eqn'):
            for (t, o, j), c in unit.material_splitting_eqn.items():
                iscale.constraint_scaling_transform(c, 1, overwrite=False)
        if hasattr(unit, 'temperature_equality_eqn'):
            for (t, o), c in unit.temperature_equality_eqn.items():
                iscale.constraint_scaling_transform(c, 1e-2, overwrite=False)
        if hasattr(unit, 'molar_enthalpy_equality_eqn'):
            for (t, o), c in unit.molar_enthalpy_equality_eqn.items():
                iscale.constraint_scaling_transform(c, 1e-3, overwrite=False)
        if hasattr(unit, 'molar_enthalpy_splitting_eqn'):
            for (t, o), c in unit.molar_enthalpy_splitting_eqn.items():
                iscale.constraint_scaling_transform(c, 1e-3, overwrite=False)
        if hasattr(unit, 'pressure_equality_eqn'):
            for (t, o), c in unit.pressure_equality_eqn.items():
                iscale.constraint_scaling_transform(c, 1e-5, overwrite=False)
        if hasattr(unit, 'sum_split_frac'):
            for t, c in unit.sum_split_frac.items():
                iscale.constraint_scaling_transform(c, 1e2, overwrite=False)

        # flash adds same as splitter, plus one more
        if hasattr(unit, 'split_fraction_eq'):
            for (t, o), c in unit.split_fraction_eq.items():
                iscale.constraint_scaling_transform(c, 1e2, overwrite=False)

        # pressurechanger constraints

        if hasattr(unit, "ratioP_calculation"):
            for t, c in unit.ratioP_calculation.items():
                iscale.constraint_scaling_transform(c, 1e-5, overwrite=False)

        if hasattr(unit, "fluid_work_calculation"):
            for t, c in unit.fluid_work_calculation.items():
                iscale.constraint_scaling_transform(c, 1e-5, overwrite=False)

        if hasattr(unit, "actual_work"):
            for t, c in unit.actual_work.items():
                iscale.constraint_scaling_transform(c, 1e-3, overwrite=False)

        if hasattr(unit, "isentropic_pressure"):
            for t, c in unit.isentropic_pressure.items():
                iscale.constraint_scaling_transform(c, 1e-5, overwrite=False)

        if hasattr(unit, "isothermal"):
            for t, c in unit.isothermal.items():
                iscale.constraint_scaling_transform(c, 1e-2, overwrite=False)

        if hasattr(unit, "isentropic"):
            for t, c in unit.isentropic.items():
                iscale.constraint_scaling_transform(c, 1e-1, overwrite=False)

        if hasattr(unit, "isentropic_energy_balance"):
            for t, c in unit.isentropic_energy_balance.items():
                iscale.constraint_scaling_transform(c, 1e-3, overwrite=False)

        if hasattr(unit, "zero_work_equation"):
            for t, c in unit.zero_work_equation.items():
                iscale.constraint_scaling_transform(c, 1e-3, overwrite=False)

        if hasattr(unit, "state_material_balances"):
            for (t, j), c in unit.state_material_balances.items():
                iscale.constraint_scaling_transform(c, 1, overwrite=False)

        # heater and reactor only add 0D control volume constraints
        if hasattr(unit, 'material_holdup_calculation'):
            for (t, p, j), c in unit.material_holdup_calculation.items():
                iscale.constraint_scaling_transform(c, 1, overwrite=False)
        if hasattr(unit, 'rate_reaction_stoichiometry_constraint'):
            for (t, p, j), c in unit.rate_reaction_stoichiometry_constraint.items():
                iscale.constraint_scaling_transform(c, 1, overwrite=False)
        if hasattr(unit, 'equilibrium_reaction_stoichiometry_constraint'):
            for (t, p, j), c in unit.equilibrium_reaction_stoichiometry_constraint.items():
                iscale.constraint_scaling_transform(c, 1, overwrite=False)
        if hasattr(unit, 'inherent_reaction_stoichiometry_constraint'):
            for (t, p, j), c in unit.inherent_reaction_stoichiometry_constraint.items():
                iscale.constraint_scaling_transform(c, 1, overwrite=False)
        if hasattr(unit, 'material_balances'):
            for (t, p, j), c in unit.material_balances.items():
                iscale.constraint_scaling_transform(c, 1, overwrite=False)
        if hasattr(unit, 'element_balances'):
            for (t, e), c in unit.element_balances.items():
                iscale.constraint_scaling_transform(c, 1, overwrite=False)
        if hasattr(unit, 'elemental_holdup_calculation'):
            for (t, e), c in unit.elemental_holdup_calculation.items():
                iscale.constraint_scaling_transform(c, 1, overwrite=False)
        if hasattr(unit, 'enthalpy_balances'):
            for t, c in unit.enthalpy_balances.items():
                iscale.constraint_scaling_transform(c, 1e-3, overwrite=False)
        if hasattr(unit, 'energy_holdup_calculation'):
            for (t, p), c in unit.energy_holdup_calculation.items():
                iscale.constraint_scaling_transform(c, 1e-3, overwrite=False)
        if hasattr(unit, 'pressure_balance'):
            for t, c in unit.pressure_balance.items():
                iscale.constraint_scaling_transform(c, 1e-5, overwrite=False)
        if hasattr(unit, 'sum_of_phase_fractions'):
            for t, c in unit.sum_of_phase_fractions.items():
                iscale.constraint_scaling_transform(c, 1e2, overwrite=False)
        if hasattr(unit, "material_accumulation_disc_eq"):
            for (t, p, j), c in unit.material_accumulation_disc_eq.items():
                iscale.constraint_scaling_transform(c, 1, overwrite=False)

        if hasattr(unit, "energy_accumulation_disc_eq"):
            for (t, p), c in unit.energy_accumulation_disc_eq.items():
                iscale.constraint_scaling_transform(c, 1e-3, overwrite=False)

        if hasattr(unit, "element_accumulation_disc_eq"):
            for (t, e), c in unit.element_accumulation_disc_eq.items():
                iscale.constraint_scaling_transform(c, 1, overwrite=False)

    # equality constraints between ports at Arc sources and destinations
    for arc in m.fs.component_data_objects(Arc, descend_into=True):
        for c in arc.component_data_objects(Constraint, descend_into=True):
            if hasattr(unit, "enth_mol_equality"):
                for t, c in unit.enth_mol_equality.items():
                    iscale.constraint_scaling_transform(c, 1e-3, overwrite=False)
            if hasattr(unit, "flow_mol_equality"):
                for t, c in unit.flow_mol_equality.items():
                    iscale.constraint_scaling_transform(c, 1, overwrite=False)
            if hasattr(unit, "mole_frac_comp_equality"):
                for (t, j), c in unit.mole_frac_comp_equality.items():
                    iscale.constraint_scaling_transform(c, 1e2, overwrite=False)
            if hasattr(unit, "pressure_equality"):
                for t, c in unit.pressure_equality.items():
                    iscale.constraint_scaling_transform(c, 1e-5, overwrite=False)

    iscale.calculate_scaling_factors(m)

def initialize_flowsheet(m, list_unit, list_inlet, list_outlet, iterlim):

    # Initialize and solve flowsheet
    
    # solver options
    seq = SequentialDecomposition()
    seq.options.select_tear_method = "heuristic"
    seq.options.tear_method = "Wegstein"
    seq.options.iterLim = iterlim

    # Using the SD tool, build the network we will solve
    G = seq.create_graph(m)
    heuristic_tear_set = seq.tear_set_arcs(G, method="heuristic")

    # order = seq.calculation_order(G)
    # print('\nTear Stream:')
    # for o in heuristic_tear_set:
    #     print(o.name, ': ', o.source.name, ' to ', o.destination.name)
    # print('\nCalculation order:')
    # for o in order:
    #     for p in o:
    #         print(p.name, end=' ')
    #     print()

    tear_guesses = {
        "flow_mol": {0: tot_flow},
        "mole_frac_comp": {
                (0, "CH4"): 1e-6,
                (0, "CO"): 0.33207,
                (0, "H2"): 0.66792,
                (0, "CH3OH"): 1e-6},
#                (0, "H2O"): 1e-6},
        "enth_mol": {0: -36848},
        "pressure": {0: 3e6}}
    
    # automatically build stream set for flowsheet and find the tear stream
    stream_set = [arc for arc in m.fs.component_data_objects(Arc)]
    for stream in stream_set:
        if stream in heuristic_tear_set:
            seq.set_guesses_for(stream.destination, tear_guesses)

    def function(unit):

        # print('solving ', str(unit))
        unit.initialize(outlvl=idaeslog.ERROR)  # no output unless it breaks

        for stream in stream_set:
            if stream.source.parent_block() == unit:
                propagate_state(arc=stream)  # this is an outlet of the unit
            stream.destination.unfix()
    
    seq.run(m, function)
    
    for stream in stream_set:
        if stream.destination.is_fixed() is True:
            # print('Unfixing ', stream.destination.name, '...')
            stream.destination.unfix()
        
    print("\nDegrees of Freedom (after initialize) = %d" % degrees_of_freedom(m))

def add_costing(m, list_unit):

    # Expression to compute the total cooling cost (F/R cooling not assumed)
    expression = 'm.fs.cooling_cost = Expression(expr=(0.0'
    if 'cooler_1' in list_unit:
        expression += ' -m.fs.cooler_1.heat_duty[0] * 0.212e-7' 
    if 'cooler_2' in list_unit:
        expression += ' -m.fs.cooler_2.heat_duty[0] * 0.212e-7' 
    if 'flash_0' in list_unit:
        expression += ' -m.fs.flash_0.heat_duty[0] * 0.25e-7' 
    if 'flash_1' in list_unit:
        expression += ' -m.fs.flash_1.heat_duty[0] * 0.25e-7' 
    if 'StReactor_1' in list_unit:
        expression += ' -m.fs.StReactor_1.heat_duty[0] * 2.2e-7' 
    if 'StReactor_2' in list_unit:
        expression += ' -m.fs.StReactor_2.heat_duty[0] * 2.2e-7' 
    expression += '))'
    exec(expression)

    # Expression to compute the total heating cost (F/R heating not assumed)
    expression = 'm.fs.heating_cost = Expression(expr=(0.0'
    if 'heater_1' in list_unit:
        expression += ' + m.fs.heater_1.heat_duty[0] * 2.2e-7' 
    if 'heater_2' in list_unit:
        expression += ' + m.fs.heater_2.heat_duty[0] * 2.2e-7' 
    expression += '))'
    exec(expression)

    # Expression to compute the total electricity cost (utilities - credit)
    expression = 'm.fs.electricity_cost = Expression(expr=(0.0'
    if 'expander_1' in list_unit:
        expression += ' - m.fs.expander_1.work_isentropic[0] * 0.08e-5' 
    if 'expander_2' in list_unit:
        expression += ' - m.fs.expander_2.work_isentropic[0] * 0.08e-5' 
    if 'compressor_1' in list_unit:
        expression += ' + m.fs.compressor_1.work_mechanical[0] * 0.12e-5' 
    if 'compressor_2' in list_unit:
        expression += ' + m.fs.compressor_2.work_mechanical[0] * 0.12e-5' 
    expression += '))'
    exec(expression)

    # Expression to compute the total operating cost
    m.fs.operating_cost = Expression(
        expr=(3600 * 24 * 365 * (m.fs.heating_cost + m.fs.cooling_cost
                                 + m.fs.electricity_cost)))

    # Expression to compute the annualized capital cost
    expression = 'm.fs.annualized_capital_cost = Expression(expr=(0.0'

    # Computing StReactor capital cost
    if 'StReactor_1' in list_unit:
        m.fs.StReactor_1.get_costing()
        m.fs.StReactor_1.diameter.fix(2)
        m.fs.StReactor_1.length.fix(4)  # for initial problem at 75% conversion
        init_costing(m.fs.StReactor_1.costing)
        # Reactor length (size, and capital cost) is adjusted based on conversion
        # surrogate model which scales length linearly with conversion
        m.fs.StReactor_1.length.unfix()
        m.fs.StReactor_1.L_eq = Constraint(expr=m.fs.StReactor_1.length ==
                                    13.2000*m.fs.StReactor_1.conversion - 5.9200)
        # m.fs.StReactor_1.conversion_lb = Constraint(expr=m.fs.StReactor_1.conversion >= 0.75)
        # m.fs.StReactor_1.conversion_ub = Constraint(expr=m.fs.StReactor_1.conversion <= 0.85)
        expression += ' + m.fs.StReactor_1.costing.purchase_cost'

    if 'StReactor_2' in list_unit:
        m.fs.StReactor_2.get_costing()
        m.fs.StReactor_2.diameter.fix(2)
        m.fs.StReactor_2.length.fix(4)  # for initial problem at 75% conversion
        init_costing(m.fs.StReactor_2.costing)
        # Reactor length (size, and capital cost) is adjusted based on conversion
        # surrogate model which scales length linearly with conversion
        m.fs.StReactor_2.length.unfix()
        m.fs.StReactor_2.L_eq = Constraint(expr=m.fs.StReactor_2.length ==
                                    13.2000*m.fs.StReactor_2.conversion - 5.9200)
        # m.fs.StReactor_2.conversion_lb = Constraint(expr=m.fs.StReactor_2.conversion >= 0.75)
        # m.fs.StReactor_2.conversion_ub = Constraint(expr=m.fs.StReactor_2.conversion <= 0.85)
        expression += ' + m.fs.StReactor_2.costing.purchase_cost'

    # Computing flash capital cost
    if 'flash_0' in list_unit:
        m.fs.flash_0.get_costing()
        m.fs.flash_0.diameter.fix(2)
        m.fs.flash_0.length.fix(4)
        init_costing(m.fs.flash_0.costing)
        expression += ' + m.fs.flash_0.costing.purchase_cost'

    if 'flash_1' in list_unit:
        m.fs.flash_1.get_costing()
        m.fs.flash_1.diameter.fix(2)
        m.fs.flash_1.length.fix(4)
        init_costing(m.fs.flash_1.costing)
        expression += ' + m.fs.flash_1.costing.purchase_cost'

    # Computing heater/cooler capital costs
    # Surrogates prepared with IDAES shell and tube hx considering IP steam and
    # assuming steam outlet is condensed
    if 'heater_1' in list_unit:
        m.fs.heater_1.cost_heater = Expression(
            expr=0.036158*m.fs.heater_1.heat_duty[0] + 63931.475,
            doc='capital cost of heater in $')
        expression += ' + m.fs.heater_1.cost_heater'

    if 'heater_2' in list_unit:
        m.fs.heater_2.cost_heater = Expression(
            expr=0.036158*m.fs.heater_2.heat_duty[0] + 63931.475,
            doc='capital cost of heater in $')
        expression += ' + m.fs.heater_2.cost_heater'

    # Surrogates prepared with IDAES shell and tube hx considering cooling
    # water assuming that water inlet T is 25 deg C and outlet T is 40 deg C
    if 'cooler_1' in list_unit:
        m.fs.cooler_1.cost_heater = Expression(
            expr=0.10230*(-m.fs.cooler_1.heat_duty[0]) + 100421.572,
            doc='capital cost of cooler in $')
        expression += ' + m.fs.cooler_1.cost_heater'

    if 'cooler_2' in list_unit:
        m.fs.cooler_2.cost_heater = Expression(
            expr=0.10230*(-m.fs.cooler_2.heat_duty[0]) + 100421.572,
            doc='capital cost of cooler in $')
        expression += ' + m.fs.cooler_2.cost_heater'

    expression += ') * 5.4 / 15)'
    exec(expression)

    # methanol price $449 us dollars per metric ton  - 32.042 g/mol
    # - 1 gr = 1e-6 MT  -- consider 1000
    # H2 $16.51 per kilogram - 2.016 g/mol
    # CO $62.00 per kilogram - 28.01 g/mol
    expression = 'm.fs.sales = Expression(expr=(0.0'
    if 'flash_0' in list_unit:
        expression += ' + m.fs.flash_0.liq_outlet.flow_mol[0] * m.fs.flash_0.liq_outlet.mole_frac_comp[0, "CH3OH"] * 32.042 * 1e-6 * 449'

    expression += ') * 3600 *24 *365)'
    exec(expression)

    expression = 'm.fs.raw_mat_cost = Expression(expr=(0.0'
    if 'mixer_0' in list_unit:
        expression += ' + m.fs.mixer_0.inlet_1.flow_mol[0] * 16.51 * 2.016 / 1000'
        expression += ' + m.fs.mixer_0.inlet_2.flow_mol[0] * 62.00 * 28.01 / 1000'
    expression += ') * 3600 * 24 * 365)'
    exec(expression)

    m.fs.revenue = Expression(expr=(m.fs.sales - m.fs.operating_cost - \
        m.fs.annualized_capital_cost - m.fs.raw_mat_cost)/1000)

    m.fs.cost_mol = Expression(expr=(m.fs.operating_cost + m.fs.annualized_capital_cost)/m.fs.meth_flow)

    print("\nDegrees of Freedom (add costing) = %d" % degrees_of_freedom(m))

    # if a unit is used, check for costing scaling and get scaling factors
    for unit in ['flash_0', 'flash_1', 'StReactor_1', 'StReactor_2']:
        if unit in list_unit:
            costing_module = getattr(m.fs, unit).costing  # shorter alias
            for var in costing_module.component_data_objects(Var):
                if iscale.get_scaling_factor(var) is None:
                    iscale.set_scaling_factor(var, 1)
            costing.calculate_scaling_factors(costing_module)

def report(m, list_unit):

    print("\nDisplay some results:")

    if 'StReactor_1' in list_unit:
        print('StReactor 1 Reaction conversion (0.75): ', 
        m.fs.StReactor_1.conversion.value)
    if 'StReactor_2' in list_unit:
        print('StReactor 2 Reaction conversion (0.75): ', 
        m.fs.StReactor_2.conversion.value)
    if 'flash_0' in list_unit:
        print('Methanol recovery(%): ', value(100*m.fs.flash_0.recovery))
        print('CH3OH flow rate (mol/s): ', value(m.fs.meth_flow))
        print("methanol production rate(%): ", value(m.fs.efficiency)*100)
        print('cost per mol product: ', value(m.fs.cost_mol))
    print('annualized capital cost ($/year) =', value(m.fs.annualized_capital_cost))
    print('operating cost ($/year) = ', value(m.fs.operating_cost))
    print('sales ($/year) = ', value(m.fs.sales))
    print('raw materials cost ($/year) =', value(m.fs.raw_mat_cost))
    print('revenue (1000$/year)= ', value(m.fs.revenue))

def add_bounds_v1(m, list_unit):

    # Set up Optimization Problem (Maximize Revenue)
    # keep process pre-reaction fixed and unfix some post-process specs
    
    if 'StReactor_1' in list_unit:
        # m.fs.StReactor_1.conversion.unfix()
        # m.fs.StReactor_1.conversion_lb = Constraint(expr=m.fs.StReactor_1.conversion >= 0.75)
        # m.fs.StReactor_1.conversion_ub = Constraint(expr=m.fs.StReactor_1.conversion <= 0.85)
        m.fs.StReactor_1.outlet_temp.deactivate()
        m.fs.StReactor_1.outlet_t_lb = Constraint(
            expr=m.fs.StReactor_1.control_volume.properties_out[0.0].temperature >= 405)
        m.fs.StReactor_1.outlet_t_ub = Constraint(
            expr=m.fs.StReactor_1.control_volume.properties_out[0.0].temperature <= 505) #solving process is very sensitive to this temperature

    if 'StReactor_2' in list_unit:
        # m.fs.StReactor_2.conversion.unfix()
        # m.fs.StReactor_2.conversion_lb = Constraint(expr=m.fs.StReactor_2.conversion >= 0.75)
        # m.fs.StReactor_2.conversion_ub = Constraint(expr=m.fs.StReactor_2.conversion <= 0.85)
        m.fs.StReactor_2.outlet_temp.deactivate()
        m.fs.StReactor_2.outlet_t_lb = Constraint(
            expr=m.fs.StReactor_2.control_volume.properties_out[0.0].temperature >= 405)
        m.fs.StReactor_2.outlet_t_ub = Constraint(
            expr=m.fs.StReactor_2.control_volume.properties_out[0.0].temperature <= 505)

    # Optimize turbine work (or delta P)
    if 'expander_1' in list_unit:
        m.fs.expander_1.deltaP.unfix()  # optimize turbine work recovery/pressure drop
        m.fs.expander_1.outlet_p_lb = Constraint(
            expr=m.fs.expander_1.outlet.pressure[0] >= 10E5)
        m.fs.expander_1.outlet_p_ub = Constraint(
            expr=m.fs.expander_1.outlet.pressure[0] <= 51E5*0.8)
        # m.fs.expander_1.deltaP.setlb(-2.5e6)
        # m.fs.expander_1.deltaP.setlb(-1.5e6)

    if 'expander_2' in list_unit:
        m.fs.expander_2.deltaP.unfix()  # optimize turbine work recovery/pressure drop
        m.fs.expander_2.outlet_p_lb = Constraint(
            expr=m.fs.expander_2.outlet.pressure[0] >= 10E5)
        m.fs.expander_2.outlet_p_ub = Constraint(
            expr=m.fs.expander_2.outlet.pressure[0] <= 51E5*0.8)
        # m.fs.expander_2.deltaP.setlb(-2.5e6)
        # m.fs.expander_2.deltaP.setlb(-1.5e6)

    # # Optimize cooler outlet temperature - unfix cooler outlet temperature
    # if 'cooler_1' in list_unit: 
    #     m.fs.cooler_1.outlet_temp.deactivate()
    #     m.fs.cooler_1.outlet_t_lb = Constraint(
    #         expr=m.fs.cooler_1.control_volume.properties_out[0.0].temperature
    #         >= 407.15*0.9)
    #     m.fs.cooler_1.outlet_t_ub = Constraint(
    #         expr=m.fs.cooler_1.control_volume.properties_out[0.0].temperature
    #         <= 407.15*1.1)
    #     # m.fs.cooler_1.heat_duty.setub(0)

    # if 'cooler_2' in list_unit: 
    #     m.fs.cooler_2.outlet_temp.deactivate()
    #     m.fs.cooler_2.outlet_t_lb = Constraint(
    #         expr=m.fs.cooler_2.control_volume.properties_out[0.0].temperature
    #         >= 407.15*0.9)
    #     m.fs.cooler_2.outlet_t_ub = Constraint(
    #         expr=m.fs.cooler_2.control_volume.properties_out[0.0].temperature
    #         <= 407.15*1.1)
    #     # m.fs.cooler_2.heat_duty.setub(0)
        
    # Optimize heater properties
    if 'heater_1' in list_unit:
        m.fs.heater_1.outlet_temp.deactivate()
        m.fs.heater_1.outlet_t_lb = Constraint(
            expr=m.fs.heater_1.control_volume.properties_out[0.0].temperature
            >= 480)
        m.fs.heater_1.outlet_t_ub = Constraint(
            expr=m.fs.heater_1.control_volume.properties_out[0.0].temperature
            <= 490)
        # m.fs.heater_1.heat_duty.setlb(0)
    
    if 'heater_2' in list_unit:
        m.fs.heater_2.outlet_temp.deactivate()
        m.fs.heater_2.outlet_t_lb = Constraint(
            expr=m.fs.heater_2.control_volume.properties_out[0.0].temperature
            >= 480)
        m.fs.heater_2.outlet_t_ub = Constraint(
            expr=m.fs.heater_2.control_volume.properties_out[0.0].temperature
            <= 490)
        # m.fs.heater_2.heat_duty.setlb(0)
    
    # Optimize flash properties
    if 'flash_0' in list_unit:
        # option 3
        m.fs.flash_0.deltaP.unfix()  # allow pressure change in streams

    if 'flash_1' in list_unit:
        # option 3
        # m.fs.flash_1.deltaP.unfix()
        # option 1
        m.fs.flash_1.deltaP.unfix()  # allow pressure change in streams
        m.fs.flash_1.isothermal = Constraint(
            expr=m.fs.flash_1.control_volume.properties_out[0].temperature ==
            m.fs.flash_1.control_volume.properties_in[0].temperature)
        # option 2
        # m.fs.flash_1.outlet_temp.deactivate()
        # m.fs.flash_1.outlet_t_lb = Constraint(
        #     expr=m.fs.flash_1.control_volume.properties_out[0.0].temperature
        #     >= 400)
        # m.fs.flash_1.outlet_t_ub = Constraint(
        #     expr=m.fs.flash_1.control_volume.properties_out[0.0].temperature
        #     <= 410)

    if 'flash_2' in list_unit:
        # option 3
        # m.fs.flash_2.deltaP.unfix()
        # option 1
        m.fs.flash_2.deltaP.unfix()  # allow pressure change in streams
        m.fs.flash_2.isothermal = Constraint(
            expr=m.fs.flash_2.control_volume.properties_out[0].temperature ==
            m.fs.flash_2.control_volume.properties_in[0].temperature)
        # option 2
        # m.fs.flash_2.outlet_temp.deactivate()
        # m.fs.flash_2.outlet_t_lb = Constraint(
        #     expr=m.fs.flash_2.control_volume.properties_out[0.0].temperature
        #     >= 400)
        # m.fs.flash_2.outlet_t_ub = Constraint(
        #     expr=m.fs.flash_2.control_volume.properties_out[0.0].temperature
        #     <= 410)

    # Optimize splitter properties
    if 'splitter_1' in list_unit:
        m.fs.splitter_1.split_fraction[0, "outlet_2"].unfix()
        m.fs.splitter_1.split_fraction_lb = \
            Constraint(expr=m.fs.splitter_1.split_fraction[0, "outlet_2"] >= 0.10)
        m.fs.splitter_1.split_fraction_ub = \
            Constraint(expr=m.fs.splitter_1.split_fraction[0, "outlet_2"] <= 0.60)

    if 'splitter_2' in list_unit:
        m.fs.splitter_2.split_fraction[0, "outlet_2"].unfix()
        m.fs.splitter_2.split_fraction_lb = \
            Constraint(expr=m.fs.splitter_2.split_fraction[0, "outlet_2"] >= 0.10)
        m.fs.splitter_2.split_fraction_ub = \
            Constraint(expr=m.fs.splitter_2.split_fraction[0, "outlet_2"] <= 0.60)

    if 'flash_0' in list_unit:
        m.fs.system_efficiency = Constraint(expr=m.fs.efficiency >= 0.4)

    print("\nDegrees of Freedom (add bounds) = %d" % degrees_of_freedom(m))

def examples(i):
    
    if i == 1:
        flowsheet_name = 'methanol_base_'+str(i)
        # straight line - base example
        list_unit = ['mixer_0', 'compressor_1', 'heater_1', 'StReactor_1', 'expander_1', \
            'cooler_1', 'flash_0', 'exhaust_2', 'product_0']
        list_inlet = ['compressor_1.inlet', 'heater_1.inlet', 'StReactor_1.inlet', \
            'expander_1.inlet', 'cooler_1.inlet', 'flash_0.inlet', \
                'exhaust_2.inlet', 'product_0.inlet']
        list_outlet = ['mixer_0.outlet', 'compressor_1.outlet', 'heater_1.outlet', \
            'StReactor_1.outlet', 'expander_1.outlet', 'cooler_1.outlet', \
                'flash_0.vap_outlet', 'flash_0.liq_outlet']

    if i == 2:
        flowsheet_name = 'methanol_base_'+str(i)
        # remove cooler_1
        list_unit = ['mixer_0', 'compressor_1', 'heater_1', 'StReactor_1', 'expander_1', 'flash_0', 'exhaust_2', 'product_0']
        list_inlet = ['compressor_1.inlet', 'heater_1.inlet', 'StReactor_1.inlet', \
            'expander_1.inlet', 'flash_0.inlet', 'exhaust_2.inlet', \
                'product_0.inlet']
        list_outlet = ['mixer_0.outlet', 'compressor_1.outlet', 'heater_1.outlet', \
            'StReactor_1.outlet', 'expander_1.outlet', 'flash_0.vap_outlet', \
                'flash_0.liq_outlet']
    
    if i == 2.5:
        flowsheet_name = 'methanol_base_'+str(i)
        # remove cooler_1, expander_1
        list_unit = ['mixer_0', 'compressor_1', 'heater_1', 'StReactor_1', 'flash_0', 'exhaust_2', 'product_0']
        list_inlet = ['compressor_1.inlet', 'heater_1.inlet', 'StReactor_1.inlet', \
            'flash_0.inlet', 'exhaust_2.inlet', 'product_0.inlet']
        list_outlet = ['mixer_0.outlet', 'compressor_1.outlet', 'heater_1.outlet', \
            'StReactor_1.outlet', 'flash_0.vap_outlet', 'flash_0.liq_outlet']

    if i == 2.75:
        flowsheet_name = 'methanol_base_'+str(i)
        # remove cooler_1, expander_1, compressor_1
        list_unit = ['mixer_0', 'heater_1', 'StReactor_1', 'flash_0', 'exhaust_2', 'product_0']
        list_inlet = ['heater_1.inlet', 'StReactor_1.inlet', 'flash_0.inlet', \
            'exhaust_2.inlet', 'product_0.inlet']
        list_outlet = ['mixer_0.outlet', 'heater_1.outlet', 'StReactor_1.outlet', \
            'flash_0.vap_outlet', 'flash_0.liq_outlet']

    if i == 3:
        # replace cooler_1 by flash_1
        flowsheet_name = 'methanol_base_'+str(i)
        list_unit = ['mixer_0', 'compressor_1', 'heater_1', 'StReactor_1', 'expander_1', \
            'flash_1', 'flash_0', 'exhaust_1', 'exhaust_2', 'product_0']
        list_inlet = ['compressor_1.inlet', 'heater_1.inlet', 'StReactor_1.inlet', \
            'expander_1.inlet', 'flash_1.inlet', 'flash_0.inlet', \
                'exhaust_1.inlet', 'exhaust_2.inlet', 'product_0.inlet']
        list_outlet = ['mixer_0.outlet', 'compressor_1.outlet', 'heater_1.outlet', \
            'StReactor_1.outlet', 'expander_1.outlet', 'flash_1.liq_outlet', \
                'flash_1.vap_outlet', 'flash_0.vap_outlet', 'flash_0.liq_outlet']
    
    if i == 4:
        flowsheet_name = 'methanol_base_'+str(i)
        # recycle flash_1 by splitter_1, and recycle splitter_1.outlet_2
        list_unit = ['mixer_0', 'mixer_1', 'compressor_1', \
            'compressor_2', 'heater_1', 'StReactor_1', 'expander_1', \
            'flash_0', 'splitter_1', 'exhaust_2', 'product_0']
        list_inlet = ['mixer_1.inlet_1', 'compressor_1.inlet', 'heater_1.inlet', 'StReactor_1.inlet', \
            'expander_1.inlet', 'splitter_1.inlet', 'flash_0.inlet', 'compressor_2.inlet', \
                'mixer_1.inlet_2', 'exhaust_2.inlet', 'product_0.inlet']
        list_outlet = ['mixer_0.outlet', 'mixer_1.outlet', 'compressor_1.outlet', 'heater_1.outlet', \
            'StReactor_1.outlet', 'expander_1.outlet', 'splitter_1.outlet_2', 'splitter_1.outlet_1', \
                'compressor_2.outlet', 'flash_0.vap_outlet', 'flash_0.liq_outlet']
    
    if i == 5:
        # recycle flash_1.vap_outlet
        flowsheet_name = 'methanol_base_'+str(i)
        list_unit = ['mixer_0', 'mixer_1', 'compressor_1', 'heater_1', 'StReactor_1', 'expander_1', \
            'flash_1', 'flash_0', 'exhaust_1', 'splitter_1', 'exhaust_2', 'product_0']
        list_inlet = ['mixer_1.inlet_1', 'compressor_1.inlet', 'heater_1.inlet', 'StReactor_1.inlet', \
            'expander_1.inlet', 'flash_1.inlet', 'flash_0.inlet', 'splitter_1.inlet', 
            'exhaust_1.inlet', 'mixer_1.inlet_2', 'exhaust_2.inlet', 'product_0.inlet']
        list_outlet = ['mixer_0.outlet', 'mixer_1.outlet', 'compressor_1.outlet', 'heater_1.outlet', \
            'StReactor_1.outlet', 'expander_1.outlet', 'flash_1.liq_outlet', 'flash_1.vap_outlet', \
                'splitter_1.outlet_2', 'splitter_1.outlet_1', 'flash_0.vap_outlet', 'flash_0.liq_outlet']
    
    if i == 6:
        flowsheet_name = 'methanol_base_'+str(i)
        # recycle flash_0.vap_outlet with splitter_1 (no cooler)
        list_unit = ['mixer_0', 'mixer_1', 'compressor_1', 'compressor_2', \
            'heater_1', 'StReactor_1', 'expander_1', 'flash_0', 'splitter_1', 'exhaust_1', 'product_0']
        list_inlet = ['mixer_1.inlet_1', 'compressor_1.inlet', 'heater_1.inlet', 'StReactor_1.inlet', \
            'expander_1.inlet', 'flash_0.inlet', 'mixer_1.inlet_2', 'splitter_1.inlet', \
                'exhaust_1.inlet', 'compressor_2.inlet', 'product_0.inlet']
        list_outlet = ['mixer_0.outlet', 'mixer_1.outlet', 'compressor_1.outlet', 'heater_1.outlet', \
            'StReactor_1.outlet', 'expander_1.outlet', 'compressor_2.outlet', 'flash_0.vap_outlet', \
                'splitter_1.outlet_2', 'splitter_1.outlet_1', 'flash_0.liq_outlet']

    if i == 7:
        flowsheet_name = 'methanol_base_'+str(i)
        # recycle flash_0.vap_outlet with splitter_1 (with cooler)
        list_unit = ['mixer_0', 'mixer_1', 'compressor_1', 'heater_1', \
            'cooler_1', 'StReactor_1', 'expander_1', 'flash_0', 'splitter_1', 'exhaust_1', 'product_0']
        list_inlet = ['mixer_1.inlet_1', 'compressor_1.inlet', 'heater_1.inlet', 'StReactor_1.inlet', \
            'expander_1.inlet', 'cooler_1.inlet', 'flash_0.inlet', 'splitter_1.inlet', \
                'exhaust_1.inlet', 'mixer_1.inlet_2', 'product_0.inlet']
        list_outlet = ['mixer_0.outlet', 'mixer_1.outlet', 'compressor_1.outlet', 'heater_1.outlet', \
            'StReactor_1.outlet', 'expander_1.outlet', 'cooler_1.outlet', 'flash_0.vap_outlet', \
                'splitter_1.outlet_2', 'splitter_1.outlet_1', 'flash_0.liq_outlet']
        
    if i == 8:
        flowsheet_name = 'methanol_base_'+str(i)
        # recycle flash_0.vap_outlet with splitter_1 (with cooler, switch compressor_1 and heater_1)
        list_unit = ['mixer_0', 'mixer_1', 'compressor_1', 'heater_1', 'cooler_1', 'StReactor_1', 'expander_1', \
            'flash_0', 'splitter_1', 'exhaust_1', 'product_0']
        list_inlet = ['mixer_1.inlet_1', 'heater_1.inlet', 'compressor_1.inlet', 'StReactor_1.inlet', \
            'expander_1.inlet', 'cooler_1.inlet', 'flash_0.inlet', 'splitter_1.inlet', \
                'exhaust_1.inlet', 'mixer_1.inlet_2', 'product_0.inlet']
        list_outlet = ['mixer_0.outlet', 'mixer_1.outlet', 'heater_1.outlet', 'compressor_1.outlet', \
            'StReactor_1.outlet', 'expander_1.outlet', 'cooler_1.outlet', 'flash_0.vap_outlet', \
                'splitter_1.outlet_2', 'splitter_1.outlet_1', 'flash_0.liq_outlet']
        
    return list_unit, list_inlet, list_outlet, flowsheet_name
        
#%%
def run_optimization(flowsheet_name, list_unit, list_inlet, list_outlet, visualize_flowsheet = False):
    
    # start from initial score of 500
    score = 500
    extra_score = 0
    delta_scoreA = 1000  #bonus/penalty option 1
    delta_scoreB = 500   #bonus/penalty option 2
    status = ['infeasible', 0.0, 0.0] # store status, cost/mol, system efficiency
    costs = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    # print the lists of unit, inlet and outlet
    print('\nlist of units, inlets, outlets:')
    print(list_unit)
    print(list_inlet)
    print(list_outlet)

    # build flowsheet and run simulation
    m = ConcreteModel(name=flowsheet_name)
    m.fs = FlowsheetBlock(default={"dynamic": False})

    # if the inputs cannot build a flowsheet
    try:
        build_model(m, list_unit, list_inlet, list_outlet, scale_thermo=True)  # build flowsheet
        scale_variables(m, list_unit) # add better scaling factors for unit variables
        scale_constraints(m, list_unit) # add better scaling factors for unit constraints
        set_inputs(m, list_unit)  # unit and stream specifications
        initialize_flowsheet(m, list_unit, list_inlet, list_outlet, 5)  # rigorous initialization scheme
        score = score + delta_scoreB # bonus for passing the initialization process
    except:
        print('initialization process: aborted or failed')
        return score, extra_score, status, costs

    # pre solve
    solver = get_solver() # create the solver object
    try:
        # solver.options = {'tol': 1e-6, 'max_iter': 5000, 'halt_on_ampl_error': 'yes'}
        solver.options = {'tol': 1e-7, 'max_iter': 5000}
        results = solver.solve(m, tee=False)
        print('pre solve - physical operational? ', results.solver.termination_condition.value)
    except:
        print('pre solve - aborted or failed')

    # initial solve
    add_costing(m, list_unit)  # re-solve with costing equations
    # m.fs.objective = Objective(expr=m.fs.revenue, sense=maximize)
    # m.fs.objective = Objective(expr=m.fs.meth_flow, sense=maximize)
    m.fs.objective = Objective(expr=m.fs.cost_mol, sense=minimize)

    try:
        # solver.options = {'tol': 1e-6, 'max_iter': 5000, 'halt_on_ampl_error': 'yes'}
        solver.options = {'tol': 1e-7, 'max_iter': 5000}
        results = solver.solve(m, tee=False)
        print('initial solve - physical operational? ', results.solver.termination_condition.value)

        score = score + delta_scoreA # bonus for passing the initial solve
        # status = [results.solver.termination_condition.value, value(m.fs.cost_mol), value(m.fs.efficiency)]
        # costs = [value(m.fs.annualized_capital_cost), value(m.fs.operating_cost), 
        #          value(m.fs.sales), value(m.fs.raw_mat_cost), value(m.fs.revenue)]
        if results.solver.termination_condition.value == 'optimal':
            report(m, list_unit)
    except:
        print('initial solve - aborted or failed')
    
    # # save initial solve results
    # status_init = status
    # costs_init = costs

    # optimal solve
    add_bounds_v1(m, list_unit)
    try:
        # solver.options = {'tol': 1e-6, 'max_iter': 5000, 'halt_on_ampl_error': 'yes'}
        solver.options = {'tol': 1e-7, 'max_iter': 5000}
        results = solver.solve(m, tee=False)
        print('optimal solve - physical operational? ', results.solver.termination_condition.value)
        
        score = score + delta_scoreA
        status = [results.solver.termination_condition.value, value(m.fs.cost_mol), value(m.fs.efficiency)]
        costs = [value(m.fs.annualized_capital_cost), value(m.fs.operating_cost), 
                 value(m.fs.sales), value(m.fs.raw_mat_cost), value(m.fs.revenue)]
        if results.solver.termination_condition.value == 'optimal':
            report(m, list_unit)
    except:
        print('optimal solve - aborted or failed')

    # # in case of no optimal solution
    # if status[0] != 'optimal':
    #     if status_init[0] == 'optimal':
    #         status = status_init
    #         costs = costs_init

    # score: update according to system efficiency
    if status[0] == 'optimal':
        score = 5000
        extra_score = (status[2]-0.40)/0.20*delta_scoreA
        # end = timer()
        # status[1] = end-start

    # visualize flowsheet
    if visualize_flowsheet == True:
        m.fs.visualize(flowsheet_name)
    
    # m.fs.flash_1.report()
    # m.fs.flash_0.report()
    
    return score, extra_score, status, costs

#%%
if __name__ == "__main__":

    start = timer()
    status_list = []
    
    for i in [2.5, 2.75, 1, 2, 3, 4, 5, 6, 7, 8]: #[2.5, 2.75, 1, 2, 3, 4, 5, 6, 7, 8]
        
        print('\n\n------------------------------------------------------------\n\n')
        print('Evaluate example ', i)
        list_unit, list_inlet, list_outlet, flowsheet_name = examples(i)
        score, extra_score, status, costs = run_optimization(flowsheet_name, list_unit, list_inlet, list_outlet, visualize_flowsheet = False)
        status_list.append(status[0])

        # # pre-screen
        # sys.path.append(os.path.abspath("../"))
        # from RL_ENV import pre_screen
        # list_unit = np.array(list_unit, dtype=str)
        # list_inlet = np.array([x.split(".") for x in list_inlet], dtype=str)
        # list_outlet = np.array([x.split(".") for x in list_outlet], dtype=str)
        
        # constraints_consume = np.zeros((1, 12))
        # pres_score, constraints_consume = pre_screen(list_unit, list_inlet, list_outlet, constraints_consume)
        # print("pre-screening with reward: ", pres_score, flush=True)

    end = timer()
    print('\nTime consuming: ', end-start, ' s', flush=True)
    print('Optimization status:')
    print(status_list)


