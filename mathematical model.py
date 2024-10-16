import sys

import gurobipy as gp
import numpy as np
from gurobipy import GRB, max_
import pandas as pd
from pygantt.pygantt import *
import pandas as pd
import re

sys.path.append("../")

try:
    # Read Instance
    data_csv = pd.read_csv("../", skip_blank_lines=True, index_col=False)

    position_dict = table_find_pos(data_csv, ['Factories', 'Families', 'Machines', 'Total number of jobs', 'Scenarios',
                                              'Number of Jobs in each Family', 'Jobs in each Family',
                                              'Under scenario', 'On machine', 'Time Window',
                                              'Weight'])

    # Preference coefficient
    p = 0.95

    # A sufficient large positive number
    h = 0x0000ffff
    # Number of jobs
    num_jobs = int(data_csv.iloc[position_dict['Total number of jobs'][0][1] + 1,
    data_csv.columns.get_loc(position_dict['Total number of jobs'][0][0])])
    # Number of machines
    num_machines = int(data_csv.iloc[position_dict['Machines'][0][1],
    data_csv.columns.get_loc(position_dict['Machines'][0][0]) + 1])
    # Number of groups
    num_groups = int(data_csv.iloc[position_dict['Families'][0][1],
    data_csv.columns.get_loc(position_dict['Families'][0][0]) + 1])
    # Number of factories
    num_factories = int(data_csv.iloc[position_dict['Factories'][0][1],
    data_csv.columns.get_loc(position_dict['Factories'][0][0]) + 1])
    # Number of scenarios
    num_scenarios = int(data_csv.iloc[position_dict['Scenarios'][0][1],
    data_csv.columns.get_loc(position_dict['Scenarios'][0][0]) + 1])

    data_row = position_dict['Jobs in each Family'][0][1] + 1

    # Jobs in each group
    jobs_in_each_group = [np.array([0])]
    for l in np.arange(0, num_groups):
        jobs_in_each_group.append(np.insert(np.array(data_csv.loc[data_row, :].dropna().astype(int)) + 1, 0, 0))
        data_row = data_row + 1
    jobs_in_each_group = np.array(jobs_in_each_group, dtype=object)

    # The processing time of jobs under different scenarios
    # The first dimension of process_time represents different scenarios.
    process_time = np.empty((num_scenarios, num_jobs + 1, num_machines + 1), dtype=int)
    for s, scenario_position in enumerate(position_dict['Under scenario']):
        data_row = scenario_position[1] + 1
        process_time_under_scenario = np.array(data_csv.iloc[data_row:data_row + num_jobs, 0:num_machines].astype(int))
        process_time_under_scenario = np.insert(process_time_under_scenario, 0, 0, axis=0)
        process_time_under_scenario = np.insert(process_time_under_scenario, 0, 0, axis=1)
        process_time[s] = process_time_under_scenario

    # The setup time between groups on different machines
    setup_time = np.array([np.zeros((num_groups + 1, num_groups + 1), dtype=int)])
    for setup_time_position in position_dict['On machine']:
        data_row = setup_time_position[1] + 1
        setup_time_on_one_machine = np.array(data_csv.iloc[data_row: data_row + num_groups,
                                             0:num_groups].astype(int))
        setup_time_on_one_machine = np.insert(setup_time_on_one_machine, 0,
                                              setup_time_on_one_machine[np.diag_indices(num_groups)], 0)
        setup_time_on_one_machine = np.insert(setup_time_on_one_machine, 0,
                                              np.zeros((1, num_groups + 1), dtype=int), 1)
        setup_time = np.append(setup_time, [setup_time_on_one_machine], 0)

    data_row = position_dict['Time Window'][0][1] + 1
    # Delivery time windows of each group
    TimeWindow = np.array(data_csv.iloc[data_row: data_row + num_groups, 0:2].astype(int))
    TimeWindow = np.insert(TimeWindow, 0, np.zeros((1, 2), dtype=int), 0)

    data_row = position_dict['Weight'][0][1] + 1
    # Earliness weight and tardiness weight of each group
    Weight = np.array(data_csv.iloc[data_row: data_row + num_groups, 0:2].astype(int))
    Weight = np.insert(Weight, 0, np.zeros((1, 2), dtype=int), 0)

    group_array = np.arange(0, num_groups + 1)
    machine_array = np.arange(1, num_machines + 1)
    scenario_array = np.arange(num_scenarios)

    positions_in_each_group = [np.array([0])]
    for l in group_array[1:]:
        positions_in_each_group.append(np.arange(len(jobs_in_each_group[l])))
    positions_in_each_group = np.array(positions_in_each_group, dtype=object)

    positions_of_group = np.arange(num_groups + 1)

    # Create a new model
    model = gp.Model("DBFGSP_UPTs_TWET")
    model.setParam(GRB.Param.TimeLimit, 3600)
    model.setParam('NonConvex', 2)

    # Create decision variables
    # Completion time of jobs
    c = {}
    for s in scenario_array:
        for l in group_array[1:]:
            c[s, l] = model.addVars(positions_in_each_group[l][1:], [l], machine_array, vtype=GRB.INTEGER,
                                    name=f"c[{s}]")

    # Departure time of jobs
    d = {}
    for s in scenario_array:
        for l in group_array[1:]:
            d[s, l] = model.addVars(positions_in_each_group[l][1:], [l], machine_array, vtype=GRB.INTEGER,
                                    name=f"d[{s}]")

    # Completion time of groups
    c_fam = {}
    for s in scenario_array:
        for l in group_array[1:]:
            c_fam[s, l] = model.addVar(vtype=GRB.INTEGER, name=f"c_fam[{s},{l}]")

    # Binary decision variables
    x = model.addVars(positions_of_group[1:], group_array[1:], vtype=GRB.BINARY, name="x")
    xx = model.addVars(positions_of_group[1:], vtype=GRB.BINARY, name="xx")

    y = {}
    for l in group_array[1:]:
        y[l] = model.addVars(positions_in_each_group[l][1:], jobs_in_each_group[l][1:], [l], vtype=GRB.BINARY, name="y")

    # Total earliness of each group under different scenarios
    Earliness = model.addVars(scenario_array, group_array[1:], vtype=GRB.INTEGER, name="Earliness")
    # Total tardiness of each group under different scenarios
    Tardiness = model.addVars(scenario_array, group_array[1:], vtype=GRB.INTEGER, name="Tardiness")
    # Total weighted earliness and tardiness under different scenarios
    TWET = model.addVars(scenario_array, vtype=GRB.INTEGER, name="TWET")
    # Mean of TWET under all scenarios
    Mean = model.addVar(vtype=GRB.CONTINUOUS, name="TWET_mean")
    # Variance under each scenario
    variance = model.addVars(scenario_array, vtype=GRB.CONTINUOUS, name="variance")
    avg_variance = model.addVar(vtype=GRB.CONTINUOUS, name="avg_varinance")
    # Standard deviation of TWET under all scenarios
    Std = model.addVar(vtype=GRB.CONTINUOUS, name="TWET_std")
    # Robust objective
    RO = model.addVar(vtype=GRB.CONTINUOUS, name="robust_obj")

    model.setParam(GRB.Param.IntFeasTol, 1e-9)

    # Set objective
    # (1)
    model.setObjective(RO, GRB.MINIMIZE)

    # Add constraints
    # (2)
    model.addConstrs(gp.quicksum(x[p, l] for p in positions_of_group[1:]) == 1
                     for l in group_array[1:])

    # (3)
    model.addConstrs(gp.quicksum(x[p, l] for l in group_array[1:]) == 1
                     for p in positions_of_group[1:])

    # (4)
    model.addConstr(xx[1] == 1)
    # (5)
    model.addConstr(gp.quicksum(xx[p] for p in positions_of_group[1:]) <= num_factories)

    # (6)
    model.addConstrs(gp.quicksum(y[l][p, j, l] for p in positions_in_each_group[l][1:]) == 1
                     for l in group_array[1:]
                     for j in jobs_in_each_group[l][1:])

    # (7)
    model.addConstrs(gp.quicksum(y[l][p, j, l] for j in jobs_in_each_group[l][1:]) == 1
                     for l in group_array[1:]
                     for p in positions_in_each_group[l][1:])

    # (8)
    model.addConstrs(d[s, l][p, l, i] >= c[s, l][p, l, i]
                     for s in scenario_array
                     for l in group_array[1:]
                     for p in positions_in_each_group[l][1:]
                     for i in machine_array[:-1])

    model.addConstrs(d[s, l][p, l, machine_array[-1]] == c[s, l][p, l, machine_array[-1]]
                     for s in scenario_array
                     for l in group_array[1:]
                     for p in positions_in_each_group[l][1:])

    # (9)
    model.addConstrs(c[s, l][p, l, i] >= d[s, l][p-1, l, i] + gp.quicksum(process_time[s, j, i]*y[l][p, j, l] for j in jobs_in_each_group[l][1:])
                     for s in scenario_array
                     for l in group_array[1:]
                     for p in positions_in_each_group[l][2:]
                     for i in machine_array)

    # (10)
    model.addConstrs(
        c[s, l1][1, l1, i] >= d[s, l][positions_in_each_group[l][-1], l, i] + setup_time[i, l, l1]
        + gp.quicksum(process_time[s, j1, i] * y[l1][1, j1, l1] for j1 in jobs_in_each_group[l1][1:])
        + (x[p-1, l] + x[p, l1] - xx[p] - 2) * h
        for s in scenario_array
        for l in group_array[1:]
        for l1 in group_array[1:]
        if l != l1
        for p in positions_of_group[2:]
        for i in machine_array)

    # (11)
    model.addConstrs(c[s, l][1, l, i] >= setup_time[i, 0, l]
                     + gp.quicksum(process_time[s, j, i] * y[l][1, j, l] for j in jobs_in_each_group[l][1:])
                     + (x[p, l] + xx[p] - 2) * h
                     for s in scenario_array
                     for l in group_array[1:]
                     for p in positions_of_group[1:]
                     for i in machine_array)
    # (12)
    model.addConstrs(c[s, l][p, l, i + 1] == d[s, l][p, l, i] + gp.quicksum(process_time[s, j, i+1] * y[l][p, j, l] for j in jobs_in_each_group[l][1:])
                     for s in scenario_array
                     for l in group_array[1:]
                     for p in positions_in_each_group[l][1:]
                     for i in machine_array[:-1])

    # (13)
    model.addConstrs(c_fam[s, l] == d[s, l][positions_in_each_group[l][-1], l, num_machines]
                     for s in scenario_array
                     for l in group_array[1:])
    # (14)
    model.addConstrs(Earliness[s, l] >= TimeWindow[l][0] - c_fam[s, l]
                     for s in scenario_array
                     for l in group_array[1:])

    # (15)
    model.addConstrs(Tardiness[s, l] >= c_fam[s, l] - TimeWindow[l][1]
                     for s in scenario_array
                     for l in group_array[1:])

    # (16)
    model.addConstrs(TWET[s] == gp.quicksum((Weight[l][0] * Earliness[s, l] + Weight[l][1] * Tardiness[s, l])
                                            for l in group_array[1:])
                     for s in scenario_array)

    # (17)
    model.addConstr(Mean == (gp.quicksum(TWET[s] for s in scenario_array) / num_scenarios))

    # (18)
    model.addConstrs(variance[s] == (TWET[s] - Mean) * (TWET[s] - Mean)
                     for s in scenario_array)

    model.addConstr(avg_variance == gp.quicksum(variance[s] for s in scenario_array) / num_scenarios)

    model.addConstr(Std * Std == avg_variance)


    model.addConstr(RO == p * Mean + (1 - p) * Std)

    # Optimize model
    model.optimize()


    for v in model.getVars():
        print('%s %g' % (v.Varname, v.X))

    print('Obj: %g' % model.ObjVal)

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))
except AttributeError as e:
    print('Encountered an attribute error')
