# model/model_nile.py

import numpy as np
import pandas as pd
from .model_classes import Reservoir, Catchment, IrrigationDistrict, HydropowerPlant
from .smash import Policy

class ModelNile:
    def __init__(self):
        self.read_settings_file("../settings/settings_file_Nile.xlsx")
        self.catchments = {name: Catchment(name) for name in self.catchment_names}
        self.irr_districts = {name: IrrigationDistrict(name) for name in self.irr_district_names}
        self.reservoirs = {}

        for name in self.reservoir_names:
            new_reservoir = Reservoir(name)
            new_plant = HydropowerPlant(new_reservoir)
            new_reservoir.hydropower_plants.append(new_plant)

            initial_storage = float(self.reservoir_parameters.loc[name, "Initial Storage(m3)"])
            new_reservoir.storage_vector = np.append(new_reservoir.storage_vector, initial_storage)

            variable_names_raw = self.reservoir_parameters.columns[-4:].values.tolist()
            for i, plant in enumerate(new_reservoir.hydropower_plants):
                for variable in variable_names_raw:
                    setattr(
                        plant,
                        variable.replace(" ", "_").lower(),
                        eval(self.reservoir_parameters.loc[name, variable])[i],
                    )
            self.reservoirs[name] = new_reservoir
        del self.reservoir_parameters

        self.overarching_policy = Policy()
        for policy in self.policies:
            self.overarching_policy.add_policy_function(**policy)
        del self.policies

    def __call__(self, *args, **kwargs):
        lever_count = self.overarching_policy.get_total_parameter_count()
        input_parameters = [kwargs["v" + str(i)] for i in range(lever_count)]
        (
            egypt_irr,
            egypt_low_had,
            sudan_irr,
            ethiopia_hydro,
        ) = self.evaluate(np.array(input_parameters))
        return egypt_irr, egypt_low_had, sudan_irr, ethiopia_hydro

    def evaluate(self, parameter_vector):
        self.reset_parameters()
        self.overarching_policy.assign_free_parameters(parameter_vector)
        self.simulate()

        bcm_def_egypt = [
            month * 3600 * 24 * self.nu_of_days_per_month[i % 12] * 1e-9
            for i, month in enumerate(self.irr_districts["Egypt"].deficit)
        ]

        egypt_agg_def = np.sum(bcm_def_egypt) / 20
        egypt_freq_low_HAD = np.sum(self.reservoirs["HAD"].level_vector < 159) / len(
            self.reservoirs["HAD"].level_vector
        )

        sudan_irr_districts = [
            value for key, value in self.irr_districts.items() if key not in {"Egypt"}
        ]
        sudan_agg_def_vector = np.repeat(0.0, self.simulation_horizon)
        for district in sudan_irr_districts:
            sudan_agg_def_vector += district.deficit
        bcm_def_sudan = [
            month * 3600 * 24 * self.nu_of_days_per_month[i % 12] * 1e-9
            for i, month in enumerate(sudan_agg_def_vector)
        ]
        sudan_agg_def = np.sum(bcm_def_sudan) / 20

        ethiopia_agg_hydro = (
            np.sum(self.reservoirs["GERD"].actual_hydropower_production)
        ) / (20 * 1e6)

        return (
            egypt_agg_def,
            egypt_freq_low_HAD,
            sudan_agg_def,
            ethiopia_agg_hydro,
        )

    def simulate(self):
        total_monthly_inflow = self.inflowTOT00
        Taminiat_leftover = [0.0, 0.0]

        for t in np.arange(self.simulation_horizon):
            moy = (self.init_month + t - 1) % 12 + 1
            nu_of_days = self.nu_of_days_per_month[moy - 1]

            storages = [reservoir.storage_vector[t] for reservoir in self.reservoirs.values()]
            input = storages + [moy, total_monthly_inflow]
            uu = self.overarching_policy.functions["release"].get_output_norm(np.array(input))
            decision_dict = {reservoir.name: uu[index] for index, reservoir in enumerate(self.reservoirs.values())}

            self.reservoirs["GERD"].integration(
                nu_of_days,
                decision_dict["GERD"],
                self.catchments["BlueNile"].inflow[t],
                moy,
                self.integration_interval,
            )

            self.reservoirs["Roseires"].integration(
                nu_of_days,
                decision_dict["Roseires"],
                self.catchments["GERDToRoseires"].inflow[t] + self.reservoirs["GERD"].release_vector[-1],
                moy,
                self.integration_interval,
            )

            USSennar_input = self.reservoirs["Roseires"].release_vector[-1] + self.catchments["RoseiresToAbuNaama"].inflow[t]
            self.irr_districts["USSennar"].received_flow_raw = np.append(self.irr_districts["USSennar"].received_flow_raw, USSennar_input)
            self.irr_districts["USSennar"].received_flow = np.append(self.irr_districts["USSennar"].received_flow, min(USSennar_input, self.irr_districts["USSennar"].demand[t]))
            USSennar_leftover = max(0, USSennar_input - self.irr_districts["USSennar"].received_flow[-1])

            self.reservoirs["Sennar"].integration(
                nu_of_days,
                decision_dict["Sennar"],
                USSennar_leftover + self.catchments["SukiToSennar"].inflow[t],
                moy,
                self.integration_interval,
            )

            Gezira_input = self.reservoirs["Sennar"].release_vector[-1]
            self.irr_districts["Gezira"].received_flow_raw = np.append(self.irr_districts["Gezira"].received_flow_raw, Gezira_input)
            self.irr_districts["Gezira"].received_flow = np.append(self.irr_districts["Gezira"].received_flow, min(self.irr_districts["Gezira"].demand[t], Gezira_input))
            Gezira_leftover = max(0, Gezira_input - self.irr_districts["Gezira"].received_flow[-1])

            DSSennar_input = Gezira_leftover + self.catchments["Dinder"].inflow[t] + self.catchments["Rahad"].inflow[t]
            self.irr_districts["DSSennar"].received_flow_raw = np.append(self.irr_districts["DSSennar"].received_flow_raw, DSSennar_input)
            self.irr_districts["DSSennar"].received_flow = np.append(self.irr_districts["DSSennar"].received_flow, min(DSSennar_input, self.irr_districts["USSennar"].demand[t]))
            DSSennar_leftover = max(0, DSSennar_input - self.irr_districts["DSSennar"].received_flow[-1])

            Taminiat_input = DSSennar_leftover + self.catchments["WhiteNile"].inflow[t]
            self.irr_districts["Taminiat"].received_flow_raw = np.append(self.irr_districts["Taminiat"].received_flow_raw, Taminiat_input)
            self.irr_districts["Taminiat"].received_flow = np.append(self.irr_districts["Taminiat"].received_flow, min(Taminiat_input, self.irr_districts["Taminiat"].demand[t]))
            Taminiat_leftover.append(max(0, Taminiat_input - self.irr_districts["Taminiat"].received_flow[-1]))
            del Taminiat_leftover[0]

            if t == 0:
                Hassanab_input = 934.2
            else:
                Hassanab_input = Taminiat_leftover[0] + self.catchments["Atbara"].inflow[t - 1]
            self.irr_districts["Hassanab"].received_flow_raw = np.append(self.irr_districts["Hassanab"].received_flow_raw, Hassanab_input)
            self.irr_districts["Hassanab"].received_flow = np.append(self.irr_districts["Hassanab"].received_flow, min(Hassanab_input, self.irr_districts["Hassanab"].demand[t]))
            Hassanab_leftover = max(0, Hassanab_input - self.irr_districts["Hassanab"].received_flow[-1])

            self.reservoirs["HAD"].integration(
                nu_of_days,
                decision_dict["HAD"],
                Hassanab_leftover,
                moy,
                self.integration_interval,
            )
            self.irr_districts["Egypt"].received_flow_raw = np.append(self.irr_districts["Egypt"].received_flow_raw, self.reservoirs["HAD"].release_vector[-1])
            self.irr_districts["Egypt"].received_flow = np.append(self.irr_districts["Egypt"].received_flow, min(self.reservoirs["HAD"].release_vector[-1], self.irr_districts["Egypt"].demand[t]))
            total_monthly_inflow = sum([x.inflow[t] for x in self.catchments.values()])

            for district in self.irr_districts.values():
                district.deficit = np.append(district.deficit, self.deficit_from_target(district.received_flow[-1], district.demand[t]))

            for reservoir in self.reservoirs.values():
                hydropower_production = 0
                for plant in reservoir.hydropower_plants:
                    production = plant.calculate_hydropower_production(reservoir.release_vector[-1], reservoir.level_vector[-1], nu_of_days)
                    hydropower_production += production
                reservoir.actual_hydropower_production = np.append(reservoir.actual_hydropower_production, hydropower_production)

            if t == (self.GERD_filling_time * 12):
                self.reservoirs["GERD"].filling_schedule = None

    @staticmethod
    def deficit_from_target(realisation, target):
        return max(0, target - realisation)

    @staticmethod
    def squared_deficit_from_target(realisation, target):
        return pow(max(0, target - realisation), 2)

    @staticmethod
    def squared_deficit_normalised(sq_deficit, target):
        if target == 0:
            return 0
        else:
            return sq_deficit / pow(target, 2)

    def set_GERD_filling_schedule(self, duration):
        target_storage = 50e9
        difference = target_storage - self.reservoirs["GERD"].storage_vector[0]
        secondly_diff = difference / (duration * 365 * 24 * 3600)
        weights = self.catchments["BlueNile"].inflow[:12]
        self.reservoirs["GERD"].filling_schedule = (weights * 12 * secondly_diff) / weights.sum()

    def reset_parameters(self):
        for reservoir in self.reservoirs.values():
            reservoir.storage_vector = reservoir.storage_vector[:1]
            attributes = ["level_vector", "release_vector", "level_vector", "actual_hydropower_production", "hydropower_deficit"]
            for var in attributes:
                setattr(reservoir, var, np.empty(0))
        for irr_district in self.irr_districts.values():
            attributes = ["received_flow", "deficit"]
            for var in attributes:
                setattr(irr_district, var, np.empty(0))

    def read_settings_file(self, filepath):
        model_parameters = pd.read_excel(filepath, sheet_name="ModelParameters")
        for _, row in model_parameters.iterrows():
            name = row["in Python"]
            value = eval(str(row["Value"])) if row["Data Type"] != "str" else row["Value"]
            if row["Data Type"] == "np.array":
                value = np.array(value)
            setattr(self, name, value)

        self.reservoir_parameters = pd.read_excel(filepath, sheet_name="Reservoirs")
        self.reservoir_parameters.set_index("Reservoir Name", inplace=True)

        self.policies = []
        full_df = pd.read_excel(filepath, sheet_name="PolicyParameters")
        splitpoints = list(full_df.loc[full_df["Parameter Name"] == "Name"].index)
        for i in range(len(splitpoints)):
            try:
                one_policy = full_df.iloc[splitpoints[i] : splitpoints[i + 1], :]
            except IndexError:
                one_policy = full_df.iloc[splitpoints[i] :, :]
            input_dict = {}
            for _, row in one_policy.iterrows():
                key = row["in Python"]
                value = eval(str(row["Value"])) if row["Data Type"] != "str" else row["Value"]
                if row["Data Type"] == "np.array":
                    value = np.array(value)
                input_dict[key] = value
            self.policies.append(input_dict)
