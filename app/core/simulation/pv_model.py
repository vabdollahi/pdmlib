"""
Main PV simulation model that integrates PVLib models with weather data.

This module provides the PVModel class that combines PVLib configurations
with weather data to run solar power simulations.
"""

from typing import Dict

import pandas as pd
from pvlib import modelchain
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.simulation.pvlib_models import PVLibModel, PVLibResultsColumns
from app.core.simulation.weather_provider import WeatherProviderProtocol
from app.core.utils.logging import get_logger

logger = get_logger("pv_model")


class PVModel(BaseModel):
    """
    PV power plant physical model based on PVLib setup and weather data.

    It computes the solar power output of the power plant using weather data and
    solar plant configurations such as coordinates, PV module angles, and either
    AC capacity and DC/AC ratio, or arrays and inverters specifications.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pv_config: PVLibModel = Field(description="PVLib model configuration")
    weather_provider: WeatherProviderProtocol = Field(
        description="Weather data provider configuration"
    )

    @model_validator(mode="after")
    def validate_pv_config(self) -> "PVModel":
        """Validate that the PV configuration has at least one system with arrays."""
        if not self.pv_config.pv_systems:
            raise ValueError("At least one PV system must be configured")

        for i, pv_system in enumerate(self.pv_config.pv_systems):
            if not pv_system.pv_arrays:
                raise ValueError(f"PV system {i} must have at least one array")

        return self

    async def run_simulation(self) -> pd.DataFrame:
        """
        Run the PVLib model with the provided weather data.

        Returns:
            pd.DataFrame: Results of the PVLib model simulation.
        """
        logger.info("Starting PV simulation")

        # Create PVLib model chain
        model_chain = self.pv_config.create()

        # Get weather data for the location
        weather_data = await self.weather_provider.get_data()

        logger.info(f"Running simulation for {len(weather_data)} weather data points")

        # Run the PVLib model
        model_chain.run_model(weather_data)

        # Check if we have results
        if not hasattr(model_chain, "results") or model_chain.results is None:
            raise RuntimeError("PVLib simulation failed - no results generated")

        # Validate AC results exist
        if not hasattr(model_chain.results, "ac") or model_chain.results.ac is None:
            raise RuntimeError("PVLib simulation failed - no AC results generated")

        # Process results based on DC model type
        try:
            dc_model_name = self._get_dc_model_name(model_chain)

            if "pvwatts" in dc_model_name.lower():
                results = self._process_pv_watts_results(model_chain)
            elif "cec" in dc_model_name.lower():
                results = self._process_cec_results(model_chain)
            else:
                logger.warning(
                    f"Unknown DC model: {dc_model_name}, using default processing"
                )
                results = self._process_default_results(model_chain)

            logger.info(f"Simulation completed with {len(results)} result points")
            return results
        except Exception as e:
            logger.error(f"Error processing simulation results: {e}")
            return self._create_minimal_results(model_chain)

    def _get_dc_model_name(self, model_chain: modelchain.ModelChain) -> str:
        """Get the DC model name safely."""
        try:
            dc_model = getattr(model_chain, "dc_model", None)
            if dc_model is not None and hasattr(dc_model, "__name__"):
                return str(dc_model.__name__)
            elif dc_model is not None:
                return str(dc_model)
            else:
                return "unknown"
        except Exception:
            return "unknown"

    def _create_minimal_results(
        self, model_chain: modelchain.ModelChain
    ) -> pd.DataFrame:
        """Create minimal results when processing fails."""
        try:
            ac_result = getattr(model_chain.results, "ac", None)
            if (
                ac_result is not None
                and hasattr(ac_result, "index")
                and ac_result.index is not None
            ):
                index = ac_result.index
                return pd.DataFrame(
                    {
                        PVLibResultsColumns.DATE_TIME.value: index,
                        PVLibResultsColumns.AC.value: [0.0] * len(index),
                    }
                )
            else:
                # Return empty DataFrame with correct column structure
                return pd.DataFrame(
                    {
                        PVLibResultsColumns.DATE_TIME.value: [],
                        PVLibResultsColumns.AC.value: [],
                    }
                )
        except Exception:
            return pd.DataFrame(
                {
                    PVLibResultsColumns.DATE_TIME.value: [],
                    PVLibResultsColumns.AC.value: [],
                }
            )

    @staticmethod
    def get_dc_column_name(array_name: str) -> str:
        """Generate the column name for DC power of the given array."""
        return f"{array_name} {PVLibResultsColumns.DC_ARRAY.value}"

    def _process_pv_watts_results(
        self, model_chain: modelchain.ModelChain
    ) -> pd.DataFrame:
        """
        Process PV watts DC model results.

        Args:
            model_chain: PVLib model chain.

        Returns:
            pd.DataFrame: DataFrame containing the processed results.
        """
        logger.debug("Processing PV watts model results")

        # Validate AC results with explicit null checks
        ac_result = getattr(model_chain.results, "ac", None)
        if (
            ac_result is None
            or not hasattr(ac_result, "index")
            or not hasattr(ac_result, "values")
        ):
            raise ValueError("Invalid AC results structure")

        # Store inverters' AC power
        results = pd.DataFrame(
            {
                PVLibResultsColumns.DATE_TIME.value: ac_result.index,
                PVLibResultsColumns.AC.value: ac_result.values,
            }
        )

        # Store arrays' DC power if available
        dc_results = getattr(model_chain.results, "dc", None)
        if dc_results is not None:
            self._add_dc_results_pv_watts(results, model_chain)

        return results

    def _add_dc_results_pv_watts(
        self, results: pd.DataFrame, model_chain: modelchain.ModelChain
    ) -> None:
        """Add DC results for PV Watts model."""
        dc_columns = []
        try:
            dc_results = getattr(model_chain.results, "dc", None)
            if dc_results is not None:
                for array, pdc in zip(model_chain.system.arrays, dc_results):
                    if hasattr(pdc, "values"):
                        dc_col_name = self.get_dc_column_name(array.name)
                        results[dc_col_name] = pdc.values
                        dc_columns.append(dc_col_name)

                # Store total DC power
                if dc_columns:
                    total_dc = results[dc_columns].sum(axis=1)
                    results[PVLibResultsColumns.DC.value] = total_dc
        except Exception as e:
            logger.warning(f"Could not process DC results for PV Watts: {e}")

    def _process_cec_results(self, model_chain: modelchain.ModelChain) -> pd.DataFrame:
        """
        Process CEC DC model results.

        Args:
            model_chain: PVLib model chain.

        Returns:
            pd.DataFrame: DataFrame containing the processed results.
        """
        logger.debug("Processing CEC model results")

        # Validate AC results with explicit null checks
        ac_result = getattr(model_chain.results, "ac", None)
        if ac_result is None or not hasattr(ac_result, "index"):
            raise ValueError("Invalid AC results structure")

        # For CEC model, AC results might have p_mp attribute
        ac_values = None
        if (
            hasattr(ac_result, "p_mp")
            and ac_result.p_mp is not None
            and hasattr(ac_result.p_mp, "values")
        ):
            ac_values = ac_result.p_mp.values
        elif hasattr(ac_result, "values"):
            ac_values = ac_result.values
        else:
            raise ValueError("Cannot extract AC power values")

        # Store inverters' AC power
        results = pd.DataFrame(
            {
                PVLibResultsColumns.DATE_TIME.value: ac_result.index,
                PVLibResultsColumns.AC.value: ac_values,
            }
        )

        # Store arrays' DC power if available
        dc_results = getattr(model_chain.results, "dc", None)
        if dc_results is not None:
            self._add_dc_results_cec(results, model_chain)

        return results

    def _add_dc_results_cec(
        self, results: pd.DataFrame, model_chain: modelchain.ModelChain
    ) -> None:
        """Add DC results for CEC model."""
        dc_columns = []
        try:
            dc_results = getattr(model_chain.results, "dc", None)
            if dc_results is not None:
                for array, pdc in zip(model_chain.system.arrays, dc_results):
                    dc_col_name = self.get_dc_column_name(array.name)
                    # Try different ways to access DC power for CEC model
                    if hasattr(pdc, "__getitem__") and "p_mp" in pdc:
                        results[dc_col_name] = pdc["p_mp"].values
                    elif hasattr(pdc, "values"):
                        results[dc_col_name] = pdc.values
                    else:
                        msg = f"Could not extract DC power for array {array.name}"
                        logger.warning(msg)
                        continue
                    dc_columns.append(dc_col_name)

                # Store total DC power
                if dc_columns:
                    total_dc = results[dc_columns].sum(axis=1)
                    results[PVLibResultsColumns.DC.value] = total_dc
        except Exception as e:
            logger.warning(f"Could not process DC results for CEC: {e}")

    def _process_default_results(
        self, model_chain: modelchain.ModelChain
    ) -> pd.DataFrame:
        """
        Process default model results when model type is unknown.

        Args:
            model_chain: PVLib model chain.

        Returns:
            pd.DataFrame: DataFrame containing the processed results.
        """
        logger.debug("Processing default model results")

        try:
            ac_result = getattr(model_chain.results, "ac", None)
            if ac_result is None or not hasattr(ac_result, "index"):
                raise ValueError("Invalid AC results structure")

            # Try to get AC power values in different ways
            ac_power = None
            if (
                hasattr(ac_result, "p_mp")
                and ac_result.p_mp is not None
                and hasattr(ac_result.p_mp, "values")
            ):
                ac_power = ac_result.p_mp.values
            elif hasattr(ac_result, "values"):
                ac_power = ac_result.values
            else:
                raise ValueError("Cannot extract AC power values")

            results = pd.DataFrame(
                {
                    PVLibResultsColumns.DATE_TIME.value: ac_result.index,
                    PVLibResultsColumns.AC.value: ac_power,
                }
            )

            # Try to get DC power if available
            dc_results = getattr(model_chain.results, "dc", None)
            if dc_results is not None:
                self._add_dc_results_generic(results, model_chain)

            return results

        except Exception as e:
            logger.error(f"Error processing default results: {e}")
            return self._create_minimal_results(model_chain)

    def _add_dc_results_generic(
        self, results: pd.DataFrame, model_chain: modelchain.ModelChain
    ) -> None:
        """Add DC results for generic/unknown model."""
        dc_columns = []
        try:
            dc_results = getattr(model_chain.results, "dc", None)
            if dc_results is not None:
                for array, pdc in zip(model_chain.system.arrays, dc_results):
                    dc_col_name = self.get_dc_column_name(array.name)

                    # Try multiple ways to extract DC values
                    dc_values = None
                    if hasattr(pdc, "__getitem__"):
                        # Try dictionary-like access
                        try:
                            if "p_mp" in pdc and hasattr(pdc["p_mp"], "values"):
                                dc_values = pdc["p_mp"].values
                        except (KeyError, TypeError):
                            pass

                    if dc_values is None and hasattr(pdc, "values"):
                        dc_values = pdc.values

                    if dc_values is not None:
                        results[dc_col_name] = dc_values
                        dc_columns.append(dc_col_name)
                    else:
                        msg = f"Could not extract DC power for array {array.name}"
                        logger.warning(msg)

                # Store total DC power
                if dc_columns:
                    total_dc = results[dc_columns].sum(axis=1)
                    results[PVLibResultsColumns.DC.value] = total_dc
        except Exception as e:
            logger.warning(f"Could not process generic DC results: {e}")

    def get_system_capacity(self) -> Dict[str, float]:
        """
        Get the total system capacity information.

        Returns:
            Dict containing AC and DC capacity information.
        """
        total_ac_capacity = 0
        total_dc_capacity = 0

        for pv_system in self.pv_config.pv_systems:
            if hasattr(pv_system, "max_power_output_ac_w"):
                total_ac_capacity += pv_system.max_power_output_ac_w or 0

            # Calculate DC capacity from arrays
            for array in pv_system.pv_arrays:
                # Use the create method to get module parameters
                try:
                    module_params = array.pv_modules.create()
                    dc_per_module = 0

                    if isinstance(module_params, dict):
                        # Try different parameter names depending on module type
                        # For CEC modules: STC (Standard Test Conditions) rating
                        # For PVWatts modules: pdc0
                        dc_per_module = (
                            module_params.get("pdc0", 0)  # PVWatts style
                            or module_params.get("STC", 0)  # CEC style
                            or module_params.get("PTC", 0)  # CEC PTC rating
                        )
                    else:
                        # If it's a pandas Series, try the same attributes
                        dc_per_module = (
                            getattr(module_params, "pdc0", 0)
                            or getattr(module_params, "STC", 0)
                            or getattr(module_params, "PTC", 0)
                        )
                except Exception:
                    dc_per_module = 0

                array_dc = (
                    dc_per_module
                    * array.pv_modules.count
                    * array.array_setup.number_of_strings
                )
                total_dc_capacity += array_dc

        dc_ac_ratio = (
            total_dc_capacity / total_ac_capacity if total_ac_capacity > 0 else 0
        )

        return {
            "ac_capacity_w": total_ac_capacity,
            "dc_capacity_w": total_dc_capacity,
            "dc_ac_ratio": dc_ac_ratio,
        }
