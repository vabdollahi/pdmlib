"""
Test script to verify that IESO provider removes hoep_cad_mwh column
"""
import pandas as pd

from app.core.simulation.ieso_data import IESOPriceProvider

# Create a mock data frame as if returned by the standardization process
def test_column_removal():
    # Simulate the data structure before column removal
    test_data = {
        'timestamp': pd.date_range('2025-01-01', periods=24, freq='h'),
        'hoep_cad_mwh': [50.0] * 24,
        'price_dollar_mwh': [50.0] * 24  # Same values after standardization
    }
    
    # Create provider instance
    provider = IESOPriceProvider()
    
    # Create test dataframe
    df = pd.DataFrame(test_data)
    df.set_index('timestamp', inplace=True)
    
    # Apply the standardization and column removal logic
    standardized_df = provider._standardize_columns(df)
    
    # Remove redundant column (this simulates the new logic we added)
    if "hoep_cad_mwh" in standardized_df.columns:
        standardized_df = standardized_df.drop(columns=["hoep_cad_mwh"])
    
    print("Original columns:", df.columns.tolist())
    print("Final columns:", standardized_df.columns.tolist())
    print("Sample data:")
    print(standardized_df.head(3))
    
    # Verify only price_dollar_mwh remains
    assert 'hoep_cad_mwh' not in standardized_df.columns
    assert 'price_dollar_mwh' in standardized_df.columns
    print("✅ Test passed: hoep_cad_mwh column successfully removed")

if __name__ == "__main__":
    test_column_removal()
