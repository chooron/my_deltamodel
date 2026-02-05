"""
Simple test script to verify BlendHydroV3 can be instantiated
"""
import torch
from project.hydro_selection.models.blend_hydro_v3 import BlendHydroV3

# Test configuration
config = {
    "warm_up": 365,
    "warm_up_states": True,
    "variables": ["prcp", "tmean", "pet"],
    "nmul": 1,
    "num_attributes": 10,
    "selected_models": ["HBV", "SHM", "EXPHYDRO", "HYMOD"]
}

# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BlendHydroV3(config=config, device=device)

print(f"✓ Model created successfully: {model.name}")
print(f"✓ Number of models: {len(model.model_order)}")
print(f"✓ Model order: {model.model_order}")
print(f"✓ Learnable parameters: {model.learnable_param_count}")
print(f"✓ Device: {device}")

# Test with dummy data
batch_size = 2
time_steps = 100
n_vars = 3

x_dict = {
    "x_phy": torch.randn(time_steps, batch_size, n_vars).to(device),
    "x_nn_norm": torch.randn(time_steps, batch_size, n_vars).to(device),
    "c_nn_norm": torch.randn(batch_size, config["num_attributes"]).to(device)
}

# Create dummy parameters
parameters = {}
for model_name in model.model_order:
    n_params = len(model.phy_param_names_by_model[model_name])
    parameters[model_name] = torch.randn(batch_size, n_params * config["nmul"]).to(device)

# Add routing parameters (2 params per model)
parameters["GAMMA_UH"] = torch.randn(batch_size, len(model.model_order) * 2).to(device)

print("\n✓ Testing forward pass...")
try:
    output = model(x_dict, parameters)
    print(f"✓ Forward pass successful!")
    print(f"✓ Output keys: {list(output.keys())}")
    print(f"✓ Streamflow shape: {output['streamflow'].shape}")
    print(f"\n✓ All tests passed! BlendHydroV3 is working correctly.")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
