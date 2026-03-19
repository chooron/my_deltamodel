import torch
from project.hydro_selection.models.layers.jit_core import exphydro_timestep_loop


def test_exphydro_basic():
    T = 10
    B = 2
    E = 3

    P = torch.ones((T, B, E), dtype=torch.float32)
    Tavg = torch.full((T, B, E), 5.0)
    PET = torch.full((T, B, E), 0.5)

    soil_storage = torch.zeros((B, E))
    snow_storage = torch.zeros((B, E))

    f = torch.full((B, E), 0.1)
    ddf = torch.full((B, E), 0.2)
    smax = torch.full((B, E), 50.0)
    qmax = torch.full((B, E), 2.0)
    mint = torch.full((B, E), 0.0)
    maxt = torch.full((B, E), 1.0)

    qsim, et, melt, snow_f, soil_f = exphydro_timestep_loop(
        P, Tavg, PET, soil_storage, snow_storage,
        f, ddf, smax, qmax, mint, maxt, 1e-6
    )

    assert qsim.shape == (T, B, E)
    assert et.shape == (T, B, E)
    assert melt.shape == (T, B, E)

    # basic non-negativity
    assert torch.all(qsim >= -1e-6)
    assert torch.all(et >= -1e-6)
    assert torch.all(snow_f >= 0.0)
    assert torch.all(soil_f >= 0.0)


if __name__ == "__main__":
    test_exphydro_basic()
    print("exphydro basic test passed")
