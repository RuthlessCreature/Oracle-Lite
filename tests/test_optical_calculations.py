import math
import pytest
from talker.optical_calculations import OpticalInputs


def test_optical_calculation_units_and_reciprocal_relation():
    values = OpticalInputs(36, 3072, 3.45).calculate()
    assert values['sampling_um_per_px'] == 11.71875
    assert math.isclose(values['sensor_width_mm'], 10.5984)
    assert math.isclose(values['magnification'], 0.2944)
    assert math.isclose(values['magnification'] * values['sampling_um_per_px'], 3.45)
    assert values['accuracy_verified'] is False


@pytest.mark.parametrize('inputs', [(0, 2448, 3.45), (20, 0, 3.45), (20, 2.5, 3.45),
                                   (20, 2448, float('nan')), (float('inf'), 2448, 3.45),
                                   (20, True, 3.45), (True, 2448, 3.45)])
def test_invalid_physical_inputs_cannot_produce_a_plausible_number(inputs):
    with pytest.raises(ValueError):
        OpticalInputs(*inputs).calculate()
