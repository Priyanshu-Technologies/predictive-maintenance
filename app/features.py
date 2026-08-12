from __future__ import annotations


def engineer_features(
    machine_type: str,
    air_temperature: float,
    process_temperature: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
) -> dict:
    """
    Recreate every column the model was trained on from the six raw inputs.

    machine_type is the product quality variant: 'L' (low), 'M' (medium),
    or 'H' (high). It was one-hot encoded with drop_first=True in training,
    so 'H' is the implicit baseline (Type_L = Type_M = 0).
    """
    # Guard against divide-by-zero on degenerate inputs (real sensor values
    # never hit these, but an API should never crash on bad input).
    safe_air_temp = air_temperature if air_temperature != 0 else 1e-6
    safe_process_temp = process_temperature if process_temperature != 0 else 1e-6
    safe_rot_speed = rotational_speed if rotational_speed != 0 else 1e-6

    features = {
        "Air temperature": air_temperature,
        "Process temperature": process_temperature,
        "Rotational speed": rotational_speed,
        "Torque": torque,
        "Tool wear": tool_wear,
        # 1. Temperature difference (Process - Air)
        "temp_diff": process_temperature - air_temperature,
        # 2. Power (Torque * Rotational speed)
        "power": torque * rotational_speed,
        # 3. Tool wear ratio (wear relative to dataset max of 255 min)
        "tool_wear-ratio": tool_wear / 255,
        # 4. Efficiency (rotational speed / Process temperature)
        "efficiency": rotational_speed / safe_process_temp,
        # 5. Thermal stress
        "thermal_stress": (process_temperature - air_temperature) / safe_air_temp,
        # 6. Torque to speed ratio
        "torque_speed_ratio": torque / safe_rot_speed,
        # 7. Temperature ratio
        "temp_ratio": process_temperature / safe_air_temp,
        # One-hot encoding of Type (drop_first=True dropped 'H' at train time)
        "Type_L": 1 if machine_type == "L" else 0,
        "Type_M": 1 if machine_type == "M" else 0,
    }
    return features
