"""
Pydantic models for request validation and response shaping.
"""
from pydantic import BaseModel, Field
from typing import Literal, List


class MachineReading(BaseModel):
    """A single snapshot of sensor readings from a machine."""

    type: Literal["L", "M", "H"] = Field(
        ..., description="Product quality variant: Low, Medium, or High."
    )
    air_temperature: float = Field(
        ..., description="Air temperature in Kelvin.", examples=[300.5]
    )
    process_temperature: float = Field(
        ..., description="Process temperature in Kelvin.", examples=[310.2]
    )
    rotational_speed: float = Field(
        ..., gt=0, description="Rotational speed in rpm.", examples=[1500]
    )
    torque: float = Field(..., description="Torque in Nm.", examples=[40.5])
    tool_wear: float = Field(
        ..., ge=0, description="Tool wear in minutes.", examples=[110]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "M",
                    "air_temperature": 300.5,
                    "process_temperature": 310.2,
                    "rotational_speed": 1500,
                    "torque": 40.5,
                    "tool_wear": 110,
                }
            ]
        }
    }


class BatchMachineReadings(BaseModel):
    """Multiple machine readings scored in a single request."""

    readings: List[MachineReading] = Field(..., min_length=1, max_length=500)


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = Normal, 1 = Predicted failure.")
    label: str = Field(..., description="Human-readable prediction label.")
    failure_probability: float = Field(
        ..., description="Model's estimated probability of failure (0-1)."
    )
    risk_level: str = Field(
        ..., description="Banded risk level: Low, Medium, High, or Critical."
    )


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]


class ModelInfoResponse(BaseModel):
    model_name: str
    features_used: List[str]
    metrics: dict
    all_models_compared: dict
