"""
Quick smoke test — hits the running API and checks the responses make sense.
Run the server first (uvicorn app.main:app --reload), then:

    python test_api.py
"""
import requests

BASE = "http://127.0.0.1:8000"

def test_health():
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True
    print("health OK:", r.json())

def test_predict_low_risk():
    payload = {
        "type": "M", "air_temperature": 300.5, "process_temperature": 310.2,
        "rotational_speed": 1500, "torque": 40.5, "tool_wear": 110,
    }
    r = requests.post(f"{BASE}/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    print("low-risk case:", body)
    assert body["risk_level"] in ("Low", "Medium")

def test_predict_high_risk():
    payload = {
        "type": "L", "air_temperature": 303.5, "process_temperature": 312.9,
        "rotational_speed": 1350, "torque": 68.0, "tool_wear": 230,
    }
    r = requests.post(f"{BASE}/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    print("high-risk case:", body)
    assert body["prediction"] == 1

def test_validation_error():
    payload = {
        "type": "Z", "air_temperature": 300.5, "process_temperature": 310.2,
        "rotational_speed": 1500, "torque": 40.5, "tool_wear": 110,
    }
    r = requests.post(f"{BASE}/predict", json=payload)
    assert r.status_code == 422
    print("validation correctly rejected bad type")

if __name__ == "__main__":
    test_health()
    test_predict_low_risk()
    test_predict_high_risk()
    test_validation_error()
    print("\nAll smoke tests passed.")
