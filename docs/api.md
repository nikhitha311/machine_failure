# API Documentation

## Base URL
https://machine-failure-api-5ev9.onrender.com

## POST /predict

Send sensor reading, get failure prediction.

**Request:**
```json
{
  "type": "M",
  "air_temperature": 298.1,
  "process_temperature": 308.6,
  "rotational_speed": 1551,
  "torque": 42.8,
  "tool_wear": 0
}
```

**Response:**
```json
{
  "failure_probability": 0.08,
  "prediction": 0,
  "risk_level": "NORMAL",
  "explanation": ["Torque influence (value=42.80, importance=0.325)"],
  "recommendation": "NORMAL: Continue operation."
}
```

## GET /health
Returns API status.

## GET /docs
Interactive Swagger UI.

## Risk Levels

- **NORMAL**: < 30%
- **WARNING**: 30-70%
- **CRITICAL**: > 70%