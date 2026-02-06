# House Price Prediction ML Project

An end-to-end machine learning project for predicting house prices using Docker deployment.

## Project Overview

This project demonstrates a complete ML deployment pipeline:
- Data acquisition and exploratory data analysis
- Feature engineering and model training
- FastAPI web service for predictions
- Docker containerization
- Cloud deployment to Render.com

## Technology Stack

- **Python:** 3.14
- **ML Framework:** scikit-learn, XGBoost
- **Web Framework:** FastAPI
- **Frontend:** HTML/CSS/JavaScript with Bootstrap
- **Containerization:** Docker
- **Dataset:** Kaggle House Prices (Ames, Iowa)

## Project Structure

```
house-price-predictor/
├── data/              # Dataset storage (gitignored)
├── notebooks/         # Jupyter notebooks for EDA
├── src/               # Source code
│   ├── data/         # Data preprocessing
│   ├── models/       # Model training
│   ├── api/          # FastAPI application
│   └── utils/        # Utility functions
├── models/           # Saved models (gitignored)
├── static/           # Frontend files
├── tests/            # Unit tests
└── scripts/          # Utility scripts
```

## Setup Instructions

### Prerequisites
- Python 3.11+
- Docker Desktop
- Kaggle account

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd house-price-predictor
```

2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Mac/Linux
```

3. Install dependencies:
```bash
python -m pip install -r requirements.txt
```

4. Configure Kaggle API:
- Download `kaggle.json` from Kaggle Account settings
- Place in `C:\Users\<YourUsername>\.kaggle\` (Windows)

5. Download dataset:
```bash
kaggle competitions download -c house-prices-advanced-regression-techniques
```

## Usage

### Train Model
```bash
python src/models/train.py
```

### Run API Locally
```bash
uvicorn src.api.main:app --reload
```
Visit: http://localhost:8000/docs for Swagger UI

### Docker Deployment

**Option 1: Docker Compose (Recommended)**
```bash
docker-compose up -d
```

**Option 2: Manual Docker Build & Run**
```bash
# Build the image
docker build -t house-price-predictor:latest .

# Run the container
docker run -d -p 8000:8000 --name house-price-app house-price-predictor:latest

# Check logs
docker logs house-price-app

# Stop container
docker stop house-price-app

# Remove container
docker rm house-price-app
```

**Access the Application:**
- Web UI: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Run Tests
```bash
pytest tests/ -v
```

## Development Phases

- [x] Phase 1: Environment Setup
- [x] Phase 2: Exploratory Data Analysis
- [x] Phase 3: Feature Engineering & Model Training
- [x] Phase 4: API Development
- [x] Phase 5: Dockerization
- [ ] Phase 6: Testing & Cloud Deployment

## Learning Goals

- Understand Docker containerization
- Build complete ML deployment pipeline
- Deploy production-ready ML application
- Learn FastAPI web framework
- Practice DevOps best practices

## License

MIT

## Author

Built as a learning project for understanding ML deployment and Docker.
