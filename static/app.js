// House Price Predictor - Frontend JavaScript

// Load model info on page load
document.addEventListener('DOMContentLoaded', async () => {
    await loadModelInfo();
});

// Load model information
async function loadModelInfo() {
    try {
        const response = await fetch('/model/info');
        if (response.ok) {
            const data = await response.json();
            const modelInfoDiv = document.getElementById('model-info');
            modelInfoDiv.innerHTML = `
                Model: ${data.model_name} |
                R² Score: ${(data.train_r2 * 100).toFixed(1)}% |
                Trained: ${data.trained_on}
            `;
        }
    } catch (error) {
        console.error('Failed to load model info:', error);
    }
}

// Handle form submission
document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    // Show loading, hide results and errors
    document.getElementById('loading').classList.remove('d-none');
    document.getElementById('result').classList.add('d-none');
    document.getElementById('error').classList.add('d-none');
    document.getElementById('placeholder').classList.add('d-none');

    // Gather form data
    const formData = {
        OverallQual: parseInt(document.getElementById('overallQual').value),
        GrLivArea: parseInt(document.getElementById('grLivArea').value),
        GarageCars: parseInt(document.getElementById('garageCars').value),
        GarageArea: parseInt(document.getElementById('garageArea').value),
        TotalBsmtSF: parseInt(document.getElementById('totalBsmtSF').value),
        '1stFlrSF': parseInt(document.getElementById('firstFlrSF').value),
        '2ndFlrSF': parseInt(document.getElementById('secondFlrSF').value) || 0,
        FullBath: parseInt(document.getElementById('fullBath').value),
        TotRmsAbvGrd: parseInt(document.getElementById('totRmsAbvGrd').value),
        YearBuilt: parseInt(document.getElementById('yearBuilt').value),
        YearRemodAdd: parseInt(document.getElementById('yearRemodAdd').value),
        LotArea: 8000,
        BsmtFullBath: 0,
        HalfBath: 0,
        BedroomAbvGr: 3,
        KitchenAbvGr: 1,
        Fireplaces: 0,
        GarageYrBlt: parseInt(document.getElementById('yearBuilt').value),
        Neighborhood: 'NAmes',
        MSZoning: 'RL'
    };

    try {
        // Make prediction request
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const result = await response.json();

        // Hide loading
        document.getElementById('loading').classList.add('d-none');

        // Display result
        document.getElementById('result').classList.remove('d-none');
        document.getElementById('predicted-price').textContent = result.predicted_price_formatted;
        document.getElementById('model-name').textContent = result.model_name;

        if (result.confidence_interval) {
            document.getElementById('confidence-range').textContent =
                `${result.confidence_interval.lower_formatted} - ${result.confidence_interval.upper_formatted}`;
        }

        // Scroll to result on mobile
        if (window.innerWidth < 768) {
            document.getElementById('result').scrollIntoView({ behavior: 'smooth' });
        }

    } catch (error) {
        // Hide loading
        document.getElementById('loading').classList.add('d-none');

        // Show error
        document.getElementById('error').classList.remove('d-none');
        document.getElementById('error-message').textContent =
            'Failed to predict price. Please check your inputs and try again.';

        console.error('Prediction error:', error);
    }
});

// Load example house data
function loadExample(type) {
    const examples = {
        starter: {
            overallQual: 5,
            grLivArea: 1100,
            garageCars: 1,
            garageArea: 250,
            totalBsmtSF: 800,
            firstFlrSF: 800,
            secondFlrSF: 300,
            fullBath: 1,
            totRmsAbvGrd: 5,
            yearBuilt: 1975,
            yearRemodAdd: 1990
        },
        average: {
            overallQual: 7,
            grLivArea: 1500,
            garageCars: 2,
            garageArea: 500,
            totalBsmtSF: 1000,
            firstFlrSF: 1000,
            secondFlrSF: 500,
            fullBath: 2,
            totRmsAbvGrd: 7,
            yearBuilt: 2000,
            yearRemodAdd: 2000
        },
        luxury: {
            overallQual: 9,
            grLivArea: 2500,
            garageCars: 3,
            garageArea: 800,
            totalBsmtSF: 1500,
            firstFlrSF: 1500,
            secondFlrSF: 1000,
            fullBath: 3,
            totRmsAbvGrd: 10,
            yearBuilt: 2010,
            yearRemodAdd: 2010
        }
    };

    const example = examples[type];
    if (example) {
        document.getElementById('overallQual').value = example.overallQual;
        document.getElementById('grLivArea').value = example.grLivArea;
        document.getElementById('garageCars').value = example.garageCars;
        document.getElementById('garageArea').value = example.garageArea;
        document.getElementById('totalBsmtSF').value = example.totalBsmtSF;
        document.getElementById('firstFlrSF').value = example.firstFlrSF;
        document.getElementById('secondFlrSF').value = example.secondFlrSF;
        document.getElementById('fullBath').value = example.fullBath;
        document.getElementById('totRmsAbvGrd').value = example.totRmsAbvGrd;
        document.getElementById('yearBuilt').value = example.yearBuilt;
        document.getElementById('yearRemodAdd').value = example.yearRemodAdd;

        // Scroll to form on mobile
        if (window.innerWidth < 768) {
            document.getElementById('prediction-form').scrollIntoView({ behavior: 'smooth' });
        }
    }
}
