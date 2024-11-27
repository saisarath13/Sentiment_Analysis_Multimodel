import pytest
from app import create_app  # Assuming your Flask app is initialized here

@pytest.fixture
def client():
    app = create_app()
    with app.test_client() as client:
        yield client

def test_home(client):
    """
    Test the home page loads correctly.
    """
    response = client.get('/')
    assert response.status_code == 200
    assert b"Welcome to the Sentiment Analysis App" in response.data
def test_predict(client):
    """
    Test the /predict route that handles sentiment prediction.
    """
    # Simulate a POST request with text input
    response = client.post('/predict', data={'text': 'I love this!'})
    
    # Check if the response contains the expected results
    assert response.status_code == 200
    assert b"Prediction" in response.data  # You can assert the result content here
    assert b"positive" in response.data  # Adjust based on the output format

def test_empty_text(client):
    """
    Test for empty input in the prediction.
    """
    response = client.post('/predict', data={'text': ''})
    assert response.status_code == 400  # Assuming you return a 400 for bad requests
    assert b"Please provide some text" in response.data
def test_invalid_input(client):
    """
    Test handling of invalid input (e.g., unsupported characters).
    """
    response = client.post('/predict', data={'text': '#$%&**@@!!'})
    assert response.status_code == 400
    assert b"Invalid input" in response.data
