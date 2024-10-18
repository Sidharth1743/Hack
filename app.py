from flask import Flask, render_template, request
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

class ESGScorer:
    def calculate_esg_score(self, company, ticker):
        # Dummy implementation for the ESG scoring
        # Replace with your actual calculation logic
        logging.debug(f'Calculating ESG score for {company} ({ticker})')
        return 75  # Example static score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    company = request.form['company']
    logging.debug(f'Received request for company: {company}')
    
    ticker_mapping = {
        'Apple': 'AAPL',
        'Microsoft': 'MSFT',
        'Google': 'GOOGL',
        'Amazon': 'AMZN',
        'Tesla': 'TSLA'
    }

    try:
        ticker = ticker_mapping[company]
        esg_scorer = ESGScorer()
        esg_score = esg_scorer.calculate_esg_score(company, ticker)
        logging.debug(f'ESG Score: {esg_score}')
        return render_template('results.html', score=esg_score)
    except Exception as e:
        logging.error(f'Error occurred: {str(e)}')
        return "An error occurred while calculating the ESG score."

if __name__ == '__main__':
    app.run(debug=True)
