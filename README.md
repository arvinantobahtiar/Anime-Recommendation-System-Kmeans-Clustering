# Anime Recommendation System with K-means Clustering

This project is an Anime Recommendation System built with Flask, a Python web framework. It utilizes machine learning techniques such as K-means clustering and TF-IDF vectorization to recommend anime titles based on user input.

## Features

- **Anime Recommendation**: Users can input the title of an anime, and the system will recommend similar anime titles based on machine learning models.
- **Fuzzy String Matching**: The system uses fuzzy string matching to find the closest matching anime title to the user input, improving accuracy.
- **Personalized Recommendations**: Recommendations are tailored to the user's genre of anime title they inputed, providing personalized anime suggestions based on type of anime they usually watch.

## Technologies Used

- **Flask**: Python web framework used for backend development.
- **Pandas**: Library for data manipulation and analysis.
- **Scikit-learn**: Library for machine learning tasks such as clustering and vectorization.
- **FuzzyWuzzy**: Library for fuzzy string matching.
- **HTML/CSS**: Frontend for user interface design.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/anime-recommendation-system.git
   ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Flask app:

    ```bash
    python app.py
    ```

Open a web browser and navigate to http://localhost:5500.

## Deployment

This project can be deployed with any platforms possible. Ensure to set up environment variables and configure the deployment settings accordingly. I personally already deployed it with GCP and zeet as a third party tool to deploy this project.
LINK: https://anime-recommendation-system-kmeans-clustering-21w-5qnji3asoq-as.a.run.app/ 

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please fork the repository and create a pull request. Feel free to open an issue to report bugs or suggest new features. Thank you!!