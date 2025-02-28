# E-commerce Recommendation System

An advanced product recommendation engine for e-commerce platforms that combines multiple recommendation strategies to provide highly personalized product suggestions.

## Features

- **Hybrid Recommendation System**: Combines three powerful recommendation approaches:
  - Content-Based Filtering
  - Collaborative Filtering
  - Neural Collaborative Filtering (Deep Learning)

- **Ensemble Methodology**: Weighted combination of multiple recommendation techniques to maximize relevance

- **User Activity Tracking**: Captures views, searches, and purchases with temporal weighting

- **User Profiles**: Authentication system with personalized recommendations based on activity history

- **Admin Dashboard**: Monitor system performance and adjust recommendation weights

- **Interactive UI**: Modern web interface with responsive design

- **Recommendation Visualization**: Clear explanation of how the recommendation system works

- **Feedback System**: Collect and analyze user ratings on recommendations



### Core Components

1. **EnhancedRecommendationSystem**: Main orchestrator that combines all recommendation approaches
2. **RecommendationModels**: Traditional content-based and collaborative filtering approaches
3. **NeuralCollaborativeFiltering**: Deep learning model for advanced pattern recognition
4. **User Activity Tracking**: Records user interactions for personalization
5. **Admin Analytics**: Performance monitoring and system adjustment

## Technical Implementation

- **Backend**: Flask (Python)
- **Database**: SQLite with SQLAlchemy ORM
- **Machine Learning**: TF-IDF, SVD, Neural Networks
- **Deep Learning**: TensorFlow for Neural Collaborative Filtering
- **Frontend**: HTML, CSS, JavaScript, Bootstrap


## Installation

1. Clone the repository
```bash
cd ecommerce-recommendation-system
```

2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Initialize the database
```bash
flask db init
flask db migrate
flask db upgrade
```

5. Run the application
```bash
flask run
```

## Usage

1. **User Navigation**: Users can browse products and receive personalized recommendations
2. **Search**: Enter product names to get content-based recommendations
3. **User Account**: Register/login to get personalized recommendations based on past activity
4. **Feedback**: Rate recommendations to help improve the system
5. **Admin Access**: Access the admin dashboard by logging in with admin credentials (user_id=1)

## How It Works

### Content-Based Filtering
Uses TF-IDF vectorization and SVD to find products similar to ones a user has interacted with, based on product features like name, brand, and tags.

### Collaborative Filtering
Recommends products based on what similar users have liked, using patterns in user behavior with temporal decay to prioritize recent interactions.

### Neural Collaborative Filtering
Uses deep learning with embedding layers to model complex user-item interactions and discover non-linear patterns that traditional methods might miss.

### Ensemble Method
Combines recommendations from all three approaches using weighted scoring to provide the most relevant recommendations.

## Future Improvements

- Contextual recommendations based on time, season, and user demographics
- A/B testing framework for systematic evaluation of different approaches
- Exploration vs. exploitation balancing using multi-armed bandit techniques
- Recommendation diversity controls for varied user experiences
- Automated weight adjustment based on user feedback

## License

[MIT License](LICENSE)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

Aayush Sharma
