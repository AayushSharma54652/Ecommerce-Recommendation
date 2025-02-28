from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import random
import functools
from datetime import datetime, timedelta
import os
import tensorflow as tf

# Import the new enhanced recommendation system
from enhanced_recommendation_system import EnhancedRecommendationSystem

app = Flask(__name__)

# load files===========================================================================================================
trending_products = pd.read_csv("models/trending_products.csv")
train_data = pd.read_csv("models/clean_data.csv")

# database configuration---------------------------------------
app.secret_key = "alskdjfwoeieiurlskdjfslkdjf"
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///ecom.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.permanent_session_lifetime = timedelta(days=7)
db = SQLAlchemy(app)


# Define your model classes -----------------------------------
class Signup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

class Signin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

class UserActivity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('signup.id'), nullable=False)
    product_name = db.Column(db.String(255), nullable=False)
    activity_type = db.Column(db.String(50))  # search, view, purchase, etc.
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# New model to track recommendation feedback
class RecommendationFeedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('signup.id'), nullable=False)
    product_name = db.Column(db.String(255), nullable=False)
    recommendation_type = db.Column(db.String(50))  # Content-Based, Collaborative, Neural, etc.
    rating = db.Column(db.Integer)  # User feedback rating (1-5)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    comments = db.Column(db.Text, nullable=True)

# Define global variables for the recommendation models
user_activity_data = None
recommendation_system = None

def get_user_activity_data():
    # Query all user activities from the database and convert to DataFrame
    activities = UserActivity.query.all()
    
    # Convert to DataFrame
    if activities:
        activity_data = []
        for activity in activities:
            activity_data.append({
                'user_id': activity.user_id,
                'product_name': activity.product_name,
                'activity_type': activity.activity_type,
                'timestamp': activity.timestamp
            })
        return pd.DataFrame(activity_data)
    else:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=['user_id', 'product_name', 'activity_type', 'timestamp'])

# Initialize database and models within application context
with app.app_context():
    db.create_all()
    # Initialize the recommendation models
    user_activity_data = get_user_activity_data()
    
    # Initialize the recommendation system with error handling
    try:
        recommendation_system = EnhancedRecommendationSystem(train_data, user_activity_data)
    except Exception as e:
        print(f"Error initializing Enhanced Recommendation System: {e}")
        print("Falling back to traditional RecommendationModels")
        from recommendation_models import RecommendationModels
        recommendation_system = RecommendationModels(train_data, user_activity_data)
    
# Login required decorator
def login_required(view_function):
    @functools.wraps(view_function)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('index'))
        return view_function(*args, **kwargs)
    return decorated_function


# Recommendations functions============================================================================================
# Function to truncate product name
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text


def content_based_recommendations(train_data, item_name, top_n=10):
    """Get content-based recommendations using the enhanced system"""
    global recommendation_system
    return recommendation_system.get_recommendations(None, item_name, top_n)


# Collaborative Filtering Recommendation Function
def collaborative_filtering_recommendations(train_data, target_user_id, top_n=10):
    """Get collaborative filtering recommendations using the enhanced system"""
    global recommendation_system
    return recommendation_system.get_recommendations(target_user_id, None, top_n)

# Enhanced Hybrid Recommendations Function
def enhanced_recommendations(train_data, target_user_id, item_name=None, top_n=10):
    """Get enhanced recommendations using the new system"""
    global recommendation_system
    return recommendation_system.get_recommendations(target_user_id, item_name, top_n=top_n)

# Add a new function to refresh recommendation models with updated user data
def refresh_recommendation_models():
    """Refresh recommendation models with updated user activity data"""
    global recommendation_system, user_activity_data
    user_activity_data = get_user_activity_data()
    recommendation_system.refresh_models(user_activity_data)
    print("Recommendation models refreshed with updated user data")

# routes===============================================================================
# List of predefined image URLs
random_image_urls = [
    "static/img/img_1.png",
    "static/img/img_2.png",
    "static/img/img_3.png",
    "static/img/img_4.png",
    "static/img/img_5.png",
    "static/img/img_6.png",
    "static/img/img_7.png",
    "static/img/img_8.png",
]


@app.route("/")
def index():
    # Initialize random_product_image_urls for both paths
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(8)]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    
    # Check if user is logged in for personalized recommendations
    if 'user_id' in session:
        # Get user's recent activities
        user_activities = UserActivity.query.filter_by(user_id=session['user_id']).order_by(UserActivity.timestamp.desc()).limit(3).all()
        
        if user_activities:
            # Get recommendations based on user's most recent activity
            recent_product = user_activities[0].product_name
            try:
                # Use enhanced recommendations instead of just content-based
                personalized_recommendations = enhanced_recommendations(
                    train_data, 
                    session['user_id'], 
                    item_name=recent_product, 
                    top_n=8
                )
                
                if not personalized_recommendations.empty:
                    # Get the recommendation types for styling
                    recommendation_types = personalized_recommendations['DominantRecommendationType'].tolist() if 'DominantRecommendationType' in personalized_recommendations.columns else None
                    
                    # Remove metadata columns for template rendering
                    if 'RecommendationScore' in personalized_recommendations.columns:
                        display_recommendations = personalized_recommendations.drop(
                            columns=['RecommendationScore']
                        )
                    else:
                        display_recommendations = personalized_recommendations
                    
                    return render_template('index.html', 
                                          trending_products=display_recommendations, 
                                          truncate=truncate,
                                          random_price=random.choice(price),
                                          random_product_image_urls=random_product_image_urls,
                                          user_name=session.get('username', ''),
                                          personalized=True,
                                          recommendation_types=recommendation_types)
            except Exception as e:
                print(f"Error generating personalized recommendations: {e}")
                # Fall back to default trending products if personalization fails
    
    # Default: show trending products
    return render_template('index.html', 
                          trending_products=trending_products.head(8), 
                          truncate=truncate,
                          random_product_image_urls=random_product_image_urls,
                          random_price=random.choice(price),
                          user_name=session.get('username', ''))


@app.route("/main")
def main():
    # Initialize an empty DataFrame for content_based_rec
    # so the template doesn't throw an error when first loaded
    empty_df = pd.DataFrame(columns=['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating', 'DominantRecommendationType'])
    return render_template('main.html', content_based_rec=empty_df, truncate=truncate)


@app.route("/index")
def indexredirect():
    return redirect(url_for('index'))


@app.route("/signup", methods=['POST','GET'])
def signup():
    if request.method=='POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Check if username already exists
        existing_user = Signup.query.filter_by(username=username).first()
        if existing_user:
            return render_template('index.html', signup_message='Username already exists!')

        new_signup = Signup(username=username, email=email, password=password)
        db.session.add(new_signup)
        db.session.commit()
        
        # Set session for the new user
        session['user_id'] = new_signup.id
        session['username'] = username
        session.permanent = True

        return redirect(url_for('index'))


@app.route('/signin', methods=['POST', 'GET'])
def signin():
    if request.method == 'POST':
        username = request.form['signinUsername']
        password = request.form['signinPassword']
        
        # Verify user credentials
        user = Signup.query.filter_by(username=username, password=password).first()
        if user:
            # Set user session
            session['user_id'] = user.id
            session['username'] = username
            session.permanent = True
            
            # Record login
            new_signin = Signin(username=username, password=password)
            db.session.add(new_signin)
            db.session.commit()
            
            # Redirect to personalized homepage
            return redirect(url_for('index'))
        else:
            # If invalid credentials
            return render_template('index.html', signup_message='Invalid username or password!')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('index'))


@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    if request.method == 'POST':
        prod = request.form.get('prod')
        nbr = int(request.form.get('nbr'))
        
        # Record the search activity if user is logged in
        if 'user_id' in session:
            new_activity = UserActivity(
                user_id=session['user_id'],
                product_name=prod,
                activity_type='search'
            )
            db.session.add(new_activity)
            db.session.commit()
            
            # Use enhanced recommendations for logged in users
            # Changed from hybrid_recommendations to enhanced_recommendations
            recommended_products = enhanced_recommendations(
                train_data, 
                session['user_id'], 
                item_name=prod, 
                top_n=nbr
            )
            
            if not recommended_products.empty:
                # Get the recommendation types for styling
                recommendation_types = recommended_products['DominantRecommendationType'].tolist() if 'DominantRecommendationType' in recommended_products.columns else None
                
                # Remove metadata columns for template rendering if they exist
                if 'RecommendationScore' in recommended_products.columns:
                    display_recommendations = recommended_products.drop(
                        columns=['RecommendationScore']
                    )
                else:
                    display_recommendations = recommended_products
                
                price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
                return render_template('main.html', 
                                      content_based_rec=display_recommendations, 
                                      truncate=truncate,
                                      random_price=random.choice(price),
                                      personalized=True,
                                      recommendation_types=recommendation_types)
        else:
            # Use content-based recommendations for non-logged in users
            content_based_rec = content_based_recommendations(train_data, prod, top_n=nbr)

            if not content_based_rec.empty:
                price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
                return render_template('main.html', 
                                      content_based_rec=content_based_rec, 
                                      truncate=truncate,
                                      random_price=random.choice(price))
        
        # If we get here, then no recommendations were found or an error occurred
        message = "No recommendations available for this product."
        return render_template('main.html', message=message, content_based_rec=pd.DataFrame(), truncate=truncate)
    
    # Handle GET requests (if any)
    return render_template('main.html', content_based_rec=pd.DataFrame(), truncate=truncate)


@app.route("/view_product/<path:product_name>")
def view_product(product_name):
    # Record the view if user is logged in
    if 'user_id' in session:
        new_activity = UserActivity(
            user_id=session['user_id'],
            product_name=product_name,
            activity_type='view'
        )
        db.session.add(new_activity)
        db.session.commit()
        
        # Consider refreshing models after accumulating several new activities
        # to avoid unnecessary computation
        user_activity_count = UserActivity.query.filter_by(user_id=session['user_id']).count()
        if user_activity_count % 5 == 0:  # Refresh every 5 activities
            refresh_recommendation_models()
    
    # Find the product in the dataset
    product = None
    try:
        product = train_data[train_data['Name'] == product_name].iloc[0].to_dict()
    except:
        return "Product not found", 404
    
    price = random.choice([40, 50, 60, 70, 100, 122, 106, 50, 30, 50])
    return render_template('product_detail.html', product=product, price=price)


@app.route('/profile')
@login_required
def profile():
    user = Signup.query.get(session['user_id'])
    
    if not user:
        return redirect(url_for('logout'))
    
    # Get user's activity history
    activities = UserActivity.query.filter_by(user_id=user.id).order_by(UserActivity.timestamp.desc()).limit(10).all()
    
    # Get personalized recommendations based on most frequent activity
    recommendations = pd.DataFrame()
    similar_users = []
    
    if activities:
        # Use the most recent activity for recommendations
        recent_product = activities[0].product_name
        try:
            # Changed from hybrid_recommendations to enhanced_recommendations
            # Try to use get_advanced_recommendations first
            try:
                results = recommendation_system.get_advanced_recommendations(
                    user.id,
                    item_name=recent_product,
                    top_n=5,
                    include_similar_users=True
                )
                
                recommendations = results['recommendations']
                similar_users = results.get('similar_users', [])
            except (AttributeError, Exception) as e:
                # Fall back to basic enhanced recommendations if advanced isn't available
                print(f"Falling back to basic recommendations: {e}")
                recommendations = enhanced_recommendations(
                    train_data, 
                    session['user_id'], 
                    item_name=recent_product, 
                    top_n=5
                )
        except Exception as e:
            print(f"Error getting recommendations: {e}")
    
    # Pass the truncate function to the template
    return render_template('profile.html', 
                          user=user, 
                          activities=activities, 
                          recommendations=recommendations, 
                          truncate=truncate,
                          personalized=True,
                          similar_users=similar_users)

# Add route for recommendation feedback
@app.route('/recommendation_feedback', methods=['POST'])
@login_required
def recommendation_feedback():
    """Collect user feedback on recommendations"""
    product_name = request.form.get('product_name')
    recommendation_type = request.form.get('recommendation_type')
    rating = request.form.get('rating')
    comments = request.form.get('comments', '')
    
    if not all([product_name, recommendation_type, rating]):
        flash('Missing required information for feedback', 'warning')
        return redirect(request.referrer or url_for('index'))
    
    try:
        rating = int(rating)
        if rating < 1 or rating > 5:
            raise ValueError("Rating must be between 1 and 5")
    except ValueError:
        flash('Rating must be a number between 1 and 5', 'warning')
        return redirect(request.referrer or url_for('index'))
    
    # Save the feedback
    feedback = RecommendationFeedback(
        user_id=session['user_id'],
        product_name=product_name,
        recommendation_type=recommendation_type,
        rating=rating,
        comments=comments
    )
    db.session.add(feedback)
    db.session.commit()
    
    flash('Thank you for your feedback!', 'success')
    return redirect(request.referrer or url_for('index'))

# Add route for recommendation visualization
@app.route('/recommendation_visualization')
def recommendation_visualization():
    """Show visualization of how the recommendation system works"""
    # Get the current weights for the visualization
    weights = {}
    
    if hasattr(recommendation_system, 'ensemble_weights'):
        weights = recommendation_system.ensemble_weights
    else:
        # Default weights if using traditional model
        weights = {
            'content_based': 0.6,
            'collaborative': 0.4,
            'neural': 0.0
        }
        
    return render_template('recommendation_visualization.html', weights=weights)

# Add dashboard for recommendation analytics (admin only)
@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    """Admin dashboard for recommendation analytics"""
    # For simplicity, just check if the logged in user has ID 1 (first user)
    if session['user_id'] != 1:
        flash('You do not have permission to access this page', 'warning')
        return redirect(url_for('index'))
    
    # Get recommendation weights
    if hasattr(recommendation_system, 'ensemble_weights'):
        weights = recommendation_system.ensemble_weights
    else:
        # Default weights if using traditional model
        weights = {
            'content_based': 0.6,
            'collaborative': 0.4,
            'neural': 0.0
        }
    
    # Get feedback statistics
    feedback = RecommendationFeedback.query.all()
    feedback_df = pd.DataFrame([
        {
            'user_id': f.user_id,
            'product_name': f.product_name,
            'recommendation_type': f.recommendation_type,
            'rating': f.rating,
            'timestamp': f.timestamp
        } for f in feedback
    ])
    
    # Calculate statistics
    if not feedback_df.empty:
        avg_rating_by_type = feedback_df.groupby('recommendation_type')['rating'].mean().to_dict()
        rating_counts = feedback_df.groupby('rating').size().to_dict()
        recent_feedback = feedback_df.sort_values('timestamp', ascending=False).head(10)
    else:
        avg_rating_by_type = {}
        rating_counts = {}
        recent_feedback = pd.DataFrame()
    
    # Get user activity statistics
    activities = UserActivity.query.all()
    activity_df = pd.DataFrame([
        {
            'user_id': a.user_id,
            'product_name': a.product_name,
            'activity_type': a.activity_type,
            'timestamp': a.timestamp
        } for a in activities
    ])
    
    if not activity_df.empty:
        activity_counts = activity_df.groupby('activity_type').size().to_dict()
        top_products = activity_df.groupby('product_name').size().sort_values(ascending=False).head(10).to_dict()
        active_users = activity_df.groupby('user_id').size().sort_values(ascending=False).head(10).to_dict()
    else:
        activity_counts = {}
        top_products = {}
        active_users = {}
    
    return render_template('admin_dashboard.html',
                          avg_rating_by_type=avg_rating_by_type,
                          rating_counts=rating_counts,
                          recent_feedback=recent_feedback,
                          activity_counts=activity_counts,
                          top_products=top_products,
                          active_users=active_users,
                          weights=weights)

# Add route for adjusting recommendation weights (admin only)
@app.route('/admin/adjust_weights', methods=['GET', 'POST'])
@login_required
def adjust_weights():
    """Adjust weights for different recommendation models"""
    # For simplicity, just check if the logged in user has ID 1 (first user)
    if session['user_id'] != 1:
        flash('You do not have permission to access this page', 'warning')
        return redirect(url_for('index'))
    
    # Check if the enhanced system is available
    if not hasattr(recommendation_system, 'set_ensemble_weights'):
        flash('Weight adjustment not available with the current recommendation system', 'warning')
        return redirect(url_for('admin_dashboard'))
    
    if request.method == 'POST':
        content_weight = float(request.form.get('content_weight', 0.3))
        collaborative_weight = float(request.form.get('collaborative_weight', 0.2))
        neural_weight = float(request.form.get('neural_weight', 0.5))
        
        # Update the weights in the recommendation system
        recommendation_system.set_ensemble_weights(
            content_based=content_weight,
            collaborative=collaborative_weight,
            neural=neural_weight
        )
        
        flash('Recommendation weights updated successfully', 'success')
        return redirect(url_for('admin_dashboard'))
    
    # GET request: show the current weights
    current_weights = recommendation_system.ensemble_weights
    
    return render_template('adjust_weights.html', 
                          weights=current_weights)

# Add API endpoint for recommendations (for mobile app)
@app.route('/api/recommendations', methods=['GET'])
def api_recommendations():
    """API endpoint for getting recommendations"""
    user_id = request.args.get('user_id')
    item_name = request.args.get('item_name')
    top_n = request.args.get('top_n', 5, type=int)
    
    # Validate parameters
    if not user_id and not item_name:
        return jsonify({'error': 'Either user_id or item_name must be provided'}), 400
    
    # Get recommendations
    try:
        if user_id:
            user_id = int(user_id)
            # Get user from database to validate
            user = Signup.query.get(user_id)
            if not user:
                return jsonify({'error': 'User not found'}), 404
            
            recommendations = enhanced_recommendations(
                train_data,
                user_id,
                item_name=item_name,
                top_n=top_n
            )
        else:
            recommendations = content_based_recommendations(
                train_data,
                item_name,
                top_n=top_n
            )
        
        if recommendations.empty:
            return jsonify({'message': 'No recommendations found', 'recommendations': []}), 200
        
        # Convert DataFrame to list of dictionaries for JSON response
        recommendations_list = recommendations.to_dict(orient='records')
        
        return jsonify({
            'message': 'Success',
            'recommendations': recommendations_list
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__=='__main__':
    app.run(debug=True)