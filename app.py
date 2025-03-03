from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import random
import functools
from datetime import datetime, timedelta
import os
import tensorflow as tf
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import the enhanced recommendation system, NLP search, and image feature extractor
from enhanced_recommendation_system import EnhancedRecommendationSystem
from nlp_search import NLPSearch  # Updated version from previous response
from image_feature_extractor import ImageFeatureExtractor
from werkzeug.utils import secure_filename
from multimodal_search import MultimodalSearch
from ai_assistant import AISalesAssistant
from flask import request, jsonify

app = Flask(__name__)

# Load files
trending_products = pd.read_csv("models/trending_products.csv")
train_data = pd.read_csv("models/clean_data.csv")

# Database configuration
app.secret_key = "alskdjfwoeieiurlskdjfslkdjf"
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///ecom.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.permanent_session_lifetime = timedelta(days=7)
db = SQLAlchemy(app)

# Define your model classes
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
nlp_search = None  # Global variable for NLP search
image_search_extractor = None

def get_user_activity_data():
    """Query all user activities from the database and convert to DataFrame"""
    activities = UserActivity.query.all()
    if activities:
        activity_data = [{
            'user_id': activity.user_id,
            'product_name': activity.product_name,
            'activity_type': activity.activity_type,
            'timestamp': activity.timestamp
        } for activity in activities]
        return pd.DataFrame(activity_data)
    return pd.DataFrame(columns=['user_id', 'product_name', 'activity_type', 'timestamp'])

# Initialize database and models within application context
with app.app_context():
    db.create_all()
    user_activity_data = get_user_activity_data()
    
    # Initialize the recommendation system with error handling
    try:
        recommendation_system = EnhancedRecommendationSystem(train_data, user_activity_data)
    except Exception as e:
        print(f"Error initializing Enhanced Recommendation System: {e}")
        print("Falling back to traditional RecommendationModels")
        from recommendation_models import RecommendationModels
        recommendation_system = RecommendationModels(train_data, user_activity_data)

    # Initialize NLP search with database
    try:
        nlp_search = NLPSearch(train_data, db=db)
        print("Initialized NLP Search system")
    except Exception as e:
        print(f"Error initializing NLP Search system: {e}")
        print("NLP search functionality may be limited")

    # Initialize image search with error handling
    try:
        print("About to initialize image search extractor...")
        image_search_extractor = ImageFeatureExtractor(train_data)
        print("Initialized Image Search feature extractor")
    except Exception as e:
        import traceback
        print(f"Error initializing Image Search feature extractor: {e}")
        print(traceback.format_exc())
        print("Image search functionality may be limited")

    # Initialize multimodal search - add this after initializing image_search_extractor and nlp_search
    try:
        print("Initializing Multimodal Search...")
        multimodal_search = MultimodalSearch(image_search_extractor, nlp_search)
        print("Multimodal Search initialized successfully")
    except Exception as e:
        import traceback
        print(f"Error initializing Multimodal Search: {e}")
        print(traceback.format_exc())
        print("Multimodal search functionality may be limited")
        multimodal_search = None

    # AI Sales Assistant
    try:
        ai_assistant = AISalesAssistant(
            product_data=train_data,
            nlp_search=nlp_search,
            image_search_extractor=image_search_extractor,
            multimodal_search=multimodal_search,
            recommendation_system=recommendation_system
        )
        print("Initialized AI Sales Assistant")
    except Exception as e:
        import traceback
        print(f"Error initializing AI Assistant: {e}")
        print(traceback.format_exc())
        ai_assistant = None
        print("AI Assistant functionality may be limited")

    

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Login required decorator
def login_required(view_function):
    @functools.wraps(view_function)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('index'))
        return view_function(*args, **kwargs)
    return decorated_function

# Recommendations functions
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    return text

def content_based_recommendations(train_data, item_name, top_n=10):
    """Get content-based recommendations using the enhanced system"""
    global recommendation_system
    return recommendation_system.get_recommendations(None, item_name, top_n)

def collaborative_filtering_recommendations(train_data, target_user_id, top_n=10):
    """Get collaborative filtering recommendations using the enhanced system"""
    global recommendation_system
    return recommendation_system.get_recommendations(target_user_id, None, top_n)

def enhanced_recommendations(train_data, target_user_id, item_name=None, top_n=10):
    """Get enhanced recommendations using the new system"""
    global recommendation_system
    return recommendation_system.get_recommendations(target_user_id, item_name, top_n=top_n)

def refresh_recommendation_models():
    """Refresh recommendation models with updated user activity data"""
    global recommendation_system, user_activity_data
    user_activity_data = get_user_activity_data()
    recommendation_system.refresh_models(user_activity_data)
    print("Recommendation models refreshed with updated user data")

# Routes
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


# Template filters for datetime formatting
@app.template_filter('parse_iso_datetime')
def parse_iso_datetime(value):
    if value and isinstance(value, str):
        try:
            return datetime.datetime.fromisoformat(value)
        except ValueError:
            # Fall back for older Python versions or different formats
            return datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f")
    return value

@app.template_filter('format_datetime')
def format_datetime(value):
    if value and isinstance(value, datetime.datetime):
        return value.strftime('%b %d, %I:%M %p')
    return value


@app.route("/")
def index():
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(8)]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    
    if 'user_id' in session:
        user_activities = UserActivity.query.filter_by(user_id=session['user_id']).order_by(UserActivity.timestamp.desc()).limit(3).all()
        if user_activities:
            recent_product = user_activities[0].product_name
            try:
                personalized_recommendations = enhanced_recommendations(
                    train_data, session['user_id'], item_name=recent_product, top_n=8
                )
                if not personalized_recommendations.empty:
                    recommendation_types = personalized_recommendations['DominantRecommendationType'].tolist() if 'DominantRecommendationType' in personalized_recommendations.columns else None
                    if 'RecommendationScore' in personalized_recommendations.columns:
                        display_recommendations = personalized_recommendations.drop(columns=['RecommendationScore'])
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
    
    return render_template('index.html', 
                          trending_products=trending_products.head(8), 
                          truncate=truncate,
                          random_product_image_urls=random_product_image_urls,
                          random_price=random.choice(price),
                          user_name=session.get('username', ''))

@app.route("/main")
def main():
    empty_df = pd.DataFrame(columns=['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating', 'DominantRecommendationType'])
    return render_template('main.html', content_based_rec=empty_df, truncate=truncate)

@app.route("/index")
def indexredirect():
    return redirect(url_for('index'))

@app.route("/signup", methods=['POST', 'GET'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        existing_user = Signup.query.filter_by(username=username).first()
        if existing_user:
            return render_template('index.html', signup_message='Username already exists!')

        new_signup = Signup(username=username, email=email, password=password)
        db.session.add(new_signup)
        db.session.commit()
        
        session['user_id'] = new_signup.id
        session['username'] = username
        session.permanent = True

        return redirect(url_for('index'))

@app.route('/signin', methods=['POST', 'GET'])
def signin():
    if request.method == 'POST':
        username = request.form['signinUsername']
        password = request.form['signinPassword']
        
        user = Signup.query.filter_by(username=username, password=password).first()
        if user:
            session['user_id'] = user.id
            session['username'] = username
            session.permanent = True
            
            new_signin = Signin(username=username, password=password)
            db.session.add(new_signin)
            db.session.commit()
            
            return redirect(url_for('index'))
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
        
        if 'user_id' in session:
            new_activity = UserActivity(
                user_id=session['user_id'],
                product_name=prod,
                activity_type='search'
            )
            db.session.add(new_activity)
            db.session.commit()
            
            recommended_products = enhanced_recommendations(
                train_data, session['user_id'], item_name=prod, top_n=nbr
            )
            
            if not recommended_products.empty:
                recommendation_types = recommended_products['DominantRecommendationType'].tolist() if 'DominantRecommendationType' in recommended_products.columns else None
                if 'RecommendationScore' in recommended_products.columns:
                    display_recommendations = recommended_products.drop(columns=['RecommendationScore'])
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
            content_based_rec = content_based_recommendations(train_data, prod, top_n=nbr)
            if not content_based_rec.empty:
                price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
                return render_template('main.html', 
                                      content_based_rec=content_based_rec, 
                                      truncate=truncate,
                                      random_price=random.choice(price))
        
        message = "No recommendations available for this product."
        return render_template('main.html', message=message, content_based_rec=pd.DataFrame(), truncate=truncate)
    
    return render_template('main.html', content_based_rec=pd.DataFrame(), truncate=truncate)

@app.route("/view_product/<path:product_name>")
def view_product(product_name):
    if 'user_id' in session:
        new_activity = UserActivity(
            user_id=session['user_id'],
            product_name=product_name,
            activity_type='view'
        )
        db.session.add(new_activity)
        db.session.commit()
        
        user_activity_count = UserActivity.query.filter_by(user_id=session['user_id']).count()
        if user_activity_count % 5 == 0:  # Refresh every 5 activities
            refresh_recommendation_models()
    
    product = train_data[train_data['Name'] == product_name].iloc[0].to_dict() if not train_data[train_data['Name'] == product_name].empty else None
    if not product:
        return "Product not found", 404
    
    price = random.choice([40, 50, 60, 70, 100, 122, 106, 50, 30, 50])
    return render_template('product_detail.html', product=product, price=price)

@app.route('/profile')
@login_required
def profile():
    user = Signup.query.get(session['user_id'])
    if not user:
        return redirect(url_for('logout'))
    
    activities = UserActivity.query.filter_by(user_id=user.id).order_by(UserActivity.timestamp.desc()).limit(10).all()
    recommendations = pd.DataFrame()
    similar_users = []
    
    if activities:
        recent_product = activities[0].product_name
        try:
            try:
                results = recommendation_system.get_advanced_recommendations(
                    user.id, item_name=recent_product, top_n=5, include_similar_users=True
                )
                recommendations = results['recommendations']
                similar_users = results.get('similar_users', [])
            except (AttributeError, Exception) as e:
                print(f"Falling back to basic recommendations: {e}")
                recommendations = enhanced_recommendations(
                    train_data, session['user_id'], item_name=recent_product, top_n=5
                )
        except Exception as e:
            print(f"Error getting recommendations: {e}")
    
    return render_template('profile.html', 
                          user=user, 
                          activities=activities, 
                          recommendations=recommendations, 
                          truncate=truncate,
                          personalized=True,
                          similar_users=similar_users,
                          ai_assistant=ai_assistant)  # Pass ai_assistant to the template

@app.route('/recommendation_feedback', methods=['POST'])
@login_required
def recommendation_feedback():
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

@app.route('/recommendation_visualization')
def recommendation_visualization():
    weights = recommendation_system.ensemble_weights if hasattr(recommendation_system, 'ensemble_weights') else {
        'content_based': 0.6, 'collaborative': 0.4, 'neural': 0.0
    }
    return render_template('recommendation_visualization.html', weights=weights)

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if session['user_id'] != 1:
        flash('You do not have permission to access this page', 'warning')
        return redirect(url_for('index'))
    
    weights = recommendation_system.ensemble_weights if hasattr(recommendation_system, 'ensemble_weights') else {
        'content_based': 0.6, 'collaborative': 0.4, 'neural': 0.0
    }
    
    feedback = RecommendationFeedback.query.all()
    feedback_df = pd.DataFrame([{
        'user_id': f.user_id, 'product_name': f.product_name, 'recommendation_type': f.recommendation_type,
        'rating': f.rating, 'timestamp': f.timestamp
    } for f in feedback])
    
    if not feedback_df.empty:
        avg_rating_by_type = feedback_df.groupby('recommendation_type')['rating'].mean().to_dict()
        rating_counts = feedback_df.groupby('rating').size().to_dict()
        recent_feedback = feedback_df.sort_values('timestamp', ascending=False).head(10)
    else:
        avg_rating_by_type, rating_counts, recent_feedback = {}, {}, pd.DataFrame()
    
    activities = UserActivity.query.all()
    activity_df = pd.DataFrame([{
        'user_id': a.user_id, 'product_name': a.product_name, 'activity_type': a.activity_type,
        'timestamp': a.timestamp
    } for a in activities])
    
    if not activity_df.empty:
        activity_counts = activity_df.groupby('activity_type').size().to_dict()
        top_products = activity_df.groupby('product_name').size().sort_values(ascending=False).head(10).to_dict()
        active_users = activity_df.groupby('user_id').size().sort_values(ascending=False).head(10).to_dict()
    else:
        activity_counts, top_products, active_users = {}, {}, {}
    
    return render_template('admin_dashboard.html',
                          avg_rating_by_type=avg_rating_by_type,
                          rating_counts=rating_counts,
                          recent_feedback=recent_feedback,
                          activity_counts=activity_counts,
                          top_products=top_products,
                          active_users=active_users,
                          activity_df=activity_df,
                          weights=weights)

@app.route('/admin/adjust_weights', methods=['GET', 'POST'])
@login_required
def adjust_weights():
    if session['user_id'] != 1:
        flash('You do not have permission to access this page', 'warning')
        return redirect(url_for('index'))
    
    if not hasattr(recommendation_system, 'set_ensemble_weights'):
        flash('Weight adjustment not available with the current recommendation system', 'warning')
        return redirect(url_for('admin_dashboard'))
    
    if request.method == 'POST':
        content_weight = float(request.form.get('content_weight', 0.3))
        collaborative_weight = float(request.form.get('collaborative_weight', 0.2))
        neural_weight = float(request.form.get('neural_weight', 0.5))
        
        recommendation_system.set_ensemble_weights(
            content_based=content_weight,
            collaborative=collaborative_weight,
            neural=neural_weight
        )
        
        flash('Recommendation weights updated successfully', 'success')
        return redirect(url_for('admin_dashboard'))
    
    current_weights = recommendation_system.ensemble_weights
    return render_template('adjust_weights.html', weights=current_weights)

@app.route('/api/recommendations', methods=['GET'])
def api_recommendations():
    user_id = request.args.get('user_id')
    item_name = request.args.get('item_name')
    top_n = request.args.get('top_n', 5, type=int)
    
    if not user_id and not item_name:
        return jsonify({'error': 'Either user_id or item_name must be provided'}), 400
    
    try:
        if user_id:
            user_id = int(user_id)
            user = Signup.query.get(user_id)
            if not user:
                return jsonify({'error': 'User not found'}), 404
            recommendations = enhanced_recommendations(train_data, user_id, item_name=item_name, top_n=top_n)
        else:
            recommendations = content_based_recommendations(train_data, item_name, top_n=top_n)
        
        if recommendations.empty:
            return jsonify({'message': 'No recommendations found', 'recommendations': []}), 200
        
        recommendations_list = recommendations.to_dict(orient='records')
        return jsonify({'message': 'Success', 'recommendations': recommendations_list}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/nlp_search", methods=['POST', 'GET'])
def nlp_search_route():
    if request.method == 'POST':
        query = request.form.get('query')
        top_n = int(request.form.get('top_n', 10))
        
        if not query:
            return render_template('nlp_search.html', message="Please enter a search query")
        
        if 'user_id' in session:
            new_activity = UserActivity(
                user_id=session['user_id'],
                product_name=query[:100],
                activity_type='nlp_search'
            )
            db.session.add(new_activity)
            db.session.commit()
        
        try:
            search_results, query_info = nlp_search.enhanced_search(query, user_id=session.get('user_id'), top_n=top_n)
            
            if search_results.empty:
                message = "No products found matching your search criteria."
                return render_template('nlp_search.html', message=message, query=query)
            
            price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
            search_metadata = {
                'original_query': query,
                'detected_attributes': {k: v for k, v in query_info['attributes'].items() if v}
            }
            
            return render_template('nlp_search.html', 
                                  search_results=search_results, 
                                  truncate=truncate,
                                  random_price=random.choice(price),
                                  query=query,
                                  search_metadata=search_metadata)
        except Exception as e:
            import traceback
            print(f"Error in NLP search: {e}")
            print(traceback.format_exc())
            message = "An error occurred while processing your search request."
            return render_template('nlp_search.html', message=message, query=query)
    
    return render_template('nlp_search.html')

@app.route("/nlp_visualization")
def nlp_visualization():
    """Show visualization of how the NLP search system works"""
    return render_template('nlp_visualization.html')

@app.route("/image_search", methods=['POST', 'GET'])
def image_search():
    if request.method == 'POST':
        if image_search_extractor is None:
            message = "Image search functionality is not available at the moment."
            return render_template('image_search.html', message=message)
        
        if 'image_file' not in request.files:
            message = "No file part in the request."
            return render_template('image_search.html', message=message)
        
        file = request.files['image_file']
        if file.filename == '':
            message = "No image selected for uploading."
            return render_template('image_search.html', message=message)
        
        if file and allowed_file(file.filename):
            file_data = file.read()
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(file_path, 'wb') as f:
                f.write(file_data)
            
            if 'user_id' in session:
                new_activity = UserActivity(
                    user_id=session['user_id'],
                    product_name=f"Image search: {filename}",
                    activity_type='image_search'
                )
                db.session.add(new_activity)
                db.session.commit()
            
            try:
                similar_products = image_search_extractor.find_similar_products(file_data, top_n=10)
                
                if similar_products.empty:
                    message = "No similar products found for your image."
                    return render_template('image_search.html', 
                                          message=message,
                                          uploaded_image=f"uploads/{filename}")
                
                price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
                return render_template('image_search.html', 
                                      search_results=similar_products, 
                                      truncate=truncate,
                                      random_price=random.choice(price),
                                      uploaded_image=f"uploads/{filename}")
            except Exception as e:
                import traceback
                print(f"Error in image search: {e}")
                print(traceback.format_exc())
                message = "An error occurred while processing your image search request."
                return render_template('image_search.html', 
                                      message=message,
                                      uploaded_image=f"uploads/{filename}")
        else:
            message = "Allowed image types are: png, jpg, jpeg, gif"
            return render_template('image_search.html', message=message)
    
    return render_template('image_search.html')

@app.route("/image_search_visualization")
def image_search_visualization():
    """Show visualization of how the image search system works"""
    return render_template('image_search_visualization.html')

@app.route("/multimodal_search_route", methods=['POST', 'GET'])
def multimodal_search_route(): 
    """Handle multimodal search requests combining image and text"""
    if multimodal_search is None:  # This refers to the global variable
        return render_template('multimodal_search.html', 
                              message="Multimodal search is not available at the moment.")
    
    if request.method == 'POST':
        # Check if image file is provided
        if 'image_file' not in request.files:
            return render_template('multimodal_search.html', 
                                  message="No image file uploaded. Please provide both image and text.")
        
        file = request.files['image_file']
        text_query = request.form.get('text_query', '')
        image_weight = float(request.form.get('image_weight', 50)) / 100  # Convert to 0-1 scale
        text_weight = 1.0 - image_weight
        
        if file.filename == '':
            return render_template('multimodal_search.html', 
                                  message="No image selected. Please upload an image.",
                                  text_query=text_query)
        
        if not text_query.strip():
            return render_template('multimodal_search.html', 
                                  message="No text query provided. Please describe what you're looking for.",
                                  text_query=text_query)
        
        if file and allowed_file(file.filename):
            file_data = file.read()
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(file_path, 'wb') as f:
                f.write(file_data)
            
            # Record user activity if logged in
            if 'user_id' in session:
                new_activity = UserActivity(
                    user_id=session['user_id'],
                    product_name=f"Multimodal search: {text_query[:50]}",
                    activity_type='multimodal_search'
                )
                db.session.add(new_activity)
                db.session.commit()
            
            try:
                # Perform multimodal search
                search_results, query_info = multimodal_search.search(
                    file_data, 
                    text_query, 
                    top_n=10,
                    weight_image=image_weight,
                    weight_text=text_weight
                )
                
                if search_results.empty:
                    message = "No products found matching your criteria."
                    return render_template('multimodal_search.html', 
                                          message=message,
                                          uploaded_image=f"uploads/{filename}",
                                          text_query=text_query)
                
                price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
                search_metadata = {
                    'original_query': text_query,
                    'detected_attributes': {k: v for k, v in query_info['attributes'].items() if v}
                }
                
                return render_template('multimodal_search.html', 
                                      search_results=search_results, 
                                      truncate=truncate,
                                      random_price=random.choice(price),
                                      uploaded_image=f"uploads/{filename}",
                                      text_query=text_query,
                                      search_metadata=search_metadata)
                
            except Exception as e:
                import traceback
                print(f"Error in multimodal search: {e}")
                print(traceback.format_exc())
                message = "An error occurred while processing your search request."
                return render_template('multimodal_search.html', 
                                      message=message,
                                      uploaded_image=f"uploads/{filename}",
                                      text_query=text_query)
        else:
            message = "Allowed image types are: png, jpg, jpeg, gif"
            return render_template('multimodal_search.html', 
                                  message=message,
                                  text_query=text_query)
    
    return render_template('multimodal_search.html')

@app.route("/multimodal_visualization")
def multimodal_visualization():
    """Show visualization of how the multimodal search system works"""
    return render_template('multimodal_visualization.html')



@app.route("/ai_assistant")
def ai_assistant_page():
    """Render the AI shopping assistant page"""
    return render_template('ai_assistant.html')

@app.route("/api/ai_assistant/chat", methods=['POST'])
@login_required
def ai_assistant_chat():
    """API endpoint for AI assistant chat"""
    if ai_assistant is None:
        return jsonify({
            'success': False,
            'message': "AI Assistant is not available at the moment.",
            'response': {
                'text': "I'm sorry, but I'm not available right now. Please try again later.",
                'products': []
            }
        }), 503

    # Get user message
    data = request.json
    message = data.get('message', '')
    
    if not message.strip():
        return jsonify({
            'success': False,
            'message': "Please provide a message.",
            'response': {
                'text': "I didn't receive any message. How can I help you?",
                'products': []
            }
        }), 400
    
    # Process message
    try:
        user_id = session.get('user_id')
        response = ai_assistant.process_message(user_id, message)
        
        # Record the interaction
        new_activity = UserActivity(
            user_id=user_id,
            product_name=message[:100],
            activity_type='ai_assistant'
        )
        db.session.add(new_activity)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'response': response
        })
    except Exception as e:
        print(f"Error processing AI assistant message: {e}")
        return jsonify({
            'success': False,
            'message': str(e),
            'response': {
                'text': "I encountered an error processing your message. Could you try again?",
                'products': []
            }
        }), 500

@app.route("/api/ai_assistant/suggestions", methods=['GET'])
@login_required
def ai_assistant_suggestions():
    """API endpoint for AI assistant suggestions"""
    if ai_assistant is None:
        return jsonify({
            'success': False,
            'suggestions': []
        }), 503
    
    try:
        user_id = session.get('user_id')
        user_preferences = ai_assistant.get_user_preferences(user_id)
        
        # Generate suggestions based on preferences
        suggestions = [
            "Show me new arrivals",
            "What's trending today?",
            "Help me find a gift"
        ]
        
        # Add personalized suggestions based on preferences
        preferred_categories = user_preferences.get('preferred_categories', {})
        preferred_brands = user_preferences.get('preferred_brands', {})
        style_preferences = user_preferences.get('style_preferences', {})
        
        if preferred_categories:
            top_category = max(preferred_categories.items(), key=lambda x: x[1])[0]
            suggestions.append(f"Show me {top_category} products")
        
        if preferred_brands:
            top_brand = max(preferred_brands.items(), key=lambda x: x[1])[0]
            suggestions.append(f"What's new from {top_brand}?")
        
        if 'color' in style_preferences:
            top_color = max(style_preferences['color'].items(), key=lambda x: x[1])[0]
            
            if preferred_categories:
                top_category = max(preferred_categories.items(), key=lambda x: x[1])[0]
                suggestions.append(f"Find {top_color} {top_category}")
        
        return jsonify({
            'success': True,
            'suggestions': suggestions
        })
    except Exception as e:
        print(f"Error generating AI assistant suggestions: {e}")
        return jsonify({
            'success': False,
            'suggestions': [
                "Show me popular products",
                "What's on sale?",
                "Help me find a product"
            ]
        })
    
# Add this route to app.py
@app.route("/ai_assistant_visualization")
def ai_assistant_visualization():
    """Show visualization of how the AI Assistant works"""
    return render_template('ai_assistant_visualization.html')


if __name__ == '__main__':
    app.run(debug=True)