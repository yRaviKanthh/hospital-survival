🏥 Hospital Patient Survival Prediction using Machine Learning & Django
This is a Hospital Patient Survival Prediction Web Application built using Python, Machine Learning algorithms, and Django.
The application predicts whether a patient will survive for 1 year based on medical history, treatment, and lifestyle factors.

📌 Features
✅ Predict patient survival using Decision Tree Classifier
✅ Web interface built with Django
✅ Handles missing values in datasets automatically
✅ One-hot encodes categorical data for better model performance
✅ User-friendly web form for inputting patient details
✅ Displays the prediction result dynamically

📂 Project Structure
graphql
Copy
Edit
GUI/
├── hospital_prediction/
│   ├── __pycache__/               # Compiled Python files
│   ├── templates/                 # HTML templates for frontend
│   │   ├── index.html              # Home Page
│   │   ├── training_data.html      # Prediction Result Page
│   ├── __init__.py                 # Package initialization
│   ├── views.py                    # Handles user requests & ML model
│   ├── models.py                   # Model definitions (if needed)
│   ├── urls.py                      # Django URL routing
│   ├── admin.py                     # Django admin setup
│   ├── apps.py                      # Django app configuration
│   ├── tests.py                     # Test cases
├── db.sqlite3                        # SQLite database file
├── training_data.csv                  # Dataset for training the model
├── manage.py                          # Django project management
├── README.md                          # Project documentation
├── .gitignore                         # Ignore unnecessary files
🎯 Technologies Used
Backend: Python, Django
Machine Learning: Scikit-Learn (Decision Tree Classifier)
Data Handling: Pandas, NumPy
Frontend: HTML, CSS, JavaScript
🔧 Installation & Setup
🔹 Clone the Repository
bash
Copy
Edit
git clone https://github.com/yRaviKanthh/hospital-patient-survival.git
cd hospital-patient-survival
🔹 Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
🔹 Run the Application
bash
Copy
Edit
python manage.py runserver
🔗 Open in your browser: http://127.0.0.1:8000/

📸 Screenshots
🔹 Prediction Page

🔹 Database

🔹 Result

🏆 Future Enhancements
✅ Improve prediction accuracy with ensemble learning models
✅ Add data visualization for patient survival trends
✅ Deploy on cloud platforms
✅ Implement an API for third-party integration
💡 Need Help?
Feel free to open an issue or contribute to improve this project! 😊
⭐ If you like this project, give it a star on GitHub! ⭐
