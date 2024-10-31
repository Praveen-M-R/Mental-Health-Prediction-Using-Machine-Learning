from flask import Flask, request, render_template, redirect, url_for, session
import joblib

app = Flask(__name__)
app.secret_key = 'your_secret_key'

model1 = joblib.load('model/gradient_boosting_model.pkl')
model2 = joblib.load('model/random_forest_model.pkl')
model3 = joblib.load('model/logistic_regression_model.pkl')
model4 = joblib.load('model/decision_tree_model.pkl')

questions = [
    {"question": "How old are you?", "type": "number"},
    {"question": "Do you feel often nervous?", "type": "multiple_choice","options": ["Yes", "Sometimes", "No"]},
    {"question": "Do you feel often panic?","type": "multiple_choice","options": ["Yes", "Sometimes", "No"]},
    {"question": "Do you get sudden bursts of rapid breathing?","type": "multiple_choice","options": ["Yes", "Sometimes", "No"]},
    {"question": "Do you get sudden bursts of excessive Sweating?","type": "multiple_choice","options": ["Yes", "Sometimes", "No"]},
    {"question": "Do you have trouble focusing on a task?","type": "multiple_choice","options": ["Yes", "Sometimes", "No"]},
    {"question": "Do you have trouble in sleeping?","type": "multiple_choice","options": ["Yes", "Sometimes", "No"]},
    {"question": "Are you having trouble with work?","type": "multiple_choice","options": ["Yes", "Sometimes", "No"]},
    {"question": "Do you feel hopeless?","type": "multiple_choice","options": ["Yes", "Sometimes", "No"]},
    {"question": "Do you feel angry?","type": "multiple_choice","options": ["Yes", "Sometimes", "No"]},
    {"question": "Do you overeact?","type": "multiple_choice","options": ["Yes", "Sometimes", "No"]},
    {"question": "Have you changed your eating pattern?","type": "multiple_choice","options": ["Yes", "Sometimes", "No"]},
    {"question": "Do you ever get suicidal thoughts?","type": "multiple_choice","options": ["Yes", "Sometimes", "No"]},
    {"question": "Are you always tired?","type": "multiple_choice","options": ["Yes", "Sometimes", "No"]},
    {"question": "Do you have a close friend(s)?","type": "multiple_choice","options": ["Yes", "No"]},
    {"question": "Are you addicted to social media?","type": "multiple_choice","options": ["Yes", "Sometimes", "No"]},
    {"question": "Have you gained weight of recently?","type": "multiple_choice","options": ["Yes", "Sometimes", "No"]},
    {"question": "Are you an introvert?","type": "multiple_choice","options": ["Yes", "Sometimes", "No"]},
    {"question": "Do you ever get pop ups of a stressful memory?","type": "multiple_choice","options": ["Yes", "Sometimes", "No"]},
    {"question": "Do you frequently experience nighmares?","type": "multiple_choice","options": ["Yes", "Sometimes", "No"]},
    {"question": "Do you often avoid people or activities?","type": "multiple_choice","options": ["Yes", "Sometimes", "No"]},
    {"question": "Do you often find yourself feeling negative?","type": "multiple_choice","options":["Yes", "Sometimes", "No"]},
    {"question": "Do you have trouble concentrating","type": "multiple_choice","options": ["Yes", "Sometimes", "No"]},
    {"question": "Do you often blame yourself for an unfortunate event?","type": "multiple_choice","options": ["Yes", "Sometimes", "No"]},
    {"question": "Do you ever get halluciations?","type": "multiple_choice","options": ["Yes", "Sometimes", "No"]},
    {"question": "Do you exhibit repetitive behavior?","type": "multiple_choice","options": ["Yes", "Sometimes", "No"]},
    {"question": "Do you get seasonal depression?","type": "multiple_choice","options": ["Yes", "Sometimes", "No"]},
    {"question": "Have you recently experience an increase in energy?","type": "multiple_choice","options": ["Yes", "Sometimes", "No"]}
]

@app.route('/')
def home():
    session.clear()
    return render_template('home.html')

@app.route('/questionnaire', methods=['GET', 'POST'])
def questionnaire():
    if 'question_index' not in session:
        session['question_index'] = 0
    if 'answers' not in session:
        session['answers'] = []

    question_index = session['question_index']

    if request.method == 'POST':
        # Check if the user clicked a "Back" button
        if 'back' in request.form:
            # Clear the current answer if going back
            if session['answers']:
                session['answers'].pop()
            session['question_index'] = max(0, session['question_index'] - 1)
        else:
            # Save the answer and move to the next question
            answer = request.form.get('answer')
            session['answers'].append(answer)
            session['question_index'] += 1

        return redirect(url_for('questionnaire'))

    if question_index < len(questions):
        question = questions[question_index]
        progress_percentage = (question_index / len(questions)) * 100
        return render_template('question.html', question=question, progress_percentage=progress_percentage)
    else:
        return redirect(url_for('submit'))



@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        if 'back' in request.form:
            session['answers'].pop()
            session['question_index'] -= 1
            return redirect(url_for('questionnaire'))
        # Directly use Model 1 for prediction
        return redirect(url_for('predict'))
    
    answers = session.get('answers', [])
    return render_template('submit.html',questions=questions, answers=answers)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    models = ['Gradient Boost with 86% Accuracy', 'Random Forest with 79% Accuracy', 'Logistic Regression with 80% Accuracy','Decision tree with 72% Accuracy']  # Ensure all models are listed

    # Determine the selected model
    selected_model = request.form.get('model') if request.method == 'POST' else request.args.get('model')

    answers = session.get('answers', [])
    if not answers:
        return "No answers found in session."

    try:
        age = int(answers[0])
        processed_answers = [age] + [2 if ans == 'Yes' else 1 if ans == 'Sometimes' else 0 for ans in answers[1:]]
    except (ValueError, IndexError) as e:
        return f"Error processing answers: {e}"

    # Select the correct prediction model
    if selected_model == 'Gradient Boost with 86% Accuracy':
        prediction = model1.predict([processed_answers])[0]
    elif selected_model == 'Random Forest with 79% Accuracy':
        prediction = model2.predict([processed_answers])[0]
    elif selected_model == 'Random Forest with 79% Accuracy':
        prediction = model4.predict([processed_answers])[0]
    else:
        prediction = model3.predict([processed_answers])[0]

    return render_template('result.html', prediction=prediction, answers=processed_answers, models=models, selected_model=selected_model)





if __name__ == '__main__':
    app.run(debug=True)