
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained machine learning model
with open("internship_match_model.pkl", "rb") as file:
    model = pickle.load(file)

FEATURES = [
    "python",
    "java",
    "sql",
    "ml",
    "web",
    "projects",
    "leetcode",
    "cgpa"
]


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    suggestions = []

    if request.method == "POST":
        try:
            # Read the selected skills
            skills = request.form.getlist("skills")

            # Read and validate the user's inputs
            projects = int(request.form["projects"])
            leetcode = int(request.form["leetcode"])
            cgpa = float(request.form["cgpa"])

            if not 0 <= projects <= 100:
                raise ValueError("Projects must be between 0 and 100.")

            if not 0 <= leetcode <= 100000:
                raise ValueError("Coding problems must be between 0 and 100000.")

            if not 0 <= cgpa <= 10:
                raise ValueError("CGPA must be between 0 and 10.")

            # Prepare input features in the same order used for training
            values = {
                "python": int("Python" in skills),
                "java": int("Java" in skills),
                "sql": int("SQL" in skills),
                "ml": int("Machine Learning" in skills),
                "web": int("Web Development" in skills),
                "projects": projects,
                "leetcode": leetcode,
                "cgpa": cgpa
            }

            input_data = pd.DataFrame(
                [values],
                columns=FEATURES
            )

            # Predict the recommended career track
            result = model.predict(input_data)[0]

            # Generate personalized improvement suggestions
            if values["python"] == 0:
                suggestions.append(
                    "Learn Python fundamentals and practice small programs."
                )

            if values["java"] == 0:
                suggestions.append(
                    "Practice Java, object-oriented programming, and data structures."
                )

            if values["sql"] == 0:
                suggestions.append(
                    "Learn SQL queries, joins, and database fundamentals."
                )

            if values["ml"] == 0:
                suggestions.append(
                    "Study machine learning basics and build a model evaluation project."
                )

            if values["web"] == 0:
                suggestions.append(
                    "Build a responsive website using HTML, CSS, and JavaScript."
                )

            if values["projects"] < 3:
                suggestions.append(
                    "Build another end-to-end project and document it on GitHub."
                )

            if values["leetcode"] < 100:
                suggestions.append(
                    "Practice DSA consistently and analyze time and space complexity."
                )

            if values["cgpa"] < 8.5:
                suggestions.append(
                    "Strengthen core subjects and continue improving your academics."
                )

            if not suggestions:
                suggestions.append(
                    "Keep building projects, practicing DSA, and tailoring your resume to internships."
                )

        except ValueError as error:
            result = None
            suggestions = [f"Please check your inputs: {error}"]

        except Exception:
            result = None
            suggestions = [
                "Something went wrong while processing your inputs. "
                "Please check your entries and try again."
            ]

    return render_template(
        "index.html",
        result=result,
        suggestions=suggestions
    )


if __name__ == "__main__":
    app.run(debug=True)
