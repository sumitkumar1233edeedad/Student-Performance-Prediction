import random
import pandas as pd

# Number of rows
num_rows = 10000

data = []

for _ in range(num_rows):
    study_hours = random.randint(0, 10)
    attendance = random.randint(50, 100)
    previous_score = random.randint(40, 90)
    sleep_hours = random.randint(4, 9)
    
    # Simple formula to calculate final score
    final_score = (
        study_hours * 5 +
        attendance * 0.2 +
        previous_score * 0.5 +
        sleep_hours * 2
    )
    
    final_score = round(min(final_score, 100))  # Limit max to 100
    
    data.append([
        study_hours,
        attendance,
        previous_score,
        sleep_hours,
        final_score
    ])

# Create DataFrame
df = pd.DataFrame(data, columns=[
    "study_hours",
    "attendance",
    "previous_score",
    "sleep_hours",
    "final_score"
])

# Save to CSV
df.to_csv("student_prediction.csv", index=False)

print("Dataset generated successfully!")