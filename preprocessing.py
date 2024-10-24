import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Function to clean text
def clean_text(text):
    """Cleans the input text by removing special characters and normalizing."""
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    return text.lower().strip()

# Function to match resume features to job features
def match_resume_to_job(resume_features, job_features):
    """Matches resume features with job features based on keyword overlap."""
    resume_keywords = set(resume_features.split())
    job_keywords = set(job_features.split())
    match_score = len(resume_keywords.intersection(job_keywords))
    return match_score

# Load the data
resume = pd.read_csv("clean_resume_resume.csv")
job = pd.read_csv("jobs_resumeset_with_features.csv")

# Display the first few rows of the resume DataFrame
print(resume.head(10))
print(resume.info())

# Drop rows with missing values in the resume DataFrame
resume = resume.dropna()
print(resume.info())

# Display the first few rows of the job DataFrame
print(job.head(10))
print(job.info())

# Tokenize and clean the 'Feature' column
resume['Feature_cleaned'] = resume['Feature'].apply(clean_text)

# Analyze category distribution
category_distribution = resume['Category'].value_counts()
word_freq = Counter(' '.join(resume['Feature_cleaned']).split()).most_common(10)
words, counts = zip(*word_freq)

# Set plot style
sns.set(style="whitegrid")

# Plot the distribution of job categories
plt.figure(figsize=(12, 8))
sns.barplot(x=category_distribution.values, y=category_distribution.index, palette="viridis")
plt.title('Job Category Distribution', fontsize=16)
plt.xlabel('Number of Resumes', fontsize=14)
plt.ylabel('Job Category', fontsize=14)
plt.tight_layout()
plt.show()

# Plot the top 10 most common words
plt.figure(figsize=(10, 6))
sns.barplot(x=list(counts), y=list(words), palette="Blues_d")
plt.title('Top 10 Most Frequent Words in Resumes', fontsize=16)
plt.xlabel('Frequency', fontsize=14)
plt.ylabel('Words', fontsize=14)
plt.tight_layout()
plt.show()

# Education analysis - Extract common degrees
def extract_degrees(text):
    degrees = ['bachelor', 'master', 'mba', 'phd', 'associate']
    found_degrees = [degree for degree in degrees if degree in text.lower()]
    return found_degrees

resume['Degrees'] = resume['Feature'].apply(extract_degrees)
all_degrees = [degree for sublist in resume['Degrees'] for degree in sublist]
degree_counts = Counter(all_degrees)

# Plot common degrees
degrees, counts = zip(*degree_counts.most_common(5))
plt.figure(figsize=(10, 5))
plt.bar(degrees, counts)
plt.title('Top 5 Degrees')
plt.xticks(rotation=45)
plt.show()

# Function to extract min and max experience
def extract_features(feature_str):
    exp_pattern = r'(\d+)\s*to\s*(\d+)\s*Years'
    match = re.search(exp_pattern, feature_str)
    
    if match:
        min_exp = int(match.group(1))
        max_exp = int(match.group(2))
    else:
        min_exp = max_exp = None
    
    description = re.sub(exp_pattern, '', feature_str).strip()
    return pd.Series([min_exp, max_exp, description])

# Apply the function to the 'Features' column
job[['Min_Experience', 'Max_Experience', 'Description']] = job['Features'].apply(extract_features)

# Display the modified DataFrame
print(job[['Role', 'Min_Experience', 'Max_Experience', 'Description']])

# Handle infinite values in the job DataFrame
job[['Min_Experience', 'Max_Experience']] = job[['Min_Experience', 'Max_Experience']].replace([float('inf'), -float('inf')], pd.NA)

# Plot the distributions of Min and Max Experience
plt.figure(figsize=(12, 6))

# Subplot for Min Experience
plt.subplot(1, 2, 1)
sns.histplot(job['Min_Experience'].dropna(), bins=10, kde=True)
plt.title('Distribution of Minimum Experience')
plt.xlabel('Years of Minimum Experience')
plt.ylabel('Frequency')

# Subplot for Max Experience
plt.subplot(1, 2, 2)
sns.histplot(job['Max_Experience'].dropna(), bins=10, kde=True)
plt.title('Distribution of Maximum Experience')
plt.xlabel('Years of Maximum Experience')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Example usage of the matching function (optional testing)
if __name__ == "__main__":
    sample_resume = "Python, Data Analysis, Machine Learning"
    sample_job = "Data Science, Machine Learning"

    # Clean and match
    cleaned_resume = clean_text(sample_resume)
    cleaned_job = clean_text(sample_job)
    score = match_resume_to_job(cleaned_resume, cleaned_job)

    # Output the result
    print(f"Match Score: {score}")
