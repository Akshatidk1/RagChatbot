from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredExcelLoader, UnstructuredPowerPointLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from getpass import getpass
from langchain_core.messages import HumanMessage, AIMessage
import os
import warnings
warnings.filterwarnings("ignore")

os.environ["GOOGLE_API_KEY"] = "AIzaSyAIujOVkqnEXrz7Yj1ztWVfdw_RVWwwVGw"

UPLOAD_DIRECTORY = "./doc_storage"

def resume_score_with_jd(jd):
    # Check if the directory exists
    if not os.path.exists(UPLOAD_DIRECTORY):
        return {"error": True, 'message': f'Directory not found: {UPLOAD_DIRECTORY}'}
    
    # Locate the PDF file
    files = [f for f in os.listdir(UPLOAD_DIRECTORY) if f.endswith(".pdf")]
    if not files:
        return {"error": True, 'message': 'No PDF file found in the directory.'}
    
    file_path = os.path.join(UPLOAD_DIRECTORY, files[0])
    
    # Load the document
    loader = PyPDFLoader(file_path)
    doc = loader.load()
    print(f"Loaded document from {files[0]}")
    
    # Set up GoogleGenerativeAI with the specified model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", convert_system_message_to_human=True)
    
    # Invoke the model with the loaded document and job description
    response = llm.invoke(
        f"""You are an advanced assistant designed to match resumes with job descriptions. 
        Provide a Matching Score and list three strengths and three weaknesses of the candidate 
        based on this resume: {doc} and the following job description: {jd}."""
    )
    
    # Return only the content
    return response.content

# Job description
jd = """Skills:
Python (Programming Language),

Looking For

Quick Learners
Prior knowledge of Python is a must
Comfortable with "Build fast and iterate"
Project members at Sugarfit dont own code. They own features and technologies. This requires understanding of product and business requirements and owning end to end execution of app features
Analyze raw data: assessing quality, cleansing, structuring for downstream processing
Design accurate and scalable prediction algorithms
Collaborate with engineering team to bring analytical prototypes to production
Generate actionable insights for business improvements

Skills

Strong problem solving skills with an emphasis on product development.
Experience using statistical computer languages (R, Python, SLQ, etc.) to manipulate data and draw insights from large data sets.
Knowledge of a variety of machine learning techniques (clustering, decision tree learning, artificial neural networks, etc.) and their real-world advantages/drawbacks.
Excellent written and verbal communication skills for coordinating across teams.
A drive to learn and master new technologies and techniques.
At least 3 years' of experience in quantitative analytics or data modeling
Deep understanding of predictive modeling, machine-learning, clustering and classification techniques, and algorithms
Fluency in a programming language (Python, C,C++, Java, SQL)
Familiarity with Big Data frameworks and visualization tools (Cassandra, Hadoop, Spark, Tableau)

Life At Sugar.fit

Life so good, youd think were kidding:

Competitive salaries. Period.
An extensive medical insurance that looks out for our employees & their dependants. Well love you and take care of you, our promise.
Flexible working hours. Just dont call us at 3AM, we like our sleep schedule.
Tailored vacation & leave policies so that you enjoy every important moment in your life.
A reward system that celebrates hard work and milestones throughout the year. Expect a gift coming your way anytime you kill it here.
Learning and upskilling opportunities. Seriously, not kidding.
Good food, games, and a cool office to make you feel like home. An environment so good, youll forget the term colleagues cant be your friends.

At Sugarfit our mission is to Enable people to lead a Normal life. Through our products and services, we want to reverse and manage diabetes. In pursuit of this objective, we are always looking for excellent team members to fulfill the vision of building these incredible and innovative products.
"""

# Run the function
data = resume_score_with_jd(jd)
print(data)
