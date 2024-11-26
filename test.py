import json
import os
import csv
import PyPDF2
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from langchain.prompts import FewShotPromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from groq import Groq
from langchain_groq import ChatGroq
import uvicorn
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import List

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add CORS Middleware with more specific configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Load questions document once at startup
def load_questions_doc():
    try:
        # return pd.read_json("questions_data.json")
        return pd.read_csv("question_ans_doc_final.csv")
    except Exception as e:
        print(f"Error loading questions: {e}")
        return None

questions_doc = load_questions_doc()


class ResumeRequest(BaseModel):
    file_name: str


def load_pdf_text(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def chunk_text(text, chunk_size=500, chunk_overlap=80):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables")
            
    print("Initializing OpenAI embeddings")
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    
    # Log before vector store creation
    print("About to create vector store...")
    
    # Create vector store with try-except
    # try:
    vector_store = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        persist_directory="resume_chroma_db"
    )
    print("Vector store created successfully")
    # except Exception as e:
    #     logger.error(f"Failed during Chroma creation: {str(e)}")
    #     raise
    return vector_store

def generate_llm_response(user_input):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OpenAI API key not found")

    llm = ChatOpenAI(
        api_key=openai_api_key,
        model="gpt-4o-mini",
        temperature=0
    )

    examples = [
        {"input": "What is the number of years of Experience of the candidate", "output": "5 years"},
        {"input": "What was the previous role of the candidate.", "output": "Junior Developer"},
        {"input": "What is the email id of the candidate.", "output": "amy0011@gmail.com"},
        {"input": "What are the skillsets of the candidate?", "output": "Python, SQL, Machine Learning"},
        {"input": "What is the tech stack of the candidate?", "output": "Python, SQL, Reactjs, Nextjs, C++"}
    ]

    example_prompt = ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")])
    few_shot_prompt = FewShotChatMessagePromptTemplate(example_prompt=example_prompt, examples=examples)

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", """
                    You are a Resume Extractor that takes resume chunks and extracts specific details. The details to be extracted are:
                    - Experience Level
                    - Role
                    - Previous Role
                    - Skillsets
                    - Tech Stack
                    - Email
                    For the provided resume context, answer the questions below by extracting the relevant information.
                    Answer very briefly in about 2- 3 words not more than that.
                    """),
        few_shot_prompt,
        ("human", "{input}")
    ])

    chain = final_prompt | llm
    response = chain.invoke({"input": user_input})
    print("\n\n\n",response)
    return response.content

def save_to_csv(extracted_info, csv_output_path):
    write_header = not os.path.exists(csv_output_path)
    with open(csv_output_path, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=extracted_info.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(extracted_info)

def process_pdf(pdf_path):
    text = load_pdf_text(pdf_path)
    text_chunks = chunk_text(text)
    vector_store = get_vector_store(text_chunks)

    extracted_info = {
        "experience": "",
        "previous_role": "",
        "email_id": "",
        "skillset": "",
        "location": ""
    }

    questions_list = [
        "What is the number of years of Experience of the candidate?",
        "What was the previous role of the candidate?",
        "What is the email id of the candidate?",
        "What are the skillsets of the candidate?",
        "What is the location of the candidate? Where are they situated?"
    ]
    keys = ["experience", "previous_role", "email_id", "skillset", "location"]

    for idx, question in enumerate(questions_list):
        retrieved_chunk = vector_store.similarity_search(question, k=3)
        user_input = f"Context: {retrieved_chunk}\n\nBased on the above context provide the answer to the question: {question}"
        response = generate_llm_response(user_input)
        extracted_info[keys[idx]] = response

    print("\n\n\n", extracted_info)
    return extracted_info



@app.post("/extract-resume-info")
async def extract_resume_info(
    file: UploadFile = File(...),
    file_name: str = Form(...)
):
    try:
        # Log received parameters
        print(f"Received file_name: {file_name}")
        print(f"Received file: {file.filename}")

        # Validation for file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        temp_path = f"temp_{file.filename}"
        
        # Save uploaded file temporarily
        content = await file.read()
        with open(temp_path, "wb") as temp_file:
            temp_file.write(content)
        
        # Process the PDF
        extracted_info = process_pdf(temp_path)
        
        # Save to CSV
        csv_path = "resume_details.csv"
        save_to_csv(extracted_info, csv_path)
        
        return {
            "status": "success",
            "file_name": file_name,
            "extracted_info": extracted_info,
            "csv_file": csv_path
        }
    
    except HTTPException as http_exc:
        raise http_exc  # Reraise HTTP exceptions
    except Exception as e:
        print(f"Error during resume extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)



def get_doc_questions_in_a_list_of_groups(questions_doc):
    if questions_doc is None:
        raise HTTPException(status_code=500, detail="Questions document not loaded")
        
    mcq_questions = questions_doc[questions_doc["Type"]=="MCQ"]
    basic_mcq_questions = mcq_questions[mcq_questions["Level"]=="basic"]
    inter_mcq_questions = mcq_questions[mcq_questions["Level"]=="intermediate"]
    adv_mcq_questions = mcq_questions[mcq_questions["Level"]=="advanced"]

    coding_questions = questions_doc[questions_doc["Type"]=="Coding"]
    basic_coding_questions = coding_questions[coding_questions["Level"]=="basic"]
    inter_coding_questions = coding_questions[coding_questions["Level"]=="intermediate"]
    adv_coding_questions = coding_questions[coding_questions["Level"]=="advanced"]

    return {
            "basic_mcq_questions": basic_mcq_questions,
            "inter_mcq_questions": inter_mcq_questions,
            "adv_mcq_questions": adv_mcq_questions,
            "basic_coding_questions": basic_coding_questions,
            "inter_coding_questions": inter_coding_questions,
            "adv_coding_questions": adv_coding_questions
        }


def select_questions_based_on_experience(questions, experience):
    mcq_list = []
    coding_list = []
    
    try:
        # Helper function to safely sample from DataFrame
        def safe_sample(df, n):
            if isinstance(df, pd.DataFrame) and len(df) >= n:
                return df.sample(n)
            elif isinstance(df, pd.DataFrame):
                # Log a warning if not enough rows
                print(f"Warning: Requested {n} samples but only {len(df)} available.")
                return df.sample(min(len(df), n))
            else:
                raise ValueError(f"Expected a DataFrame but got {type(df)}")

        if experience == 0:
            mcq_list.append(safe_sample(questions.get("basic_mcq_questions", pd.DataFrame()), 5))
            mcq_list.append(safe_sample(questions.get("inter_mcq_questions", pd.DataFrame()), 10))
            mcq_list.append(safe_sample(questions.get("adv_mcq_questions", pd.DataFrame()), 5))
            coding_list.append(safe_sample(questions.get("basic_coding_questions", pd.DataFrame()), 2))
        
        elif 0 < experience <= 3:
            mcq_list.append(safe_sample(questions.get("basic_mcq_questions", pd.DataFrame()), 2))
            mcq_list.append(safe_sample(questions.get("inter_mcq_questions", pd.DataFrame()), 10))
            mcq_list.append(safe_sample(questions.get("adv_mcq_questions", pd.DataFrame()), 8))
            coding_list.append(safe_sample(questions.get("inter_coding_questions", pd.DataFrame()), 2))

        elif 3 < experience <= 6:
            mcq_list.append(safe_sample(questions.get("inter_mcq_questions", pd.DataFrame()), 8))
            mcq_list.append(safe_sample(questions.get("adv_mcq_questions", pd.DataFrame()), 12))
            coding_list.append(safe_sample(questions.get("inter_coding_questions", pd.DataFrame()), 1))
            coding_list.append(safe_sample(questions.get("adv_coding_questions", pd.DataFrame()), 1))
        
        elif experience > 6:
            mcq_list.append(safe_sample(questions.get("inter_mcq_questions", pd.DataFrame()), 4))
            mcq_list.append(safe_sample(questions.get("adv_mcq_questions", pd.DataFrame()), 16))
            coding_list.append(safe_sample(questions.get("adv_coding_questions", pd.DataFrame()), 2))

        # Combine MCQ lists
        mcq_questions = pd.concat(mcq_list, ignore_index=True) if mcq_list else pd.DataFrame()
        mcq_questions = mcq_questions.sample(frac=1).reset_index(drop=True)

        return mcq_questions.to_dict(orient="records")

    except Exception as e:
        raise ValueError(f"Error in selecting questions: {str(e)}")

class ExperienceRequest(BaseModel):
    experience: int
@app.post("/questions-for-candidate/")
async def get_questions_for_candidates(request: ExperienceRequest):
    """
    Get questions based on candidate experience level.
    """
    try:
        if questions_doc is None:
            raise HTTPException(status_code=500, detail="Questions database not available")
        
        exper = request.experience
        questions = get_doc_questions_in_a_list_of_groups(questions_doc)
        print(questions)
        selected_questions = select_questions_based_on_experience(questions, exper)
        selected_questions_json = json.dumps(selected_questions)
        return {
            "status": "success",
            "questions": selected_questions_json
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



class UserResponse(BaseModel):
    question: str
    answer: str

class QuizData(BaseModel):
    responses: List[UserResponse]

@app.post("/submitQuiz")
async def submit_quiz(quiz_data: QuizData):
    # Now quiz_data is parsed as an object, and you can access responses like this:
    try:
        # Convert to a dict for easier manipulation
        quiz_data_dict = quiz_data.dict()
        print("Received quiz data:", quiz_data.dict())

        # Make sure to log the type of data
        print(f"Type of quiz_data_dict: {type(quiz_data_dict)}")

        # Additional debugging
        for response in quiz_data_dict.get("responses", []):
            print(f"Question: {response['question']}, Answer: {response['answer']}")

        # Load correct answers from the CSV document
        if questions_doc is None:
            raise HTTPException(status_code=500, detail="Questions document not loaded")
        
        correct_answers = questions_doc.set_index("Question")[["Answers", "Level", "Type"]].to_dict(orient="index")

        # Initialize scores and counters
        correct_score = 0
        wrong_score = 0
        unanswered_questions = 0
        total_obtained_marks = 0
        question_results = []

        # Calculate scores based on user responses
        for user_response in quiz_data_dict.get("responses", []):
            question_text = user_response.get("question")
            user_answer = user_response.get("answer")
            print("user_answer",user_answer)
            
            correct_answer_info = correct_answers.get(question_text)

            if correct_answer_info:
                correct_answer = correct_answer_info["Answers"]
                question_type = correct_answer_info["Type"]
                print(correct_answer)

                if user_answer and user_answer.strip():#is not None:
                    if question_type == "MCQ" or question_type == "Coding":
                        if user_answer.strip().lower() == correct_answer.strip().lower():
                            correct_score += 1
                            marks = 1 if question_type == "MCQ" else 5
                            question_results.append({
                                "question": question_text,
                                "correct": True,
                                "wrong": False,
                                "unanswered": False,
                                "marks":marks
                            })
                            total_obtained_marks += marks
                        else:
                            wrong_score += 1
                            question_results.append({
                                "question": question_text,
                                "correct": False,
                                "wrong": True,
                                "unanswered": False,
                                "marks": 0
                            })
                else:
                    unanswered_questions += 1
                    question_results.append({
                        "question": question_text,
                        "correct": False,
                        "wrong": False,
                        "unanswered": True,
                        "marks": 0
                    })

        total_questions =correct_score + wrong_score +unanswered_questions #len(correct_answers)
        total_marks = 20 #30 #5 * total_questions

        quiz_result = {
            "totalQuestions": total_questions,
            "correctAnswers": correct_score,
            "wrongAnswers": wrong_score,
            "unansweredQuestions": unanswered_questions,
            "score": (total_obtained_marks / total_marks) * 100,
            "obtainedMarks": total_obtained_marks,
            "totalMarks": total_marks,
            "questionResults": question_results
        }
        print(correct_score,wrong_score,unanswered_questions)

        # Save quiz result to JSON file
        json_output_path = "./quizresults/quiz_results.json"

        # Append quiz result to JSON file
        if os.path.exists(json_output_path):
            with open(json_output_path, mode='r') as json_file:
                data = json.load(json_file)
            data.append(quiz_result)
        else:
            data = [quiz_result]

        with open(json_output_path, mode='w') as json_file:
            json.dump(data, json_file, indent=4)

        # # Save quiz result to CSV file
        # csv_output_path = "./quizresults/quiz_results.csv"
        # write_header = not os.path.exists(csv_output_path)

        # with open(csv_output_path, mode='a', newline='') as csv_file:
        #     writer = csv.DictWriter(csv_file, fieldnames=quiz_result.keys())

        #     # Write the header if the file is newly created
        #     if write_header:
        #         writer.writeheader()

        #     # Write the quiz result
        #     writer.writerow(quiz_result)

        #     # print("Received quiz result:", quiz_result)

        return {
            "status": "success",
            "quiz_result": quiz_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving quiz data: {str(e)}")

# @app.get("/ok")
# def read_root():
#     return {"message": "Hello, World!"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Default port is 8000 if PORT is not set in the environment
    uvicorn.run(app, host="0.0.0.0", port=port)
